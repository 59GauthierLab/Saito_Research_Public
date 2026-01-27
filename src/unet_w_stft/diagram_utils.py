from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import torch
from io_utils import error_msg_exit


def _patch_pytorch2tikz_image_save() -> None:
    try:
        import pytorch2tikz.block.factory as factory
    except ImportError:
        return

    def safe_save_image(tensor, path: str) -> None:
        # Skip writing PNGs; the input block is replaced with a placeholder.
        return

    factory.save_image = safe_save_image


def _ensure_standalone(tex_body: str) -> str:
    if "\\documentclass" in tex_body or "\\begin{document}" in tex_body:
        return tex_body
    return (
        "\\documentclass[tikz,border=2pt]{standalone}\n"
        "\\usepackage{tikz}\n"
        "\\begin{document}\n"
        f"{tex_body}\n"
        "\\end{document}\n"
    )


def _sanitize_tex(tex_body: str) -> str:
    tex_body = re.sub(r"np\.(?:float|int)\d+\(([^)]+)\)", r"\1", tex_body)
    if "\\def\\UpColor" not in tex_body:
        tex_body = re.sub(
            r"(\\def\\ConvColor\{[^}]*\})",
            r"\1\n\\def\\UpColor{rgb,255:red,3;green,189;blue,186}",
            tex_body,
            count=1,
        )
    tex_body = re.sub(
        r"\\def\\PoolColor\{[^}]*\}",
        r"\\def\\PoolColor{rgb,255:red,0;green,137;blue,241}",
        tex_body,
    )
    tex_body = re.sub(
        r"^.*(ImgInput_|input_\\d+\\.png).*(?:\\n)?",
        "",
        tex_body,
        flags=re.MULTILINE,
    )

    def shrink_large_depth(match: re.Match[str]) -> str:
        indent = match.group(1)
        value = float(match.group(2))
        new_value = max(value * 0.18, 20.0)
        return f"{indent}depth={new_value:.2f},\n{indent}zlabel=8208"

    tex_body = re.sub(
        r"([ \t]*)depth=([0-9.]+),[ \t]*\n[ \t]*zlabel=8208",
        shrink_large_depth,
        tex_body,
    )

    return tex_body


def _add_skip_connections(arch, model) -> None:
    seq = arch._block_sequence
    try:
        from pytorch2tikz.block.connections import LoopConnection
    except ImportError:
        LoopConnection = None
    pairs = [("enc3", "dec3"), ("enc2", "dec2"), ("enc1", "dec1")]
    for enc_name, dec_name in pairs:
        enc = getattr(model, enc_name, None)
        dec = getattr(model, dec_name, None)
        if enc is None or dec is None:
            continue
        enc_mod = enc[-1] if hasattr(enc, "__getitem__") else enc
        dec_mod = dec[0] if hasattr(dec, "__getitem__") else dec
        enc_block = seq._seen_modules.get(enc_mod)
        dec_block = seq._seen_modules.get(dec_mod)
        if enc_block is None or dec_block is None:
            continue
        if LoopConnection is None:
            seq.connect(enc_block, dec_block)
        else:
            seq.connect(enc_block, dec_block, conn_type=LoopConnection)


def _style_upsample_blocks(arch, model) -> None:
    seq = arch._block_sequence
    up_map = {}
    for name in ("up3", "up2", "up1"):
        mod = getattr(model, name, None)
        if mod is not None:
            up_map[mod] = name.replace("up", "Up")
    if not up_map:
        return
    for module, block in seq._seen_modules.items():
        if module in up_map:
            block.args["fill"] = r"\UpColor"
            block.args["caption"] = up_map[module]
            block.args["opacity"] = 0.9


def _label_norm_activation_blocks(arch) -> list[tuple[str, str]]:
    seq = arch._block_sequence
    labels: list[tuple[str, str]] = []
    for module, block in seq._seen_modules.items():
        if block is None:
            continue
        label = None
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            label = "BN"
        elif isinstance(module, torch.nn.LeakyReLU):
            label = "LReLU"
        elif isinstance(module, torch.nn.ReLU):
            label = "ReLU"
        elif isinstance(module, torch.nn.Tanh):
            label = "Tanh"
        elif isinstance(module, torch.nn.Sigmoid):
            label = "Sigmoid"
        elif isinstance(module, torch.nn.GELU):
            label = "GELU"
        elif isinstance(module, torch.nn.SiLU):
            label = "SiLU"
        elif isinstance(
            module,
            (torch.nn.modules.pooling._MaxPoolNd, torch.nn.modules.pooling._AvgPoolNd),
        ):
            label = "Pool"
        if label:
            block.args["caption"] = ""
            block.args["opacity"] = 0.9
            labels.append((block.name, label))
    return labels


def _inject_legend(tex_body: str, labels: list[tuple[str, str]]) -> str:
    return tex_body


def export_model_tikz(
    model, sample_input, diagram_dir: Path, name: str = "unet_w_stft"
) -> Path:
    try:
        from pytorch2tikz import Architecture
        from pytorch2tikz.constants import COLOR_VALUES
    except ImportError:
        error_msg_exit("pytorch2tikz is required to export model tikz")

    _patch_pytorch2tikz_image_save()

    diagram_dir.mkdir(parents=True, exist_ok=True)
    tex_path = diagram_dir / f"{name}.tex"

    image_path = str(diagram_dir / "input_{i}.png")
    colors = dict(COLOR_VALUES)
    colors["POOL"] = "#0089F1"  # Custom color for pooling layers
    arch = Architecture(
        model,
        image_path=image_path,
        ignore_layers=["flatten"],
        colors=colors,
    )
    arch._block_sequence._fuseable_layers = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)
    if was_training:
        model.train()
    arch.remove_handles()
    _add_skip_connections(arch, model)
    _style_upsample_blocks(arch, model)
    inner_labels = _label_norm_activation_blocks(arch)

    tex_body = _sanitize_tex(arch.get_tex())
    tex_body = _inject_legend(tex_body, inner_labels)
    tex_path.write_text(_ensure_standalone(tex_body), encoding="utf-8")

    return tex_path


def render_tikz_pdf(tex_path: Path) -> Path:
    if not shutil.which("pdflatex"):
        error_msg_exit("pdflatex not found; install TeX or update PATH")

    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    run = subprocess.run(
        cmd, cwd=str(tex_path.parent), check=False, capture_output=True, text=True
    )
    if run.returncode != 0:
        tail = (run.stdout + "\n" + run.stderr).strip().splitlines()[-20:]
        tail_text = "\n".join(tail) if tail else "(no pdflatex output)"
        error_msg_exit("pdflatex failed while rendering tikz diagram:\n" + tail_text)

    return tex_path.with_suffix(".pdf")
