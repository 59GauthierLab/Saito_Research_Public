import base64
import json
import mimetypes
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Constants
# ============================================================

DEFAULT_BASE_URL = os.environ.get("DISCORDBOT_API_BASE", "")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
DEFAULT_FALLBACK_PATH = Path.cwd() / "notify_fallback.log"


def set_fallback_path(path: Path) -> None:
    """
    通知失敗時のログ保存先を更新
    """
    global DEFAULT_FALLBACK_PATH
    DEFAULT_FALLBACK_PATH = path


def _local_info(message: str) -> None:
    """
    ローカル出力用の簡易ログ
    """
    print(f"[INFO] {message}")


def _post_json(endpoint: str, payload: dict[str, Any], base_url: str) -> None:
    """
    Discord Bot API に JSON を POST する
    """
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    data = json.dumps(payload).encode("utf-8")
    request_obj = request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with request.urlopen(request_obj, timeout=30) as response:
        response.read()


def _encode_image(image_path: Path) -> dict[str, str]:
    """
    画像ファイルを Base64 にエンコード
    """
    content_type, _ = mimetypes.guess_type(image_path.name)
    if not content_type:
        content_type = "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "filename": image_path.name,
        "contentType": content_type,
        "data": encoded,
    }


def _convert_pdf_to_jpg(pdf_path: Path) -> Path | None:
    """
    PDF を JPG に変換する（macOS の sips を優先）
    """
    if not pdf_path.exists():
        return None
    jpg_path = pdf_path.with_suffix(".jpg")
    if jpg_path.exists() and jpg_path.stat().st_mtime >= pdf_path.stat().st_mtime:
        return jpg_path

    if shutil.which("sips"):
        cmd = ["sips", "-s", "format", "jpeg", str(pdf_path), "--out", str(jpg_path)]
    elif shutil.which("magick"):
        cmd = ["magick", str(pdf_path), str(jpg_path)]
    elif shutil.which("convert"):
        cmd = ["convert", str(pdf_path), str(jpg_path)]
    elif shutil.which("pdftocairo"):
        cmd = ["pdftocairo", "-jpeg", str(pdf_path), str(jpg_path.with_suffix(""))]
    else:
        return None

    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        return None
    if jpg_path.exists():
        return jpg_path
    fallback = pdf_path.with_name(pdf_path.stem + "-1.jpg")
    if fallback.exists():
        return fallback
    return None


def _should_skip_image(image_path: Path) -> bool:
    """
    送信対象から除外する画像パターンを判定
    """
    name = image_path.name
    return name.startswith("epoch_") and name.endswith("_spec.png")


def _append_fallback_log(fallback_path: Path, message: str) -> None:
    """
    通知失敗時のログを追記
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    with fallback_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _format_markdown(title: str, lines: list[str]) -> str:
    """
    Markdown 形式のメッセージを生成
    """
    header = f"**{title}**"
    if not lines:
        return header
    return header + "\n" + "\n".join(f"- {line}" for line in lines)


def _format_codeblock(title: str, lines: list[str]) -> str:
    """
    コードブロック形式のメッセージを生成
    """
    header = f"**{title}**"
    if not lines:
        return header
    body = "\n".join(lines)
    return f"{header}\n```\n{body}\n```"


def send_info_message(
    text: str,
    base_url: str = DEFAULT_BASE_URL,
    fallback_path: Path | None = None,
) -> None:
    """
    テキストメッセージを通知
    """
    fallback = fallback_path or DEFAULT_FALLBACK_PATH
    if not base_url:
        msg = "DISCORDBOT_API_BASE is not set; skip sending info message."
        _local_info(msg)
        _append_fallback_log(fallback, msg)
        return
    payload = {"text": text}
    try:
        _post_json("/training/result", payload, base_url)
    except Exception as exc:
        msg = f"Failed to send info message: {exc}"
        _local_info(msg)
        _append_fallback_log(fallback, msg)


def send_training_start(
    lines: list[str],
    base_url: str = DEFAULT_BASE_URL,
    fallback_path: Path | None = None,
) -> None:
    """
    学習開始メッセージを送信
    """
    send_info_message(
        _format_codeblock("# 🚀 Training Started", lines),
        base_url=base_url,
        fallback_path=fallback_path,
    )


def send_training_end(
    final_loss: float,
    output_dir: str,
    base_url: str = DEFAULT_BASE_URL,
    fallback_path: Path | None = None,
) -> None:
    """
    学習終了メッセージを送信
    """
    lines = [f"final loss: `{final_loss:.6f}`"]
    lines.append(f"output: `{output_dir}`")
    send_info_message(
        _format_markdown("## Summary", lines),
        base_url=base_url,
        fallback_path=fallback_path,
    )


def collect_training_images(output_dir: str) -> list[dict[str, str]]:
    """
    通知対象の画像を収集してエンコード
    """
    image_paths: list[Path] = []
    for path in sorted(Path(output_dir).rglob("*")):
        if not path.is_file() or _should_skip_image(path):
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            image_paths.append(path)
            continue
        if suffix == ".pdf":
            jpg_path = _convert_pdf_to_jpg(path)
            if jpg_path:
                image_paths.append(jpg_path)
    return [_encode_image(path) for path in image_paths]


def send_training_images(
    output_dir: str,
    message: str = "training images",
    base_url: str = DEFAULT_BASE_URL,
    fallback_path: Path | None = None,
) -> None:
    """
    画像一覧を通知する
    """
    fallback = fallback_path or DEFAULT_FALLBACK_PATH
    if not base_url:
        msg = "DISCORDBOT_API_BASE is not set; skip sending training images."
        _local_info(msg)
        _append_fallback_log(fallback, msg)
        return
    images = collect_training_images(output_dir)
    if not images:
        msg = f"No images found to send in {output_dir}."
        _local_info(msg)
        _append_fallback_log(fallback, msg)
        return
    payload = {"message": message, "images": images}
    try:
        _post_json("/training/images", payload, base_url)
        _local_info(f"Sent {len(images)} training images to {base_url}.")
    except Exception as exc:
        msg = f"Failed to send training images: {exc}"
        _local_info(msg)
        _append_fallback_log(fallback, msg)


def send_training_result(
    text: str,
    base_url: str = DEFAULT_BASE_URL,
    fallback_path: Path | None = None,
) -> None:
    """
    結果テキストを通知する
    """
    fallback = fallback_path or DEFAULT_FALLBACK_PATH
    if not base_url:
        msg = "DISCORDBOT_API_BASE is not set; skip sending training result."
        _local_info(msg)
        _append_fallback_log(fallback, msg)
        return
    payload = {"text": text}
    try:
        _post_json("/training/result", payload, base_url)
        _local_info(f"Sent training result to {base_url}.")
    except Exception as exc:
        msg = f"Failed to send training result: {exc}"
        _local_info(msg)
        _append_fallback_log(fallback, msg)


def notify_training_outputs(
    output_dir: str,
    result_text: str,
    message: str = "training images",
    base_url: str = DEFAULT_BASE_URL,
) -> None:
    """
    画像と結果テキストをまとめて通知する
    """
    fallback_path = Path(output_dir) / "notify_fallback.log"
    set_fallback_path(fallback_path)
    send_training_images(
        output_dir=output_dir,
        message=message,
        base_url=base_url,
        fallback_path=fallback_path,
    )
    send_training_result(
        text=result_text,
        base_url=base_url,
        fallback_path=fallback_path,
    )
