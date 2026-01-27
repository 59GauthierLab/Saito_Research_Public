from dataclasses import dataclass
from pathlib import Path

import config as cfg
import torch
import utils as u
from diagram_utils import export_model_tikz, render_tikz_pdf
from evaluation import evaluate_model
from io_utils import (
    collect_audio_pairs,
    create_spec_dataloader_from_pairs,
    error_msg_exit,
    load_audio_pair,
    save_final_outputs,
    save_snr_metrics,
    select_device,
    split_audio_pairs,
)
from model import UNet2D
from notify_utils import (
    notify_training_outputs,
    send_training_end,
    send_training_images,
    send_info_message,
    send_training_start,
    set_fallback_path,
)
from training import train_model
from training_config_utils import (
    build_training_config,
    format_training_config_lines,
    save_training_config,
)

# ============================================================
# Constants
# ============================================================

WINDOW = torch.hann_window(cfg.WIN_LENGTH)


# ============================================================
# Data structures
# ============================================================


@dataclass(frozen=True)
class PredictionResult:
    """
    各ファイルの推論結果と SNR 指標
    """

    name: str
    output_dir: str
    base_snr: float
    estimated_snr: float
    improvement: float
    base_complex_l1: float
    estimated_complex_l1: float
    complex_reduction: float


@dataclass(frozen=True)
class EvaluationSummary:
    """
    予測全体の平均 SNR 指標
    """

    num_eval: int
    base_snr_avg: float
    estimated_snr_avg: float
    improvement_avg: float
    base_complex_l1_avg: float
    estimated_complex_l1_avg: float
    complex_reduction_avg: float


def prepare_training_data() -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    torch.utils.data.DataLoader,
]:
    """
    データセットのペアを収集して train/val に分割する
    """
    all_pairs = collect_audio_pairs(cfg.DATASET_INPUT_DIR, cfg.DATASET_TEACHER_DIR)
    train_pairs, val_pairs = split_audio_pairs(
        all_pairs, cfg.VALIDATION_RATIO, cfg.SPLIT_SEED
    )
    if not train_pairs:
        error_msg_exit("no training pairs available after split")

    train_loader, _ = create_spec_dataloader_from_pairs(
        train_pairs,
        sample_rate=cfg.SR_RATE,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH,
        window=WINDOW,
        seg_frames=cfg.SEG_FRAMES,
        overlap=cfg.OVERLAP_RATIO,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        use_hilbert_features=cfg.USE_HILBERT_FEATURES,
    )
    return train_pairs, val_pairs, train_loader


def build_model() -> tuple[UNet2D, int]:
    """
    モデルを初期化し、パラメータ数を返す
    """
    freq_bins = cfg.N_FFT // 2 + 1
    model = UNet2D(
        in_ch=cfg.MODEL_IN_CH,
        base=cfg.MODEL_BASE,
        freq_bins=freq_bins,
        bottleneck_rnn_enabled=cfg.BOTTLENECK_RNN_ENABLED,
        bottleneck_rnn_channels=cfg.BOTTLENECK_RNN_CHANNELS,
        bottleneck_rnn_hidden=cfg.BOTTLENECK_RNN_HIDDEN,
        bottleneck_rnn_layers=cfg.BOTTLENECK_RNN_LAYERS,
        bottleneck_rnn_dropout=cfg.BOTTLENECK_RNN_DROPOUT,
    )
    model_parameters = sum(p.numel() for p in model.parameters())
    return model, model_parameters


def notify_training_start(
    device: torch.device,
    model_parameters: int,
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
) -> None:
    """
    学習設定を保存し、開始通知を送る
    """
    training_config = build_training_config(
        model_name=cfg.MODEL_NAME,
        model_in_ch=cfg.MODEL_IN_CH,
        model_base=cfg.MODEL_BASE,
        use_hilbert_features=cfg.USE_HILBERT_FEATURES,
        bottleneck_rnn_enabled=cfg.BOTTLENECK_RNN_ENABLED,
        bottleneck_rnn_channels=cfg.BOTTLENECK_RNN_CHANNELS,
        bottleneck_rnn_hidden=cfg.BOTTLENECK_RNN_HIDDEN,
        bottleneck_rnn_layers=cfg.BOTTLENECK_RNN_LAYERS,
        bottleneck_rnn_dropout=cfg.BOTTLENECK_RNN_DROPOUT,
        output_dir=cfg.OUTPUT_DIR,
        sample_rate=cfg.SR_RATE,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH,
        seg_frames=cfg.SEG_FRAMES,
        overlap_ratio=cfg.OVERLAP_RATIO,
        batch_size=cfg.BATCH_SIZE,
        num_epochs=cfg.NUM_EPOCHS,
        learning_rate=cfg.LEARNING_RATE,
        mag_weight=cfg.MAG_WEIGHT,
        snr_weight=cfg.SNR_WEIGHT,
        log_weight=cfg.LOG_WEIGHT,
        sil_weight=cfg.SIL_WEIGHT,
        sil_alpha=cfg.SIL_ALPHA,
        device=str(device),
        dataset_input_dir=cfg.DATASET_INPUT_DIR,
        dataset_teacher_dir=cfg.DATASET_TEACHER_DIR,
        validation_ratio=cfg.VALIDATION_RATIO,
        split_seed=cfg.SPLIT_SEED,
        train_count=len(train_pairs),
        val_count=len(val_pairs),
    )
    save_training_config(cfg.OUTPUT_DIR, training_config)
    start_lines = [
        f"Using device: {device}",
        f"Model parameters: {model_parameters}",
        *format_training_config_lines(training_config),
    ]
    send_training_start(start_lines)


def build_sample_input() -> torch.Tensor:
    """
    モデル図・要約用のダミー入力を作成する
    """
    freq_bins = cfg.N_FFT // 2 + 1
    return torch.zeros(
        1, cfg.MODEL_IN_CH, freq_bins, cfg.SEG_FRAMES, dtype=torch.float32
    )


def save_model_diagram(model: UNet2D) -> Path:
    """
    pytorch2tikz でモデル図を描画して保存する
    """
    sample_input = build_sample_input()
    diagram_dir = Path(cfg.OUTPUT_DIR) / "diagram"
    tex_path = export_model_tikz(model, sample_input, diagram_dir)
    render_tikz_pdf(tex_path)
    return diagram_dir


def save_model_summary(model: UNet2D) -> str:
    """
    テキストベースのモデル要約を保存する
    """
    diagram_dir = Path(cfg.OUTPUT_DIR) / "diagram"
    diagram_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torchinfo import summary
    except ImportError:
        error_msg_exit("torchinfo is required to export model summary")

    orig_device = next(model.parameters()).device
    model_cpu = model.to("cpu")
    model_cpu.eval()

    sample_input = build_sample_input()
    summary_text = str(
        summary(
            model,
            input_data=sample_input,
            depth=4,
            col_names=("input_size", "output_size", "num_params", "kernel_size"),
            row_settings=("var_names",),
        )
    )

    model.to(orig_device)

    summary_path = diagram_dir / "model_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    return summary_text


def format_codeblock(title: str, body: str) -> str:
    """
    Discord 送信用のコードブロック文字列を整形する
    """
    return f"**{title}**\n```\n{body}\n```"


def notify_model_diagram(model: UNet2D) -> None:
    """
    モデル図を生成して学習開始時に通知する
    """
    diagram_dir = save_model_diagram(model)
    send_training_images(output_dir=str(diagram_dir), message="# 🧩 Model Diagram")
    summary_text = save_model_summary(model)
    summary_text = summary_text[:1800] + ("\n..." if len(summary_text) > 1800 else "")
    send_info_message(format_codeblock("Model Summary", summary_text))


def train_with_notifications(
    model: UNet2D,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[float], UNet2D]:
    """
    学習を実行し、共通の学習曲線画像を通知する
    """
    losses, _learning_rates, model = train_model(
        output_dir=cfg.OUTPUT_DIR,
        model=model,
        train_loader=train_loader,
        num_epochs=cfg.NUM_EPOCHS,
        learning_rate=cfg.LEARNING_RATE,
        snr_weight=cfg.SNR_WEIGHT,
        device=device,
    )
    training_dir = str(Path(cfg.OUTPUT_DIR) / "training")
    send_training_images(output_dir=training_dir, message="# ✅ Training Finished")
    return losses, model


def evaluate_predictions(
    model: UNet2D,
    device: torch.device,
    eval_pairs: list[tuple[str, str]],
) -> tuple[list[PredictionResult], EvaluationSummary]:
    """
    推論を実行し、ファイルごとの出力と指標をまとめる
    """
    predicted_root = Path(cfg.OUTPUT_DIR) / "predicted"
    predicted_root.mkdir(parents=True, exist_ok=True)

    base_snr_sum = 0.0
    estimated_snr_sum = 0.0
    base_complex_sum = 0.0
    estimated_complex_sum = 0.0
    results: list[PredictionResult] = []

    for idx, (input_path, teacher_path) in enumerate(eval_pairs):
        eval_source, eval_target = load_audio_pair(
            input_path, teacher_path, sample_rate=cfg.SR_RATE
        )

        estimated_audio, source_spec, target_spec, estimated_spec = evaluate_model(
            model.to(device),
            eval_source,
            eval_target,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH,
            win_length=cfg.WIN_LENGTH,
            window=WINDOW,
            seg_frames=cfg.SEG_FRAMES,
            overlap_ratio=cfg.OVERLAP_RATIO,
            use_hilbert_features=cfg.USE_HILBERT_FEATURES,
            device=device,
        )

        base_snr = u.snr(eval_target, eval_source)
        estimated_snr = u.snr(eval_target, estimated_audio)
        improvement = estimated_snr - base_snr
        base_snr_sum += base_snr
        estimated_snr_sum += estimated_snr

        base_complex_l1 = u.complex_reim_l1(source_spec, target_spec)
        estimated_complex_l1 = u.complex_reim_l1(estimated_spec, target_spec)
        complex_reduction = base_complex_l1 - estimated_complex_l1
        base_complex_sum += base_complex_l1
        estimated_complex_sum += estimated_complex_l1

        output_name = f"{idx:03d}_{Path(input_path).stem}"
        predicted_dir = predicted_root / output_name
        predicted_dir.mkdir(parents=True, exist_ok=True)
        save_final_outputs(
            output_dir=str(predicted_dir),
            sample_rate=cfg.SR_RATE,
            source_audio=eval_source,
            target_audio=eval_target,
            estimated_audio=estimated_audio,
            source_spec=source_spec,
            target_spec=target_spec,
            estimated_spec=estimated_spec,
        )

        save_snr_metrics(
            output_dir=str(predicted_dir),
            target_audio=eval_target,
            source_audio=eval_source,
            estimated_audio=estimated_audio,
            source_spec=source_spec,
            target_spec=target_spec,
            estimated_spec=estimated_spec,
        )

        results.append(
            PredictionResult(
                name=Path(input_path).name,
                output_dir=str(predicted_dir),
                base_snr=base_snr,
                estimated_snr=estimated_snr,
                improvement=improvement,
                base_complex_l1=base_complex_l1,
                estimated_complex_l1=estimated_complex_l1,
                complex_reduction=complex_reduction,
            )
        )

    num_eval = len(eval_pairs)
    summary = EvaluationSummary(
        num_eval=num_eval,
        base_snr_avg=base_snr_sum / num_eval,
        estimated_snr_avg=estimated_snr_sum / num_eval,
        improvement_avg=(estimated_snr_sum - base_snr_sum) / num_eval,
        base_complex_l1_avg=base_complex_sum / num_eval,
        estimated_complex_l1_avg=estimated_complex_sum / num_eval,
        complex_reduction_avg=(base_complex_sum - estimated_complex_sum) / num_eval,
    )
    return results, summary


def save_metrics_summary(summary: EvaluationSummary) -> None:
    """
    複数ファイル時のみ平均 SNR を出力
    """
    if summary.num_eval <= 1:
        return
    summary_path = Path(cfg.OUTPUT_DIR) / "predicted" / "metrics_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"eval pairs: {summary.num_eval}\n")
        f.write(f"snr base avg: {summary.base_snr_avg:.3f} dB\n")
        f.write(f"snr estimated avg: {summary.estimated_snr_avg:.3f} dB\n")
        f.write(f"snr improvement avg: {summary.improvement_avg:.3f} dB\n")
        f.write(f"complexl1 base avg: {summary.base_complex_l1_avg:.6f}\n")
        f.write(f"complexl1 estimated avg: {summary.estimated_complex_l1_avg:.6f}\n")
        f.write(f"complexl1 reduction avg: {summary.complex_reduction_avg:.6f}\n")


def notify_prediction_results(results: list[PredictionResult]) -> None:
    """
    予測結果をファイルごとに通知する
    """
    for result in results:
        result_text = "\n".join(
            [
                f"snr base: {result.base_snr:.3f} dB",
                f"snr estimated: {result.estimated_snr:.3f} dB",
                f"snr improvement: {result.improvement:.3f} dB",
                f"complexl1 base: {result.base_complex_l1:.6f}",
                f"complexl1 estimated: {result.estimated_complex_l1:.6f}",
                f"complexl1 reduction: {result.complex_reduction:.6f}",
            ]
        )
        notify_training_outputs(
            output_dir=result.output_dir,
            result_text=result_text,
            message=f"## Prediction: {result.name}",
        )


def main() -> None:
    """
    学習〜推論までのパイプラインを実行する
    """
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    set_fallback_path(Path(cfg.OUTPUT_DIR) / "notify_fallback.log")
    device = select_device()

    # データ準備とモデル初期化
    train_pairs, val_pairs, train_loader = prepare_training_data()
    model, model_parameters = build_model()
    notify_training_start(device, model_parameters, train_pairs, val_pairs)
    notify_model_diagram(model)

    # 学習と共通通知
    losses, model = train_with_notifications(model, train_loader, device)

    # 推論と結果保存
    eval_pairs = val_pairs or train_pairs
    results, summary = evaluate_predictions(model, device, eval_pairs)
    save_metrics_summary(summary)

    # 全体通知 + ファイル別通知
    send_training_end(final_loss=losses[-1], output_dir=cfg.OUTPUT_DIR)
    notify_prediction_results(results)


if __name__ == "__main__":
    main()
