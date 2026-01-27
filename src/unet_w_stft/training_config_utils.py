import json
from pathlib import Path
from typing import Any

# ============================================================
# Training config helpers
# ============================================================


def build_training_config(
    model_name: str,
    model_in_ch: int,
    model_base: int,
    use_hilbert_features: bool,
    bottleneck_rnn_enabled: bool,
    bottleneck_rnn_channels: int,
    bottleneck_rnn_hidden: int,
    bottleneck_rnn_layers: int,
    bottleneck_rnn_dropout: float,
    output_dir: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    seg_frames: int,
    overlap_ratio: float,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    mag_weight: float,
    snr_weight: float,
    log_weight: float,
    sil_weight: float,
    sil_alpha: float,
    device: str,
    dataset_input_dir: str,
    dataset_teacher_dir: str,
    validation_ratio: float,
    split_seed: int,
    train_count: int,
    val_count: int,
) -> dict[str, Any]:
    """
    学習設定の辞書を組み立てる
    """
    config_path = str(Path(output_dir) / "training_config.json")
    config: dict[str, Any] = {
        "model": {
            "name": model_name,
            "in_ch": model_in_ch,
            "base": model_base,
            "hilbert_features": use_hilbert_features,
            "bottleneck_rnn": {
                "enabled": bottleneck_rnn_enabled,
                "channels": bottleneck_rnn_channels,
                "hidden": bottleneck_rnn_hidden,
                "layers": bottleneck_rnn_layers,
                "dropout": bottleneck_rnn_dropout,
            },
        },
        "paths": {
            "output_dir": output_dir,
            "config_file": config_path,
        },
        "audio": {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "seg_frames": seg_frames,
            "overlap_ratio": overlap_ratio,
        },
        "training": {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "mag_weight": mag_weight,
            "snr_weight": snr_weight,
            "log_weight": log_weight,
            "sil_weight": sil_weight,
            "sil_alpha": sil_alpha,
        },
        "runtime": {
            "device": device,
        },
        "dataset": {
            "input_dir": dataset_input_dir,
            "teacher_dir": dataset_teacher_dir,
            "validation_ratio": validation_ratio,
            "split_seed": split_seed,
            "train_count": train_count,
            "val_count": val_count,
        },
    }

    return config


def save_training_config(output_dir: str, config: dict[str, Any]) -> Path:
    """
    学習設定を JSON で保存する
    """
    path = Path(output_dir) / "training_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def format_training_config_lines(config: dict[str, Any]) -> list[str]:
    """
    通知向けに設定内容をフラットな文字列へ整形する
    """
    lines: list[str] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "paths" and not prefix:
                    continue
                walk(f"{prefix}{key}.", item)
            return
        key = prefix[:-1] if prefix.endswith(".") else prefix
        lines.append(f"{key}: {value}")

    walk("", config)
    return lines
