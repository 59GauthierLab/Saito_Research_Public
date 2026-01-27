import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import utils as u
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 汎用メッセージ
# ============================================================


def error_msg_exit(message: str) -> None:
    """
    エラーメッセージを表示して終了
    """
    print(f"[ERROR] {message}")
    exit(1)


def info_msg(message: str) -> None:
    """
    デバッグ用・処理確認用にメッセージを表示
    """
    print(f"[INFO] {message}")


# ============================================================
# デバイス・音声周りのヘルパー
# ============================================================


def select_device() -> torch.device:
    """
    GPU/MPS/CPU を選択
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    最大値で正規化
    """
    return audio / max(np.max(np.abs(audio)), 1e-9)


def load_audio_pair(
    input_path: str, teacher_path: str, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    入力と教師の音声を読み込み，長さを揃えて返す
    """
    input_audio, _ = u.wav_load(input_path, target_sr=sample_rate)
    teacher_audio, _ = u.wav_load(teacher_path, target_sr=sample_rate)

    if len(input_audio) != len(teacher_audio):
        error_msg_exit(
            "input/teacher length mismatch: "
            f"{len(input_audio)} vs {len(teacher_audio)}"
        )
    input_audio = normalize_audio(input_audio)
    teacher_audio = normalize_audio(teacher_audio)

    return input_audio, teacher_audio


# ============================================================
# データセット関連
# ============================================================


def collect_audio_pairs(
    input_dir: str, teacher_dir: str, extension: str = ".wav"
) -> list[tuple[str, str]]:
    """
    入力と教師のディレクトリから同名ファイルのペアを収集
    """
    input_path = Path(input_dir)
    teacher_path = Path(teacher_dir)
    if not input_path.exists():
        error_msg_exit(f"input dir not found: {input_dir}")
    if not teacher_path.exists():
        error_msg_exit(f"teacher dir not found: {teacher_dir}")

    input_files = sorted(input_path.glob(f"*{extension}"))
    teacher_files = sorted(teacher_path.glob(f"*{extension}"))
    if not input_files:
        error_msg_exit(f"no input files found in {input_dir}")
    if not teacher_files:
        error_msg_exit(f"no teacher files found in {teacher_dir}")

    # 入力/教師で同名の wav が揃っているか確認
    teacher_map = {path.name: path for path in teacher_files}
    input_names = {path.name for path in input_files}
    missing_teacher = [
        path.name for path in input_files if path.name not in teacher_map
    ]
    missing_input = [
        path.name for path in teacher_files if path.name not in input_names
    ]
    if missing_teacher:
        error_msg_exit(
            "missing teacher files: " + ", ".join(sorted(missing_teacher)[:5])
        )
    if missing_input:
        error_msg_exit("missing input files: " + ", ".join(sorted(missing_input)[:5]))

    pairs = [(str(path), str(teacher_map[path.name])) for path in input_files]
    return pairs


def split_audio_pairs(
    pairs: list[tuple[str, str]], validation_ratio: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    ペアを train / val に分割
    """
    if not 0 <= validation_ratio < 1:
        error_msg_exit(f"validation_ratio must be in [0, 1): {validation_ratio}")
    if len(pairs) < 2 or validation_ratio == 0:
        return pairs, []

    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    val_size = int(round(len(pairs) * validation_ratio))
    val_size = max(1, val_size)
    val_size = min(val_size, len(pairs) - 1)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    return train_pairs, val_pairs


def create_spec_dataloader_from_pairs(
    pairs: list[tuple[str, str]],
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    seg_frames: int,
    overlap: float,
    batch_size: int,
    shuffle: bool,
    use_hilbert_features: bool,
) -> tuple[DataLoader, dict[str, int]]:
    """
    複数ファイルからスペクトル分割して DataLoader を作成
    """
    if not pairs:
        error_msg_exit("no audio pairs provided")

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    meta: dict[str, int] = {}

    for input_path, teacher_path in pairs:
        source_audio, target_audio = load_audio_pair(
            input_path, teacher_path, sample_rate=sample_rate
        )
        src_features, src_spec = u.build_input_features(
            source_audio,
            n_fft,
            hop_length,
            win_length,
            window,
            use_hilbert_features=use_hilbert_features,
        )
        tgt_spec = u.stft_torch(target_audio, n_fft, hop_length, win_length, window)
        X, Y, hop = u.build_spec_segments(
            src_features, tgt_spec, seg_frames=seg_frames, overlap=overlap
        )
        xs.append(X.cpu())
        ys.append(Y.cpu())

        if not meta:
            meta = {"F": src_spec.shape[0], "T": src_spec.shape[1], "hop_frames": hop}

    X_all = torch.cat(xs, dim=0)
    Y_all = torch.cat(ys, dim=0)
    meta["num_segments"] = X_all.shape[0]
    meta["num_pairs"] = len(pairs)
    dl = DataLoader(TensorDataset(X_all, Y_all), batch_size=batch_size, shuffle=shuffle)
    return dl, meta


# ============================================================
# 推論結果の保存
# ============================================================


def save_final_outputs(
    output_dir: str,
    sample_rate: int,
    source_audio: np.ndarray,
    target_audio: np.ndarray,
    estimated_audio: np.ndarray,
    source_spec: torch.Tensor,
    target_spec: torch.Tensor,
    estimated_spec: torch.Tensor,
) -> None:
    """
    推論結果の保存と可視化
    """
    # --- 保存（音） ---
    sf.write(os.path.join(output_dir, "source.wav"), source_audio, sample_rate)
    sf.write(os.path.join(output_dir, "target.wav"), target_audio, sample_rate)
    sf.write(os.path.join(output_dir, "predicted.wav"), estimated_audio, sample_rate)

    # --- 可視化（時間波形 / スペクトル / ヒスト） ---
    u.save_timeplots(
        source_audio,
        target_audio,
        estimated_audio,
        sample_rate,
        os.path.join(output_dir, "final"),
    )
    u.save_specplots(
        source_spec,
        target_spec,
        estimated_spec,
        os.path.join(output_dir, "final"),
    )
    u.save_hist(
        source_audio,
        target_audio,
        estimated_audio,
        os.path.join(output_dir, "amplitude_hist.png"),
    )


def save_snr_metrics(
    output_dir: str,
    target_audio: np.ndarray,
    source_audio: np.ndarray,
    estimated_audio: np.ndarray,
    source_spec: torch.Tensor | None = None,
    target_spec: torch.Tensor | None = None,
    estimated_spec: torch.Tensor | None = None,
) -> None:
    """
    SNR 改善量を保存（必要に応じて複素スペクトル損失も併記）
    """
    base_snr = u.snr(target_audio, source_audio)
    estimated_snr = u.snr(target_audio, estimated_audio)
    improvement = estimated_snr - base_snr

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"SNR(src -> tgt): {base_snr:.3f} dB\n")
        f.write(f"SNR(est  -> tgt): {estimated_snr:.3f} dB\n")
        f.write(f"Improvement: {improvement:.3f} dB\n")

        if (
            source_spec is not None
            and target_spec is not None
            and estimated_spec is not None
        ):
            base_complex = u.complex_reim_l1(source_spec, target_spec)
            estimated_complex = u.complex_reim_l1(estimated_spec, target_spec)
            reduction = base_complex - estimated_complex
            f.write(f"ComplexL1(src -> tgt): {base_complex:.6f}\n")
            f.write(f"ComplexL1(est -> tgt): {estimated_complex:.6f}\n")
            f.write(f"Reduction: {reduction:.6f}\n")

            base_sil = u.silence_residual_metrics(source_spec, target_spec)
            est_sil = u.silence_residual_metrics(
                estimated_spec,
                target_spec,
                tgt_mag_threshold=base_sil["tgt_mag_threshold"],
            )
            f.write(
                "Silence mask (|S(tgt)| <= thr): "
                f"thr={base_sil['tgt_mag_threshold']:.6e}, "
                f"ratio={base_sil['silence_mask_ratio']:.3f}\n"
            )
            f.write(
                "Silence residual (src): "
                f"mean|S|={base_sil['silence_est_mean_abs']:.6e} "
                f"({base_sil['silence_est_mean_abs_db']:.2f} dB), "
                f"mean|S|^2={base_sil['silence_est_mean_pow']:.6e} "
                f"({base_sil['silence_est_mean_pow_db']:.2f} dB)\n"
            )
            f.write(
                "Silence residual (est): "
                f"mean|S|={est_sil['silence_est_mean_abs']:.6e} "
                f"({est_sil['silence_est_mean_abs_db']:.2f} dB), "
                f"mean|S|^2={est_sil['silence_est_mean_pow']:.6e} "
                f"({est_sil['silence_est_mean_pow_db']:.2f} dB)\n"
            )
            f.write(
                "Silence reduction (est vs src): "
                f"{(base_sil['silence_est_mean_pow_db'] - est_sil['silence_est_mean_pow_db']):.2f} dB "  # noqa: E501
                "(power)\n"
            )
