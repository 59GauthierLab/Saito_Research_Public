import numpy as np
import torch
import utils as u
from model import UNet2D

# ============================================================
# Evaluation
# ============================================================


@torch.no_grad()
def evaluate_model(
    model: UNet2D,
    source_audio: np.ndarray,
    target_audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    seg_frames: int,
    overlap_ratio: float,
    use_hilbert_features: bool,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    STFT ベースで推論を行い，推定波形とスペクトルを返す
    """
    model.eval()
    # 入力・教師の STFT を作成
    source_features, source_spec_full = u.build_input_features(
        source_audio,
        n_fft,
        hop_length,
        win_length,
        window,
        use_hilbert_features=use_hilbert_features,
    )
    source_features = source_features.to(device)
    source_spec_full = source_spec_full.to(device)
    target_spec_full = u.stft_torch(
        target_audio, n_fft, hop_length, win_length, window
    ).to(device)

    freq_bins, frames = source_spec_full.shape
    hop_frames = int(seg_frames * (1 - overlap_ratio))
    hop_frames = max(hop_frames, 1)
    # 予測スペクトルの積算用バッファ
    spec_sum = torch.zeros(2, freq_bins, frames, device=device)
    spec_count = torch.zeros(1, freq_bins, frames, device=device)

    starts = list(range(0, max(1, frames - seg_frames + 1), hop_frames))
    # 末尾の区間が欠けないように、最後の開始位置を追加
    last_start = max(0, frames - seg_frames)
    if starts[-1] != last_start:
        starts.append(last_start)

    # 分割したパッチごとにスペクトルを推定
    for t0 in starts:
        t1 = min(t0 + seg_frames, frames)
        source_patch = torch.zeros(
            source_features.shape[0], freq_bins, seg_frames, device=device
        )
        source_patch[:, :, : (t1 - t0)] = source_features[:, :, t0:t1]

        source_patch = source_patch.unsqueeze(0)
        pred = model(source_patch)[0]
        spec_sum[:, :, t0:t1] += pred[:, :, : (t1 - t0)]
        spec_count[:, :, t0:t1] += 1.0

    spec_avg = spec_sum / torch.clamp(spec_count, min=1.0)
    estimated_spec = u.channels_to_complex(spec_avg)
    # iSTFT で推定波形を再構成
    estimated_wav = (
        u.istft_torch(
            estimated_spec,
            length=len(source_audio),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )
        .cpu()
        .numpy()
    )

    return (
        estimated_wav,
        source_spec_full.cpu(),
        target_spec_full.cpu(),
        estimated_spec.cpu(),
    )
