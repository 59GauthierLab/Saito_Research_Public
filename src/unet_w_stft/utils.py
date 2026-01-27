import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Audio helpers
# ============================================================

plt.rcParams.update(
    {
        "font.size": 18,
    }
)


@torch.no_grad()
def wav_load(path, target_sr=None):
    """
    wav を読み込み，モノラル波形を返す
    """
    import librosa

    y, s = librosa.load(path, sr=target_sr, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32), s


def stft_torch(
    y_np: np.ndarray, n_fft: int, hop_length: int, win_length: int, window: torch.Tensor
):
    """
    numpy 波形を torch STFT に変換
    """
    y = torch.from_numpy(y_np)
    window = window.to(y.device)
    return torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect",
    )


def istft_torch(
    spec: torch.Tensor,
    length: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
):
    """
    torch STFT から波形へ戻す
    """
    window = window.to(spec.device)
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
        center=True,
    )


def complex_to_channels(z: torch.Tensor):
    """
    complex Tensor を実部・虚部チャンネルに分解
    """
    if z.ndim == 2:
        return torch.stack([z.real, z.imag], dim=0)
    elif z.ndim == 3:
        return torch.stack([z.real, z.imag], dim=1)
    else:
        raise ValueError(f"unexpected shape for complex_to_channels: {z.shape}")


def channels_to_complex(x: torch.Tensor):
    """
    実部・虚部チャンネルから complex Tensor を復元
    """
    if x.ndim == 3:
        return torch.complex(x[0], x[1])
    elif x.ndim == 4:
        return torch.complex(x[:, 0], x[:, 1])
    else:
        raise ValueError(f"unexpected shape for channels_to_complex: {x.shape}")


def hilbert_transform(y_np: np.ndarray) -> np.ndarray:
    """
    ヒルベルト変換（虚部）を返す
    """
    from scipy.signal import hilbert

    analytic = hilbert(y_np)
    return np.asarray(analytic.imag, dtype=np.float32)


def build_input_features(
    source_audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    use_hilbert_features: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    入力特徴量（複素2ch + 任意のヒルベルト特徴）を作成する
    """
    src_spec = stft_torch(source_audio, n_fft, hop_length, win_length, window)
    features = [complex_to_channels(src_spec)]

    if use_hilbert_features:
        hilbert_audio = hilbert_transform(source_audio)
        hilbert_spec = stft_torch(hilbert_audio, n_fft, hop_length, win_length, window)
        features.append(complex_to_channels(hilbert_spec))

    return torch.cat(features, dim=0), src_spec


def build_spec_segments(
    src_features: torch.Tensor,
    tgt_spec: torch.Tensor,
    seg_frames: int,
    overlap: float,
):
    """
    STFT を指定フレーム長で分割し，バッチ化
    """
    c, F, T = src_features.shape
    hop = int(seg_frames * (1 - overlap)) or 1
    xs, ys = [], []
    starts = list(range(0, max(1, T - seg_frames + 1), hop))
    # 末尾のフレームを必ずカバーするため、最後の開始位置を追加
    last_start = max(0, T - seg_frames)
    if starts[-1] != last_start:
        starts.append(last_start)
    for t0 in starts:
        t1 = min(t0 + seg_frames, T)
        x_seg_c = torch.zeros(
            c, F, seg_frames, dtype=src_features.dtype, device=src_features.device
        )
        y_seg_c = torch.zeros(
            2, F, seg_frames, dtype=torch.float32, device=tgt_spec.device
        )
        x = src_features[:, :, t0:t1]
        y = complex_to_channels(tgt_spec[:, t0:t1])
        x_seg_c[:, :, : (t1 - t0)] = x
        y_seg_c[:, :, : (t1 - t0)] = y
        xs.append(x_seg_c)
        ys.append(y_seg_c)
    X = torch.stack(xs, dim=0)
    Y = torch.stack(ys, dim=0)
    return X, Y, hop


class ComplexSpectralLoss(nn.Module):
    """
    複素スペクトルの L1 + Log-Mag + SNR + 無音強制（重み付き差分）ロス
    """

    def __init__(
        self,
        mag_weight=0.5,
        snr_weight=0.0,
        log_mag_weight=0.1,
        sil_weight: float = 0.0,
        sil_alpha: float = 0.01,
    ):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mag_weight = mag_weight
        self.log_mag_weight = log_mag_weight
        self.snr_weight = snr_weight
        self.sil_weight = sil_weight
        self.sil_alpha = sil_alpha

    def _snr_loss(self, est_cplx, tgt_cplx):
        est_mag = est_cplx.abs()
        tgt_mag = tgt_cplx.abs()
        num = torch.sum(tgt_mag**2, dim=(-2, -1)) + 1e-12
        den = torch.sum((tgt_mag - est_mag) ** 2, dim=(-2, -1)) + 1e-12
        snr = 10 * torch.log10(num / den)
        return -snr.mean()

    def forward(self, est_cplx, tgt_cplx):
        """
        損失値を計算
        """
        loss_reim = self.l1(est_cplx.real, tgt_cplx.real) + self.l1(
            est_cplx.imag, tgt_cplx.imag
        )
        loss_mag = self.l1(est_cplx.abs(), tgt_cplx.abs())
        loss_log_mag = self.l1(torch.log1p(est_cplx.abs()), torch.log1p(tgt_cplx.abs()))
        loss = (
            loss_reim + self.mag_weight * loss_mag + self.log_mag_weight * loss_log_mag
        )
        if self.snr_weight:
            loss = loss + self.snr_weight * self._snr_loss(est_cplx, tgt_cplx)
        if self.sil_weight:
            if self.sil_alpha <= 0:
                raise ValueError("sil_alpha must be > 0 when sil_weight is enabled")
            w = torch.exp(-tgt_cplx.abs() / self.sil_alpha)
            loss = loss + self.sil_weight * torch.mean(w * (est_cplx - tgt_cplx).abs())
        return loss


def create_spec_dataloader(
    source_audio,
    target_audio,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    seg_frames: int,
    overlap: float,
    batch_size: int = 6,
    use_hilbert_features: bool = False,
):
    """
    STFT セグメントを DataLoader にまとめる
    """
    src_features, src_spec = build_input_features(
        source_audio,
        n_fft,
        hop_length,
        win_length,
        window,
        use_hilbert_features=use_hilbert_features,
    )
    tgt_spec = stft_torch(target_audio, n_fft, hop_length, win_length, window)
    X, Y, hop = build_spec_segments(
        src_features, tgt_spec, seg_frames=seg_frames, overlap=overlap
    )
    dl = DataLoader(
        TensorDataset(X.cpu(), Y.cpu()), batch_size=batch_size, shuffle=True
    )
    return dl, {"F": src_spec.shape[0], "T": src_spec.shape[1], "hop_frames": hop}


def to_db(m):
    """
    振幅を dB スケールに変換
    """
    if torch.is_tensor(m):
        m = m.abs().cpu().numpy()
    else:
        m = np.abs(m)
    return 20 * np.log10(m + 1e-6)


def save_timeplots(src, tgt, est, sr, path_prefix):
    """
    時間波形のプロットを保存
    """
    t = np.arange(len(src)) / sr
    plt.figure(figsize=(15, 9))
    plt.subplot(3, 1, 1)
    plt.plot(t, src)
    plt.title("Input (time)")
    plt.subplot(3, 1, 2)
    plt.plot(t, tgt)
    plt.title("Target (time)")
    plt.subplot(3, 1, 3)
    plt.plot(t[: len(est)], est)
    plt.title("Predicted (time)")
    plt.tight_layout()
    plt.savefig(path_prefix + "_time.png")
    plt.close()

    zN = min(len(src), 2 * sr)
    tz = np.arange(zN) / sr
    plt.figure(figsize=(15, 9))
    plt.subplot(3, 1, 1)
    plt.plot(tz, src[:zN])
    plt.title("Input (0-2s)")
    plt.subplot(3, 1, 2)
    plt.plot(tz, tgt[:zN])
    plt.title("Target (0-2s)")
    plt.subplot(3, 1, 3)
    plt.plot(tz, est[:zN])
    plt.title("Predicted (0-2s)")
    plt.tight_layout()
    plt.savefig(path_prefix + "_time_zoom.png")
    plt.close()


def save_specplots(src_spec, tgt_spec, est_spec, path_prefix):
    """
    スペクトルのプロットを保存
    """
    plt.figure(figsize=(14, 12))
    plt.subplot(3, 1, 1)
    plt.imshow(to_db(src_spec), aspect="auto", origin="lower")
    plt.title("Input dB")
    plt.subplot(3, 1, 2)
    plt.imshow(to_db(tgt_spec), aspect="auto", origin="lower")
    plt.title("Target dB")
    plt.subplot(3, 1, 3)
    plt.imshow(to_db(est_spec), aspect="auto", origin="lower")
    plt.title("Predicted dB")
    plt.tight_layout()
    plt.savefig(path_prefix + "_spec.png")
    plt.close()


def save_hist(src, tgt, est, path):
    """
    振幅分布のヒストグラムを保存
    """
    plt.figure(figsize=(10, 6))
    plt.hist(src, bins=100, alpha=0.5, label="src")
    plt.hist(tgt, bins=100, alpha=0.5, label="tgt")
    plt.hist(est, bins=100, alpha=0.5, label="est")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def snr(ref, est):
    """
    SNR を計算
    """
    num = np.sum(ref**2) + 1e-12
    den = np.sum((ref - est) ** 2) + 1e-12
    return 10 * np.log10(num / den)


def complex_reim_l1(est_cplx: torch.Tensor, tgt_cplx: torch.Tensor) -> float:
    """
    複素スペクトルの実部・虚部に対する L1 損失（mean(|Re|)+mean(|Im|)）を返す
    """
    if not torch.is_tensor(est_cplx) or not torch.is_tensor(tgt_cplx):
        raise TypeError("est_cplx and tgt_cplx must be torch.Tensor")
    if not torch.is_complex(est_cplx) or not torch.is_complex(tgt_cplx):
        raise TypeError("est_cplx and tgt_cplx must be complex tensors")
    if est_cplx.shape != tgt_cplx.shape:
        raise ValueError(f"shape mismatch: {est_cplx.shape} vs {tgt_cplx.shape}")

    loss_re = torch.mean(torch.abs(est_cplx.real - tgt_cplx.real))
    loss_im = torch.mean(torch.abs(est_cplx.imag - tgt_cplx.imag))
    return float((loss_re + loss_im).item())


@torch.no_grad()
def silence_residual_metrics(
    est_cplx: torch.Tensor,
    tgt_cplx: torch.Tensor,
    *,
    tgt_mag_threshold: float | None = None,
    tgt_mag_quantile: float = 0.1,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    ほぼ無音（教師スペクトル振幅が小さい）領域における残留成分を評価する指標を返す．

    マスクは ``|S(y)| <= tgt_mag_threshold`` で作る．
    ``tgt_mag_threshold`` が None の場合は，
    ``|S(y)|`` の分位点 ``tgt_mag_quantile`` を用いる．
    """
    if not torch.is_tensor(est_cplx) or not torch.is_tensor(tgt_cplx):
        raise TypeError("est_cplx and tgt_cplx must be torch.Tensor")
    if not torch.is_complex(est_cplx) or not torch.is_complex(tgt_cplx):
        raise TypeError("est_cplx and tgt_cplx must be complex tensors")
    if est_cplx.shape != tgt_cplx.shape:
        raise ValueError(f"shape mismatch: {est_cplx.shape} vs {tgt_cplx.shape}")

    tgt_mag = tgt_cplx.abs()
    if tgt_mag_threshold is None:
        q = float(tgt_mag_quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("tgt_mag_quantile must be in (0, 1)")
        tgt_mag_threshold = float(torch.quantile(tgt_mag.reshape(-1), q).item())
    else:
        tgt_mag_threshold = float(tgt_mag_threshold)

    mask = tgt_mag <= tgt_mag_threshold
    mask_ratio = float(mask.float().mean().item())

    est_mag = est_cplx.abs()
    diff_mag = (est_cplx - tgt_cplx).abs()

    est_mag_sil = est_mag[mask]
    diff_mag_sil = diff_mag[mask]

    est_mean_abs = float(est_mag_sil.mean().item()) if est_mag_sil.numel() else 0.0
    diff_mean_abs = float(diff_mag_sil.mean().item()) if diff_mag_sil.numel() else 0.0
    est_mean_pow = float((est_mag_sil**2).mean().item()) if est_mag_sil.numel() else 0.0
    diff_mean_pow = (
        float((diff_mag_sil**2).mean().item()) if diff_mag_sil.numel() else 0.0
    )

    est_mean_abs_db = float(20.0 * np.log10(est_mean_abs + eps))
    diff_mean_abs_db = float(20.0 * np.log10(diff_mean_abs + eps))
    est_mean_pow_db = float(10.0 * np.log10(est_mean_pow + eps))
    diff_mean_pow_db = float(10.0 * np.log10(diff_mean_pow + eps))

    return {
        "tgt_mag_threshold": float(tgt_mag_threshold),
        "silence_mask_ratio": mask_ratio,
        "silence_est_mean_abs": est_mean_abs,
        "silence_est_mean_abs_db": est_mean_abs_db,
        "silence_diff_mean_abs": diff_mean_abs,
        "silence_diff_mean_abs_db": diff_mean_abs_db,
        "silence_est_mean_pow": est_mean_pow,
        "silence_est_mean_pow_db": est_mean_pow_db,
        "silence_diff_mean_pow": diff_mean_pow,
        "silence_diff_mean_pow_db": diff_mean_pow_db,
    }
