import os

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import utils as u
from io_utils import info_msg
from model import UNet2D
from notify_utils import send_info_message
from tqdm import tqdm

# ============================================================
# Training loop
# ============================================================


def train_model(
    output_dir: str,
    model: UNet2D,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    snr_weight: float,
    device: torch.device,
) -> tuple[list[float], list[float], UNet2D]:
    """
    U-Net を学習し，損失推移と学習率を返す
    """
    # モデルと最適化設定
    model = model.to(device)
    criterion = u.ComplexSpectralLoss(
        cfg.MAG_WEIGHT,
        snr_weight,
        cfg.LOG_WEIGHT,
        sil_weight=cfg.SIL_WEIGHT,
        sil_alpha=cfg.SIL_ALPHA,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    losses: list[float] = []
    learning_rates: list[float] = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        # 1 epoch 分の学習
        for source_batch, target_batch in progress:
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            pred = model(source_batch)
            estimated_complex = u.channels_to_complex(pred)
            target_complex = u.channels_to_complex(target_batch)

            loss = criterion(estimated_complex, target_complex)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        epoch_msg = (
            f"Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.6f}, "
            f"lr={learning_rates[-1]:.2e}"
        )
        info_msg(epoch_msg)
        if (epoch + 1) % 50 == 0:
            send_info_message(epoch_msg)

        # --- 中間可視化（最初と50エポックごと） ---
        if epoch == 0 or (epoch + 1) % 50 == 0:
            save_intermediate_spectrograms(
                output_dir=output_dir,
                model=model,
                train_loader=train_loader,
                device=device,
                epoch=epoch,
            )

    save_training_curves(
        output_dir=output_dir, losses=losses, learning_rates=learning_rates
    )
    save_training_metrics(
        output_dir=output_dir, losses=losses, learning_rates=learning_rates
    )
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))

    return losses, learning_rates, model


def save_intermediate_spectrograms(
    output_dir: str,
    model: UNet2D,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> None:
    """
    中間結果のスペクトルを保存
    """
    training_dir = os.path.join(output_dir, "training")
    os.makedirs(training_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        source_batch, target_batch = next(iter(train_loader))
        source_batch = source_batch.to(device)
        target_batch = target_batch.to(device)

        pred = model(source_batch)
        estimated_complex = u.channels_to_complex(pred)
        source_complex = u.channels_to_complex(source_batch[:, :2])
        target_complex = u.channels_to_complex(target_batch)

        # 1枚目のスペクトルを保存
        magnitude_source = u.to_db(source_complex[0])
        magnitude_target = u.to_db(target_complex[0])
        magnitude_est = u.to_db(estimated_complex[0])

        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.imshow(magnitude_source, aspect="auto", origin="lower")
        plt.title("Src dB")
        plt.subplot(3, 1, 2)
        plt.imshow(magnitude_target, aspect="auto", origin="lower")
        plt.title("Tgt dB")
        plt.subplot(3, 1, 3)
        plt.imshow(magnitude_est, aspect="auto", origin="lower")
        plt.title(f"Est dB (epoch {epoch + 1})")
        plt.tight_layout()
        plt.savefig(os.path.join(training_dir, f"epoch_{epoch + 1:03d}_spec.png"))
        plt.close()


def save_training_curves(
    output_dir: str, losses: list[float], learning_rates: list[float]
) -> None:
    """
    学習曲線と学習率の推移を画像で保存
    """
    training_dir = os.path.join(output_dir, "training")
    os.makedirs(training_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.yscale("symlog")
    plt.grid(True)
    plt.title("Training Loss")
    plt.savefig(os.path.join(training_dir, "training_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(learning_rates)
    plt.grid(True)
    plt.title("Learning Rate")
    plt.savefig(os.path.join(training_dir, "lr_curve.png"))
    plt.close()


def save_training_metrics(
    output_dir: str, losses: list[float], learning_rates: list[float]
) -> None:
    """
    学習ログを CSV で保存
    """
    np.savetxt(os.path.join(output_dir, "losses.csv"), np.array(losses), delimiter=",")
    np.savetxt(
        os.path.join(output_dir, "lrs.csv"),
        np.array(learning_rates),
        delimiter=",",
    )
