import torch
import torch.nn as nn

# ============================================================
# Model definition
# ============================================================


class BottleneckBiLSTM(nn.Module):
    """
    Bottleneck feature [B,C,F,T] -> BiLSTM over T -> [B,C,F,T]
    """

    def __init__(
        self,
        c: int,
        f: int,
        hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_dim = c * f
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden * 2, in_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, c, f, t = z.shape
        x = z.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x, _ = self.rnn(x)
        x = self.proj(x)
        x = x.view(b, t, c, f).permute(0, 2, 3, 1).contiguous()
        return x


class UNet2D(nn.Module):
    """
    2D U-Net 構成 (複素スペクトル用)
    """

    def __init__(
        self,
        in_ch: int = 2,
        base: int = 48,
        freq_bins: int = 513,
        bottleneck_rnn_enabled: bool = False,
        bottleneck_rnn_channels: int = 16,
        bottleneck_rnn_hidden: int = 256,
        bottleneck_rnn_layers: int = 2,
        bottleneck_rnn_dropout: float = 0.0,
    ):
        super().__init__()
        self.bottleneck_rnn_enabled = bottleneck_rnn_enabled
        self.bottleneck_rnn_freq_bins = freq_bins
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2),
        )
        self.pool1 = nn.MaxPool2d((1, 2))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2),
        )
        self.pool2 = nn.MaxPool2d((1, 2))
        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2),
        )
        self.pool3 = nn.MaxPool2d((1, 2))
        self.bott = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base * 8, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2),
        )
        if bottleneck_rnn_enabled:
            self.bott_rnn_in = nn.Conv2d(base * 8, bottleneck_rnn_channels, 1)
            self.bott_rnn = BottleneckBiLSTM(
                bottleneck_rnn_channels,
                freq_bins,
                hidden=bottleneck_rnn_hidden,
                num_layers=bottleneck_rnn_layers,
                dropout=bottleneck_rnn_dropout,
            )
            self.bott_rnn_out = nn.Conv2d(bottleneck_rnn_channels, base * 8, 1)
        self.up3 = nn.ConvTranspose2d(
            base * 8, base * 4, kernel_size=(1, 2), stride=(1, 2)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2),
        )
        self.up2 = nn.ConvTranspose2d(
            base * 4, base * 2, kernel_size=(1, 2), stride=(1, 2)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2),
        )
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=(1, 2), stride=(1, 2))
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Conv2d(base, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        複素スペクトルの実部・虚部を推定して返す
        """
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bott(p3)
        if self.bottleneck_rnn_enabled:
            if b.shape[2] != self.bottleneck_rnn_freq_bins:
                raise ValueError(
                    "bottleneck freq bins mismatch: "
                    f"{b.shape[2]} vs {self.bottleneck_rnn_freq_bins}"
                )
            b = self.bott_rnn_in(b)
            b = self.bott_rnn(b)
            b = self.bott_rnn_out(b)
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        pred = self.out(d1)
        return pred
