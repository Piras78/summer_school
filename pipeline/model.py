"""
model.py — U-Net per regressione immagine->immagine su Cryo-EM.

Architettura:
  Input (1, 128, 128)
  Encoder: 4 livelli, canali 32->64->128->256
  Bottleneck: 512 canali, 8x8  <-- QUI si estraggono gli embedding
  Decoder: 4 livelli speculari con skip connections
  Output: 1x1 conv -> (1, 128, 128)

Scelte tecniche:
  - GroupNorm(8) invece di BatchNorm: stabile con batch piccoli
  - GELU nel bottleneck, LeakyReLU nel resto
  - Dropout solo nel bottleneck
  - forward() ritorna (output_img, bottleneck_features) per l'estrazione embedding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Blocchi base ───────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Conv -> GroupNorm -> Activation x 2."""

    def __init__(self, in_ch: int, out_ch: int, activation: str = "leaky"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.1, inplace=True)
        groups = min(8, out_ch)   # GroupNorm richiede groups <= channels
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            act,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv + MaxPool 2x2. Ritorna (skip, downsampled)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, activation="leaky")
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    """ConvTranspose2d 2x2 + concat skip + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, activation="leaky")

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad se necessario (immagini non potenza di 2 esatta)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── U-Net ─────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net configurabile via base_channels e depth.

    Parametri:
      base_channels : canali al primo livello encoder (default 32)
      depth         : numero di livelli encoder/decoder (default 4)
      dropout_p     : dropout probability nel bottleneck (default 0.1)

    Con base_channels=32 e depth=4:
      Enc: 32 -> 64 -> 128 -> 256
      Bottleneck: 512
      Dec: 256 -> 128 -> 64 -> 32
      Output: 1
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1,
                 base_channels: int = 32, depth: int = 4,
                 dropout_p: float = 0.1):
        super().__init__()
        self.depth = depth

        # Canali per ogni livello: [32, 64, 128, 256] con base=32, depth=4
        ch = [base_channels * (2 ** i) for i in range(depth)]
        bottleneck_ch = ch[-1] * 2   # 512

        # Encoder
        self.encoders = nn.ModuleList()
        prev = in_ch
        for c in ch:
            self.encoders.append(EncoderBlock(prev, c))
            prev = c

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DoubleConv(ch[-1], bottleneck_ch, activation="gelu"),
            nn.Dropout2d(p=dropout_p),
        )

        # Decoder (reversed)
        self.decoders = nn.ModuleList()
        dec_in = bottleneck_ch
        for c in reversed(ch):
            self.decoders.append(DecoderBlock(dec_in, c, c))
            dec_in = c

        # Head finale
        self.head = nn.Conv2d(ch[0], out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor):
        skips = []

        # Encoder path
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)
        bottleneck_features = x   # shape: (B, 512, H/16, W/16)

        # Decoder path
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        output = self.head(x)
        return output, bottleneck_features


# ── Utility ───────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Verifica ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from config import CFG

    device = CFG["device"]
    model = UNet(
        base_channels=CFG["base_channels"],
        depth=CFG["depth"],
        dropout_p=CFG["dropout_bottleneck"],
    ).to(device)

    dummy = torch.randn(4, 1, 128, 128, device=device)
    with torch.no_grad():
        out, bottleneck = model(dummy)

    print(f"Input           : {dummy.shape}")
    print(f"Output          : {out.shape}")
    print(f"Bottleneck      : {bottleneck.shape}")
    print(f"Parametri       : {count_parameters(model):,}")
    assert out.shape == dummy.shape, "Shape output errata!"
    print("Model OK")
