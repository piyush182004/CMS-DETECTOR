"""
generator.py
────────────
RRDB-CA Generator for calorimeter jet image super-resolution.

Based on ESRGAN (Wang et al., ECCV Workshop 2018) with physics-specific
channel attention for ECAL/HCAL/Tracks awareness.

Architecture
────────────
    Input  : (B, 3, 64, 64)   — normalised LR jet image
    Output : (B, 3, 125, 125) — super-resolved HR jet image

Design choices (paper references)
─────────────────────────────────
    1. RRDB (Residual-in-Residual Dense Blocks) — from ESRGAN.
       Dense connections enable maximum feature reuse and gradient
       flow.  Proven superior to plain residual blocks for SR.
    2. Channel attention (SE) — lets the network focus on physically
       active channels (ECAL/HCAL/Tracks) independently.
    3. Sub-pixel conv — learnable upsampling via PixelShuffle.
    4. No BatchNorm — removes range flexibility; standard finding
       from EDSR (2017) / ESRGAN (2018).
    5. LeakyReLU — avoids dead neurons in sparse calorimeter data.
"""

import torch
import torch.nn as nn


# ── Channel Attention (Squeeze-and-Excitation) ────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation block.
    Recalibrates channel-wise feature responses adaptively.

    Parameters
    ----------
    n_feat   : number of feature channels
    reduction: squeeze ratio (default 16)
    """

    def __init__(self, n_feat: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_feat, n_feat // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_feat // reduction, n_feat, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)       # squeeze
        y = self.fc(y).view(b, c, 1, 1)       # excitation
        return x * y                           # recalibrate


# ── Residual Dense Block (ESRGAN) ─────────────────────────────────────────────

class ResidualDenseBlock(nn.Module):
    """
    Dense block with 5 convolution layers and dense (concatenation)
    connections.  From ESRGAN (Wang et al., ECCV Workshop 2018).

    Each layer receives ALL previous feature maps as input, enabling
    maximum feature reuse and strong gradient flow.

    Parameters
    ----------
    n_feat : feature channels
    growth : growth rate — channels added per dense layer (default 32)
    """

    def __init__(self, n_feat: int = 64, growth: int = 32):
        super().__init__()
        G = growth
        self.conv1 = nn.Conv2d(n_feat,         G, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(n_feat + G,     G, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(n_feat + 2 * G, G, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(n_feat + 3 * G, G, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(n_feat + 4 * G, n_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.beta  = 0.2   # residual scaling (ESRGAN default)

        # ESRGAN: initialise last conv near zero for stable residual start
        nn.init.kaiming_normal_(self.conv1.weight, a=0.2, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, a=0.2, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3.weight, a=0.2, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv4.weight, a=0.2, nonlinearity='leaky_relu')
        nn.init.zeros_(self.conv5.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * self.beta + x


# ── RRDB: Residual-in-Residual Dense Block ────────────────────────────────────

class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block from ESRGAN.

    Stacks multiple ResidualDenseBlocks with a macro residual connection
    and channel attention.  The nested residual structure enables very
    deep networks without degradation.

    Parameters
    ----------
    n_feat    : feature channels
    growth    : dense connection growth rate
    n_dense   : number of dense sub-blocks (default 3)
    reduction : channel attention squeeze ratio
    """

    def __init__(self, n_feat: int = 64, growth: int = 32,
                 n_dense: int = 3, reduction: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualDenseBlock(n_feat, growth) for _ in range(n_dense)
        ])
        self.ca   = ChannelAttention(n_feat, reduction)
        self.beta = 0.2   # macro residual scaling (ESRGAN default)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for blk in self.blocks:
            out = blk(out)
        out = self.ca(out)
        return out * self.beta + x


# ── Sub-pixel Upsampler ───────────────────────────────────────────────────────

class SubPixelUpsample(nn.Module):
    """
    Learnable upsampling via PixelShuffle (sub-pixel convolution).
    Preferred over bilinear interpolation for SR tasks.

    For non-integer scale (64 → 125) we upsample ×2 then centre-crop
    to exactly 125×125.

    Parameters
    ----------
    n_feat    : input feature channels
    scale     : integer upscale factor applied via PixelShuffle
    out_size  : final spatial size after crop (125)
    """

    def __init__(self, n_feat: int, scale: int = 2, out_size: int = 125):
        super().__init__()
        self.out_size = out_size
        self.up = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * scale * scale, 3, padding=1, bias=True),
            nn.PixelShuffle(scale),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)                        # (B, C, 128, 128)
        # Centre-crop to target size (128 → 125)
        h, w   = x.shape[2], x.shape[3]
        top    = (h - self.out_size) // 2
        left   = (w - self.out_size) // 2
        return x[:, :, top:top + self.out_size, left:left + self.out_size]


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    ESRGAN-style super-resolution generator with channel attention.

    Architecture:  Head → RRDB body → PixelShuffle upsampler → Tail

    Based on ESRGAN (Wang et al., ECCV Workshop 2018) with physics-
    specific additions:
        - Channel Attention (SE blocks) for ECAL/HCAL/Tracks awareness
        - ReLU output for non-negative energy constraint
        - Sub-pixel convolution for learnable upsampling

    Parameters
    ----------
    in_ch     : input channels  (3 — ECAL, HCAL, Tracks)
    out_ch    : output channels (3)
    n_feat    : internal feature dimension  (default 64)
    n_groups  : number of RRDB blocks       (default 6)
    n_blocks  : dense sub-blocks per RRDB   (default 3)
    growth    : dense connection growth rate (default 32)
    reduction : channel attention ratio     (default 16)
    scale     : pixel-shuffle upscale       (default 2)
    out_size  : final spatial size          (default 125)
    """

    def __init__(
        self,
        in_ch:     int = 3,
        out_ch:    int = 3,
        n_feat:    int = 64,
        n_groups:  int = 6,
        n_blocks:  int = 3,
        growth:    int = 32,
        reduction: int = 16,
        scale:     int = 2,
        out_size:  int = 125,
    ):
        super().__init__()

        # Head — shallow feature extraction
        self.head = nn.Conv2d(in_ch, n_feat, 3, padding=1, bias=True)

        # Body — deep RRDB blocks with dense connections
        self.body = nn.Sequential(
            *[RRDB(n_feat, growth, n_blocks, reduction)
              for _ in range(n_groups)],
            nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=True),
        )

        # Upsampler
        self.upsample = SubPixelUpsample(n_feat, scale, out_size)

        # Tail — map features to output channels + ReLU.
        # Calorimeter energy deposits are physically >= 0.
        self.tail = nn.Sequential(
            nn.Conv2d(n_feat, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for all convs except the last conv in each RDB
        (those are zero-initialised in ResidualDenseBlock.__init__)."""
        for m in self.modules():
            if isinstance(m, ResidualDenseBlock):
                continue   # already initialised properly
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_out",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 64, 64)  normalised LR jet image

        Returns
        -------
        (B, 3, 125, 125)  super-resolved HR jet image
        """
        feat = self.head(x)               # (B, n_feat, 64, 64)
        feat = feat + self.body(feat)     # long skip over all RRDB blocks
        feat = self.upsample(feat)        # (B, n_feat, 125, 125)
        return self.tail(feat)            # (B, 3, 125, 125)


# ── quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Generator().to(device)

    dummy  = torch.randn(4, 3, 64, 64).to(device)
    out    = model(dummy)

    print("Generator smoke-test")
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {out.shape}")
    assert out.shape == (4, 3, 125, 125), "Shape mismatch!"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params : {total_params:,}")
    print("  generator.py — OK")