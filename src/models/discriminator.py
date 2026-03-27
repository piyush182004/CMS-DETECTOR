"""
discriminator.py
────────────────
PatchGAN discriminator for calorimeter jet image super-resolution.

Architecture
────────────
    Input  : (B, 3, 125, 125) — real HR or generated SR jet image
    Output : (B, 1, H', W')   — patch-wise real/fake probability map

Design choices (for GSoC discussion)
──────────────────────────────────────
    1. PatchGAN  — evaluates realism in overlapping local patches rather
                   than a single global score. Forces the generator to
                   produce physically realistic local shower structure,
                   not just globally plausible images.

    2. Spectral Normalisation — stabilises GAN training by constraining
                                the Lipschitz constant of the discriminator.
                                Critical for physics data which has high
                                dynamic range.

    3. LeakyReLU — standard for discriminators; avoids dead neurons on
                   the large zero-energy regions in jet images.

    4. No sigmoid at output — we use LSGAN (MSE) loss which does not
                              require probabilities. More stable than
                              vanilla GAN with BCE.
"""

import torch
import torch.nn as nn


# ── Spectral-Normalised Conv Block ────────────────────────────────────────────

def conv_block(
    in_ch:   int,
    out_ch:  int,
    stride:  int  = 1,
    use_sn:  bool = True,
) -> nn.Sequential:
    """
    Conv2d (optionally spectral-normalised) → InstanceNorm → LeakyReLU.

    InstanceNorm instead of BatchNorm — more stable for small batch sizes
    and avoids leaking batch statistics across physics events.
    """
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=4,
                     stride=stride, padding=1, bias=False)
    if use_sn:
        conv = nn.utils.spectral_norm(conv)

    layers = [conv]
    if out_ch != 1:                          # no norm on first or last layer
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


# ── PatchGAN Discriminator ────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    70×70 PatchGAN discriminator with spectral normalisation.

    Parameters
    ----------
    in_ch    : input channels (3 — ECAL, HCAL, Tracks)
    base_ch  : base feature channels, doubled at each scale (default 64)
    n_layers : number of strided conv layers controlling patch size (default 4)

    Receptive field
    ───────────────
    With n_layers=4, base_ch=64 the discriminator has a ~70×70 pixel
    receptive field on the 125×125 HR image — well matched to the
    typical jet shower width.
    """

    def __init__(
        self,
        in_ch:   int = 3,
        base_ch: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()

        # First block — no InstanceNorm (standard PatchGAN practice)
        layers = [conv_block(in_ch, base_ch, stride=2, use_sn=True)]

        ch = base_ch
        for i in range(1, n_layers):
            in_c  = ch
            out_c = min(ch * 2, 512)         # cap at 512 channels
            stride = 2 if i < n_layers - 1 else 1
            layers.append(conv_block(in_c, out_c, stride=stride, use_sn=True))
            ch = out_c

        # Output layer — single channel patch map, no activation (LSGAN)
        out_conv = nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1, bias=False)
        out_conv = nn.utils.spectral_norm(out_conv)
        layers.append(out_conv)

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 125, 125)  real or generated HR jet image

        Returns
        -------
        (B, 1, H', W')  patch-wise logits (no sigmoid — use LSGAN loss)
        """
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G_out = torch.randn(4, 3, 125, 125).to(device)
    D     = Discriminator().to(device)
    patch = D(G_out)

    print("Discriminator smoke-test")
    print(f"  Input  : {G_out.shape}")
    print(f"  Output : {patch.shape}  ← patch map")

    total_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"  Params : {total_params:,}")
    print("  discriminator.py — OK")