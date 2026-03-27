"""
sr_loss.py
──────────
Physics-aware loss functions for calorimeter jet image super-resolution.

Based on ESRGAN (Wang et al., ECCV Workshop 2018) with physics losses.

Key design decisions
────────────────────
    1. Relativistic average GAN loss (RaGAN-LS) — prevents discriminator
       collapse by making D predict *relative* realism.  Critical fix
       for the D_loss→0 collapse seen in standard LSGAN.
    2. Spectral frequency loss — penalises missing high-frequency
       content via 2D FFT comparison.
    3. Energy conservation + channel fraction — physics constraints.
    4. L1 reconstruction loss — pixel-level fidelity.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SRLoss(nn.Module):
    """
    Combined super-resolution loss for GAN training.

    Total generator loss
    ────────────────────
        L_G = L1
            + λ_adv    * L_adv      (RaGAN-LS, from ESRGAN)
            + λ_energy * L_energy   (total energy conservation)
            + λ_ch_energy * L_ch_energy  (per-channel energy conservation)
            + λ_cf     * L_cf
            + λ_freq   * L_freq
    """

    CH_NAMES = ["ECAL", "HCAL", "Tracks"]

    def __init__(
        self,
        lambda_adv:       float = 0.01,
        lambda_energy:    float = 1.0,
        lambda_ch_energy: float = 0.5,
        lambda_cf:        float = 0.5,
        lambda_freq:      float = 0.1,
    ):
        super().__init__()
        self.lambda_adv       = lambda_adv
        self.lambda_energy    = lambda_energy
        self.lambda_ch_energy = lambda_ch_energy
        self.lambda_cf        = lambda_cf
        self.lambda_freq      = lambda_freq
        self.l1               = nn.L1Loss()

    @staticmethod
    def energy_conservation_loss(sr: Tensor, hr: Tensor) -> Tensor:
        """
        Log-ratio energy conservation loss (total).
        Symmetric — penalises both over and under-production equally.

            L_energy = mean( log(E_sr / E_hr)^2 )
        """
        eps   = 1e-8
        e_sr  = sr.sum(dim=(1, 2, 3))
        e_hr  = hr.sum(dim=(1, 2, 3))
        ratio = e_sr / (e_hr + eps)
        return torch.mean((torch.log(ratio + eps)) ** 2)

    @staticmethod
    def per_channel_energy_loss(sr: Tensor, hr: Tensor) -> tuple[Tensor, dict]:
        """
        Per-channel log-ratio energy conservation loss.

        Computes energy conservation independently for ECAL, HCAL, and
        Tracks so that each detector channel preserves its energy scale.
        Returns total loss and per-channel breakdown for logging.

            L_ch = mean_over_channels( mean_over_batch( log(E_sr_c / E_hr_c)^2 ) )
        """
        eps   = 1e-8
        # (B, C) — energy per channel
        e_sr  = sr.sum(dim=(2, 3))
        e_hr  = hr.sum(dim=(2, 3))
        ratio = e_sr / (e_hr + eps)
        log_sq = (torch.log(ratio + eps)) ** 2   # (B, C)

        ch_losses = log_sq.mean(dim=0)           # (C,)
        total     = ch_losses.mean()

        breakdown = {
            f"energy_err_{SRLoss.CH_NAMES[c]}": ch_losses[c]
            for c in range(min(sr.shape[1], len(SRLoss.CH_NAMES)))
        }
        return total, breakdown

    @staticmethod
    def channel_fraction_loss(sr: Tensor, hr: Tensor) -> Tensor:
        """
        Channel energy fraction loss.
        Penalises wrong relative energy balance across ECAL/HCAL/Tracks.

            frac = channel_energy / total_energy
            L_cf = mean( |frac_sr - frac_hr| )
        """
        eps    = 1e-8
        e_sr   = sr.sum(dim=(2, 3))                        # (B, C)
        e_hr   = hr.sum(dim=(2, 3))                        # (B, C)
        t_sr   = e_sr.sum(dim=1, keepdim=True) + eps       # (B, 1)
        t_hr   = e_hr.sum(dim=1, keepdim=True) + eps       # (B, 1)
        return torch.mean(torch.abs(e_sr / t_sr - e_hr / t_hr))

    @staticmethod
    def discriminator_loss(pred_real: Tensor, pred_fake: Tensor) -> Tensor:
        """
        Relativistic average LSGAN discriminator loss (from ESRGAN).

        Unlike standard LSGAN which only asks "is this real?", RaGAN
        asks "is real MORE realistic than fake?".  This prevents the
        discriminator from collapsing when G produces decent outputs,
        because D always has a meaningful comparison to make.

        L_D = 0.5 * mean((D(real) - mean(D(fake)) - 1)^2)
            + 0.5 * mean((D(fake) - mean(D(real)) + 1)^2)
        """
        mean_real = pred_real.mean().detach()
        mean_fake = pred_fake.mean().detach()
        return 0.5 * (
            torch.mean((pred_real - mean_fake - 1.0) ** 2) +
            torch.mean((pred_fake - mean_real + 1.0) ** 2)
        )

    @staticmethod
    def adversarial_loss(
        pred_fake: Tensor, pred_real: Tensor,
    ) -> Tensor:
        """
        Relativistic average LSGAN generator loss (from ESRGAN).

        The generator tries to make D believe that fake images are
        *relatively* more realistic than real ones.  This provides
        useful gradients to G even when D is confident.

        L_adv = 0.5 * mean((D(real) - mean(D(fake)) + 1)^2)
              + 0.5 * mean((D(fake) - mean(D(real)) - 1)^2)
        """
        mean_real = pred_real.mean().detach()
        mean_fake = pred_fake.mean().detach()
        return 0.5 * (
            torch.mean((pred_real - mean_fake + 1.0) ** 2) +
            torch.mean((pred_fake - mean_real - 1.0) ** 2)
        )

    @staticmethod
    def spectral_frequency_loss(sr: Tensor, hr: Tensor) -> Tensor:
        """
        Spectral (FFT) loss — penalises missing high-frequency content.

        Computes L1 distance between the 2-D Fourier magnitudes of
        sr and hr.  This directly targets the over-smoothing problem:
        the generator can't hide blurry output in pixel space when the
        frequency representation is also supervised.

            L_freq = mean( |FFT(sr)| - |FFT(hr)| )
        """
        # cuFFT requires float32 for non-power-of-2 sizes (125×125)
        fft_sr = torch.fft.rfft2(sr.float(), norm="ortho")
        fft_hr = torch.fft.rfft2(hr.float(), norm="ortho")
        return torch.mean(torch.abs(torch.abs(fft_sr) - torch.abs(fft_hr)))

    def generator_loss(
        self,
        sr:              Tensor,
        hr:              Tensor,
        pred_fake:       Tensor,
        pred_real:       Tensor = None,
        use_adversarial: bool = True,
    ) -> dict[str, Tensor]:

        l1_loss     = self.l1(sr, hr)
        energy_loss = self.energy_conservation_loss(sr, hr)
        ch_energy_loss, ch_breakdown = self.per_channel_energy_loss(sr, hr)
        cf_loss     = self.channel_fraction_loss(sr, hr)
        freq_loss   = self.spectral_frequency_loss(sr, hr)

        adv_loss = (
            self.adversarial_loss(pred_fake, pred_real)
            if use_adversarial and pred_real is not None
            else torch.tensor(0.0, device=sr.device)
        )

        g_loss = (
            1.0 * l1_loss
            + self.lambda_adv       * adv_loss
            + self.lambda_energy    * energy_loss
            + self.lambda_ch_energy * ch_energy_loss
            + self.lambda_cf        * cf_loss
            + self.lambda_freq      * freq_loss
        )

        result = {
            "g_loss":         g_loss,
            "l1_loss":        l1_loss,
            "adv_loss":       adv_loss,
            "energy_loss":    energy_loss,
            "ch_energy_loss": ch_energy_loss,
            "cf_loss":        cf_loss,
            "freq_loss":      freq_loss,
        }
        # Add per-channel breakdown for logging
        result.update(ch_breakdown)
        return result

    def forward(self, sr, hr, pred_fake, pred_real=None, use_adversarial=True):
        return self.generator_loss(sr, hr, pred_fake, pred_real, use_adversarial)


if __name__ == "__main__":
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr        = torch.rand(4, 3, 125, 125).to(device)
    hr        = torch.rand(4, 3, 125, 125).to(device)
    pred_fake = torch.rand(4, 1, 13, 13).to(device)
    pred_real = torch.rand(4, 1, 13, 13).to(device)

    criterion = SRLoss().to(device)
    d_loss    = criterion.discriminator_loss(pred_real, pred_fake)
    g_losses  = criterion.generator_loss(
        sr, hr, pred_fake, pred_real, use_adversarial=True,
    )

    print("SRLoss smoke-test (RaGAN-LS)")
    print(f"  D loss     : {d_loss.item():.6f}")
    for k, v in g_losses.items():
        print(f"  {k:<14}: {v.item():.6f}")
    print("  sr_loss.py — OK")