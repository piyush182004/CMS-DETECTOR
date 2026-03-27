"""
gan.py
──────
SRGAN training module — connects Generator and Discriminator with
adversarial + physics-aware losses into a single trainable unit.

Design choices (for GSoC discussion)
──────────────────────────────────────
    1. RaGAN-LS objective — relativistic average least-squares adversarial
                          loss from ESRGAN.  Prevents discriminator collapse
                          by making D predict *relative* realism rather than
                          absolute real/fake.

    2. Two-timescale    — discriminator updated every step, generator
                          updated every step but with slower effective
                          learning (standard SRGAN practice).

    3. Warm-up phase    — generator pre-trained with L1 only before
                          adversarial loss is switched on. Prevents
                          the discriminator from dominating too early.

    4. EMA weights      — exponential moving average of generator weights
                          for stable inference (no oscillation at eval).

    5. Warmup + Hold + Cosine — LR ramps linearly for warmup_epochs,
                          holds at peak for hold_epochs, then decays
                          via cosine to lr_min. Prevents premature LR
                          collapse on short (15-epoch) runs.
"""

import copy
import math
import torch
import torch.nn as nn
from torch import Tensor

from src.models.generator     import Generator
from src.models.discriminator import Discriminator
from src.losses.sr_loss       import SRLoss


class SRGAN(nn.Module):
    """
    Super-Resolution GAN training wrapper.

    Parameters
    ----------
    cfg : dict  — loaded from configs/srgan.yaml
        Expected keys:
            model.n_feat, model.n_groups, model.n_blocks, model.reduction
            training.lr_g, training.lr_d
            training.warmup_epochs, training.hold_epochs, training.epochs
            loss.lambda_adv, loss.lambda_energy, loss.lambda_cf
    """

    def __init__(self, cfg: dict):
        super().__init__()

        mcfg = cfg["model"]
        tcfg = cfg["training"]
        lcfg = cfg["loss"]

        # ── networks ──────────────────────────────────────────────────────────
        self.generator = Generator(
            n_feat    = mcfg["n_feat"],
            n_groups  = mcfg["n_groups"],
            n_blocks  = mcfg["n_blocks"],
            growth    = mcfg.get("growth", 32),
            reduction = mcfg["reduction"],
        )
        self.discriminator = Discriminator()

        # EMA copy of generator for stable inference
        self.ema_generator = copy.deepcopy(self.generator)
        self.ema_generator.requires_grad_(False)
        self.ema_decay = mcfg.get("ema_decay", 0.999)

        # ── loss ──────────────────────────────────────────────────────────────
        self.criterion = SRLoss(
            lambda_adv       = lcfg["lambda_adv"],
            lambda_energy    = lcfg["lambda_energy"],
            lambda_ch_energy = lcfg.get("lambda_ch_energy", 0.5),
            lambda_cf        = lcfg["lambda_cf"],
            lambda_freq      = lcfg.get("lambda_freq", 0.1),
        )

        # ── discriminator update ratio (n_critic) ─────────────────────────
        # RaGAN is self-stabilising — D always has a meaningful signal
        # because it compares real vs fake relatively.  n_critic=1 is
        # typically sufficient (unlike standard LSGAN).
        self.n_critic = tcfg.get("n_critic", 1)

        # ── optimisers ────────────────────────────────────────────────────────
        # Adam betas: (0.9, 0.999) is the proven stable choice for SR-GANs.
        # Override via config betas key if present.
        betas = tuple(tcfg.get("betas", [0.9, 0.999]))
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=tcfg["lr_g"], betas=betas,
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=tcfg["lr_d"], betas=betas,
        )

        # ── schedulers: warmup → hold → cosine decay ─────────────────────────
        warmup = tcfg.get("warmup_epochs", 0)
        hold   = tcfg.get("hold_epochs", 0)
        epochs = tcfg["epochs"]
        lr_g   = tcfg["lr_g"]
        lr_d   = tcfg["lr_d"]
        lr_min = tcfg.get("lr_min", 1e-6)
        start_factor = tcfg.get("warmup_start", 0.1)

        def _make_schedule(peak_lr):
            def schedule(epoch):
                if epoch < warmup:
                    t = epoch / max(warmup - 1, 1)
                    return start_factor + (1.0 - start_factor) * t
                if epoch < warmup + hold:
                    return 1.0
                # cosine decay from peak → lr_min
                decay_len = max(epochs - warmup - hold, 1)
                progress  = (epoch - warmup - hold) / decay_len
                cosine    = 0.5 * (1.0 + math.cos(math.pi * progress))
                return max(lr_min / peak_lr, cosine)
            return schedule

        self.sched_g = torch.optim.lr_scheduler.LambdaLR(
            self.opt_g, lr_lambda=_make_schedule(lr_g)
        )
        self.sched_d = torch.optim.lr_scheduler.LambdaLR(
            self.opt_d, lr_lambda=_make_schedule(lr_d)
        )

        self.warmup_epochs = warmup


    # ── EMA update ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _update_ema(self) -> None:
        for ema_p, gen_p in zip(
            self.ema_generator.parameters(),
            self.generator.parameters(),
        ):
            ema_p.data.mul_(self.ema_decay).add_(
                gen_p.data, alpha=1.0 - self.ema_decay
            )

    # ── discriminator step ────────────────────────────────────────────────────

    def discriminator_step(
        self,
        lr:      Tensor,
        hr_real: Tensor,
        use_amp: bool  = False,
        scaler          = None,
    ) -> dict:
        self.opt_d.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            with torch.no_grad():
                hr_fake = self.generator(lr)
            pred_real = self.discriminator(hr_real)
            pred_fake = self.discriminator(hr_fake.detach())
            loss_d    = self.criterion.discriminator_loss(pred_real, pred_fake)

        if scaler:
            scaler.scale(loss_d).backward()
            scaler.step(self.opt_d)
            scaler.update()
        else:
            loss_d.backward()
            self.opt_d.step()

        return {
            "d_loss": loss_d.item(),
            "d_real": pred_real.mean().item(),
            "d_fake": pred_fake.mean().item(),
        }

    # ── generator step ────────────────────────────────────────────────────────

    def generator_step(
        self,
        lr:      Tensor,
        hr_real: Tensor,
        epoch:   int,
        use_amp: bool  = False,
        scaler          = None,
    ) -> dict:
        self.opt_g.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            hr_fake   = self.generator(lr)
            pred_fake = self.discriminator(hr_fake)
            # RaGAN: G also needs D(real) to compute relative realism
            with torch.no_grad():
                pred_real = self.discriminator(hr_real)
            adversarial = epoch >= self.warmup_epochs
            losses    = self.criterion.generator_loss(
                hr_fake, hr_real, pred_fake, pred_real,
                use_adversarial=adversarial,
            )

        if scaler:
            scaler.scale(losses["g_loss"]).backward()
            scaler.unscale_(self.opt_g)
            nn.utils.clip_grad_norm_(
                self.generator.parameters(), max_norm=1.0
            )
            scaler.step(self.opt_g)
            scaler.update()
        else:
            losses["g_loss"].backward()
            nn.utils.clip_grad_norm_(
                self.generator.parameters(), max_norm=1.0
            )
            self.opt_g.step()

        self._update_ema()
        return {k: v.item() for k, v in losses.items()}

    # ── full training step ────────────────────────────────────────────────────

    def training_step(
        self,
        lr:      Tensor,
        hr_real: Tensor,
        epoch:   int,
        use_amp: bool = False,
        scaler         = None,
    ) -> dict:
        # Update discriminator n_critic times per generator step
        # so it stays competitive with the larger generator.
        d_metrics = {}
        for _ in range(self.n_critic):
            d_metrics = self.discriminator_step(lr, hr_real, use_amp, scaler)
        g_metrics = self.generator_step(lr, hr_real, epoch, use_amp, scaler)
        return {**d_metrics, **g_metrics}

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def infer(self, lr: Tensor, use_ema: bool = True) -> Tensor:
        """Run inference with clamping to [0, 1]."""
        net = self.ema_generator if use_ema else self.generator
        net.eval()
        return net(lr).clamp(0.0, 1.0)

    # ── scheduler step — call once per epoch ──────────────────────────────────

    def scheduler_step(self) -> None:
        self.sched_g.step()
        self.sched_d.step()

    def get_lr(self) -> dict:
        """Return current learning rates for logging."""
        return {
            "lr_g": self.opt_g.param_groups[0]["lr"],
            "lr_d": self.opt_d.param_groups[0]["lr"],
        }


if __name__ == "__main__":
    import yaml

    with open("configs/srgan.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SRGAN(cfg).to(device)

    warmup = cfg["training"].get("warmup_epochs", 0)
    hold   = cfg["training"].get("hold_epochs", 0)
    epochs = cfg["training"]["epochs"]

    print(f"LR schedule preview (warmup={warmup}, hold={hold}, "
          f"cosine={epochs - warmup - hold}):")
    for ep in range(epochs):
        model.sched_g.last_epoch = ep - 1
        model.sched_g.step()
        lr = model.opt_g.param_groups[0]["lr"]
        if ep < warmup:
            phase = "warmup"
        elif ep < warmup + hold:
            phase = "hold  "
        else:
            phase = "cosine"
        print(f"  epoch {ep:02d} [{phase}]  lr_g={lr:.2e}")

    print("\ngan.py — OK")
