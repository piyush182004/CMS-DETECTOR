"""
test_output.py
──────────────
Load best checkpoint and visualise LR input vs SR output vs HR ground truth.
Uses the TEST parquet (dataset 3) for proper generalization evaluation.

Usage
─────
    python test_output.py
    python test_output.py --config configs/srgan.yaml
    python test_output.py --ckpt path/to/checkpoint.pt

NOTE: Uses paired normalization (both LR & HR divided by max(HR))
      to match the training pipeline.
"""

import os
import glob
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import yaml

from src.models.generator import Generator


# ── find checkpoint ───────────────────────────────────────────────────────────

def find_checkpoint(ckpt_base: str) -> str:
    """Find best or last checkpoint under the checkpoints directory."""
    for tag in ["srgan_best.pt", "srgan_last.pt"]:
        candidates = glob.glob(
            os.path.join(ckpt_base, "**", tag), recursive=True
        )
        if candidates:
            latest = sorted(candidates)[-1]
            print(f"Checkpoint : {latest}  ({tag})")
            return latest
    raise FileNotFoundError(f"No checkpoint found in {ckpt_base}")


# ── load generator ────────────────────────────────────────────────────────────

def load_generator(ckpt_path: str, device: torch.device) -> tuple:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ckpt["epoch"]
    cfg   = ckpt["config"]
    mcfg  = cfg["model"]

    gen = Generator(
        n_feat    = mcfg["n_feat"],
        n_groups  = mcfg["n_groups"],
        n_blocks  = mcfg["n_blocks"],
        reduction = mcfg["reduction"],
    ).to(device)

    gen.load_state_dict(ckpt["ema_generator"])
    gen.eval()
    print(f"Loaded epoch {epoch + 1} checkpoint")
    return gen, epoch, cfg


# ── normalise (must match parquet_loader.py — PAIRED normalization) ───────────

def normalize_pair(lr: np.ndarray, hr: np.ndarray, eps: float = 1e-8):
    """Paired max normalization — both divided by max(HR).

    MUST match training normalization in parquet_loader.py.
    Using independent normalization would break energy scale.
    """
    scale = hr.max()
    if scale > eps:
        return lr / scale, hr / scale
    return lr, hr


# ── load sample jets ──────────────────────────────────────────────────────────

def load_sample_jets(parquet_path: str, n: int = 4) -> tuple:
    pf    = pq.ParquetFile(parquet_path)
    batch = next(pf.iter_batches(batch_size=n))
    df    = batch.to_pandas()

    def extract(val):
        return np.stack([
            np.array(
                [np.array(row, dtype=np.float32) for row in ch],
                dtype=np.float32
            )
            for ch in val
        ])

    lr_list, hr_list = [], []
    for i in range(n):
        lr = extract(df["X_jets_LR"].iloc[i])
        hr = extract(df["X_jets"].iloc[i])
        # PAIRED normalization — both / max(HR) to preserve energy scale
        lr, hr = normalize_pair(lr, hr)
        lr_list.append(lr)
        hr_list.append(hr)

    return np.stack(lr_list), np.stack(hr_list)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test SRGAN output")
    parser.add_argument("--config", default="configs/srgan.yaml")
    parser.add_argument("--ckpt", default=None, help="Override checkpoint path")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = find_checkpoint(yaml_cfg["results"]["checkpoints"])

    gen, epoch, cfg = load_generator(ckpt_path, device)

    # Use TEST data (dataset 3) for generalization — not train data
    test_path = yaml_cfg["data"]["test"]
    print(f"Loading sample jets from {os.path.basename(test_path)}...")
    lr_np, hr_np = load_sample_jets(test_path, n=4)

    # generate SR
    with torch.no_grad():
        lr_t  = torch.from_numpy(lr_np).to(device)
        sr_t  = gen(lr_t).clamp(0.0, 1.0)
        sr_np = sr_t.cpu().numpy()

    # metrics
    e_sr = sr_np.sum(axis=(1, 2, 3))
    e_hr = hr_np.sum(axis=(1, 2, 3))
    energy_err = np.abs(e_sr - e_hr) / (e_hr + 1e-8)

    mse  = np.mean((sr_np - hr_np)**2, axis=(1, 2, 3))
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))

    # channel fraction error
    sr_cf = sr_np.sum(axis=(2, 3)) / (e_sr[:, None] + 1e-8)   # (B, C)
    hr_cf = hr_np.sum(axis=(2, 3)) / (e_hr[:, None] + 1e-8)   # (B, C)
    cf_err = np.mean(np.abs(sr_cf - hr_cf), axis=1)            # (B,)

    # per-channel energy error
    CH_NAMES = ["ECAL", "HCAL", "Tracks"]
    ch_energy_errs = {}
    for c, ch in enumerate(CH_NAMES):
        ch_e_sr = sr_np[:, c].sum(axis=(1, 2))
        ch_e_hr = hr_np[:, c].sum(axis=(1, 2))
        ch_energy_errs[ch] = np.abs(ch_e_sr - ch_e_hr) / (np.abs(ch_e_hr) + 1e-8)

    print("\n── Per-jet metrics ──────────────────────────")
    for i in range(min(4, len(lr_np))):
        print(
            f"  Jet {i} |"
            f" PSNR={psnr[i]:.2f}dB |"
            f" EnergyErr={energy_err[i]*100:.2f}% |"
            f" ChanFrac={cf_err[i]*100:.2f}%"
        )
    print(f"\n  Mean PSNR        : {psnr.mean():.2f} dB")
    print(f"  Mean EnergyErr   : {energy_err.mean()*100:.2f}%")
    print(f"  Mean ChanFrac    : {cf_err.mean()*100:.2f}%")
    print(f"\n── Per-channel energy error ─────────────────")
    for ch in CH_NAMES:
        print(f"  {ch:<8}: {ch_energy_errs[ch].mean()*100:.2f}%")

    # ── plot ──────────────────────────────────────────────────────────────────
    n_jets   = 2
    fig, axes = plt.subplots(n_jets * 3, 3, figsize=(14, n_jets * 9))
    fig.suptitle(
        f"SR Output — epoch {epoch + 1}\n"
        f"PSNR: {psnr.mean():.2f}dB  "
        f"EnergyErr: {energy_err.mean()*100:.2f}%  "
        f"ChanFrac: {cf_err.mean()*100:.2f}%\n"
        f"ECAL: {ch_energy_errs['ECAL'].mean()*100:.2f}%  "
        f"HCAL: {ch_energy_errs['HCAL'].mean()*100:.2f}%  "
        f"Tracks: {ch_energy_errs['Tracks'].mean()*100:.2f}%",
        fontsize=12, fontweight="bold"
    )

    for jet in range(n_jets):
        row_base = jet * 3
        for c, ch in enumerate(CH_NAMES):
            vmax = max(hr_np[jet, c].max(), sr_np[jet, c].max(), 1e-6)

            lr_display = torch.nn.functional.interpolate(
                torch.from_numpy(lr_np[jet, c][None, None]),
                size=(125, 125), mode="nearest"
            ).squeeze().numpy()

            axes[row_base, c].imshow(
                lr_display, cmap="inferno", vmin=0, vmax=vmax
            )
            # Per-channel energy for this jet
            ch_e_sr_j = sr_np[jet, c].sum()
            ch_e_hr_j = hr_np[jet, c].sum()
            ch_err_j  = abs(ch_e_sr_j - ch_e_hr_j) / (abs(ch_e_hr_j) + 1e-8) * 100

            axes[row_base, c].set_title(f"Jet {jet} | {ch} | LR (64×64)", fontsize=9)
            axes[row_base, c].axis("off")

            axes[row_base+1, c].imshow(
                sr_np[jet, c], cmap="inferno", vmin=0, vmax=vmax
            )
            axes[row_base+1, c].set_title(
                f"Jet {jet} | {ch} | SR (125×125)  E_err={ch_err_j:.1f}%",
                fontsize=9, fontweight="bold", color="green"
            )
            axes[row_base+1, c].axis("off")

            axes[row_base+2, c].imshow(
                hr_np[jet, c], cmap="inferno", vmin=0, vmax=vmax
            )
            axes[row_base+2, c].set_title(f"Jet {jet} | {ch} | HR truth (125×125)", fontsize=9)
            axes[row_base+2, c].axis("off")

    plt.tight_layout()
    out_dir = yaml_cfg["results"]["plots"]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"test_output-fold120333_epoch_{epoch+1:03d}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved : {out}")


if __name__ == "__main__":
    main()