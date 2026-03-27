"""
train.py
────────
Full GAN training pipeline for calorimeter jet image super-resolution.

Usage
─────
    python -m src.train --config configs/srgan.yaml
    python -m src.train --config configs/srgan.yaml --resume
"""

import os
import sys
import signal
import argparse
import logging
import yaml
import json
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.data.parquet_loader     import build_dataloaders, build_kfold_dataloaders
from src.models.gan              import SRGAN

RUN_ID_FILE = "C:/GSOC-CMS-2K26/results/current_run_id.txt"

# ── emergency state ───────────────────────────────────────────────────────────

_EMERGENCY_STATE = {
    "model": None, "epoch": 0, "batch_idx": 0,
    "cfg": None, "ckpt_dir": None, "run_id": None, "logger": None,
}

def _emergency_save(signum, frame):
    state = _EMERGENCY_STATE
    if state["model"] is not None and state["ckpt_dir"] is not None:
        try:
            path = os.path.join(state["ckpt_dir"], "srgan_interrupt.pt")
            torch.save({
                "epoch":         state["epoch"],
                "batch_idx":     state["batch_idx"],
                "config":        state["cfg"],
                "generator":     state["model"].generator.state_dict(),
                "ema_generator": state["model"].ema_generator.state_dict(),
                "discriminator": state["model"].discriminator.state_dict(),
                "opt_g":         state["model"].opt_g.state_dict(),
                "opt_d":         state["model"].opt_d.state_dict(),
                "sched_g":       state["model"].sched_g.state_dict(),
                "sched_d":       state["model"].sched_d.state_dict(),
                "metrics":       {},
            }, path)
            save_run_id(state["run_id"], state["epoch"])
            print(
                f"\n  INTERRUPT saved —"
                f" epoch {state['epoch']}"
                f" batch {state['batch_idx']}"
            )
        except Exception as e:
            print(f"\n  Emergency save failed: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, _emergency_save)


# ── logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("srgan")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(
        os.path.join(log_dir, "train.log"),
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── run ID persistence ────────────────────────────────────────────────────────

def save_run_id(run_id: str, epoch: int = 0) -> None:
    os.makedirs(os.path.dirname(RUN_ID_FILE), exist_ok=True)
    with open(RUN_ID_FILE, "w") as f:
        f.write(f"{run_id}\n{epoch}")

def load_run_id() -> tuple:
    if os.path.exists(RUN_ID_FILE):
        with open(RUN_ID_FILE) as f:
            lines = f.read().strip().split("\n")
            run_id = lines[0]
            epoch  = int(lines[1]) if len(lines) > 1 else 0
            return run_id, epoch
    return None, 0


# ── checkpoint utilities ──────────────────────────────────────────────────────

def save_checkpoint(
    model:     SRGAN,
    epoch:     int,
    metrics:   dict,
    cfg:       dict,
    ckpt_dir:  str,
    tag:       str = "last",
    batch_idx: int = 0,
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"srgan_{tag}.pt")
    torch.save({
        "epoch":         epoch,
        "batch_idx":     batch_idx,
        "config":        cfg,
        "generator":     model.generator.state_dict(),
        "ema_generator": model.ema_generator.state_dict(),
        "discriminator": model.discriminator.state_dict(),
        "opt_g":         model.opt_g.state_dict(),
        "opt_d":         model.opt_d.state_dict(),
        "sched_g":       model.sched_g.state_dict(),
        "sched_d":       model.sched_d.state_dict(),
        "metrics":       metrics,
    }, path)

def load_checkpoint(
    model:  SRGAN,
    path:   str,
    device: torch.device,
) -> tuple:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.generator.load_state_dict(ckpt["generator"])
    model.ema_generator.load_state_dict(ckpt["ema_generator"])
    model.discriminator.load_state_dict(ckpt["discriminator"])
    model.opt_g.load_state_dict(ckpt["opt_g"])
    model.opt_d.load_state_dict(ckpt["opt_d"])
    model.sched_g.load_state_dict(ckpt["sched_g"])
    model.sched_d.load_state_dict(ckpt["sched_d"])
    return ckpt["epoch"], ckpt.get("batch_idx", 0), ckpt.get("metrics", {})

def find_best_checkpoint(ckpt_dir: str) -> tuple:
    best_path  = None
    best_tag   = None
    best_epoch = -1
    best_batch = -1
    for tag in ["last", "interrupt", "mid_epoch"]:
        path = os.path.join(ckpt_dir, f"srgan_{tag}.pt")
        if os.path.exists(path):
            try:
                ckpt  = torch.load(
                    path, map_location="cpu", weights_only=False
                )
                epoch = ckpt.get("epoch", 0)
                batch = ckpt.get("batch_idx", 0)
                if epoch > best_epoch or (
                    epoch == best_epoch and batch > best_batch
                ):
                    best_epoch = epoch
                    best_batch = batch
                    best_path  = path
                    best_tag   = tag
            except Exception:
                continue
    return best_path, best_tag


# ── training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(
    model:        SRGAN,
    loader,
    device:       torch.device,
    epoch:        int,
    logger:       logging.Logger,
    writer:       SummaryWriter,
    cfg:          dict,
    ckpt_dir:     str,
    skip_batches: int = 0,
    use_amp:      bool = False,
    scaler:       torch.amp.GradScaler | None = None,
) -> dict:
    model.generator.train()
    model.discriminator.train()

    accum = {}
    n_batches = 0

    for batch_idx, (lr, hr) in enumerate(loader):
        if batch_idx < skip_batches:
            continue

        _EMERGENCY_STATE["batch_idx"] = batch_idx

        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        metrics = model.training_step(lr, hr, epoch, use_amp=use_amp, scaler=scaler)

        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        n_batches += 1

        if batch_idx % 20 == 0:
            global_step = epoch * 10000 + batch_idx
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, global_step)
            phase = "warmup" if epoch < model.warmup_epochs else "GAN"
            logger.info(
                f"[{epoch:02d}|{batch_idx:03d}] {phase} "
                f"D={metrics['d_loss']:.4f} G={metrics['g_loss']:.4f} "
                f"L1={metrics['l1_loss']:.6f} E={metrics['energy_loss']:.4f} "
                f"ChE={metrics.get('ch_energy_loss', 0.0):.4f} "
                f"CF={metrics.get('cf_loss', 0.0):.4f} "
                f"Freq={metrics.get('freq_loss', 0.0):.4f}"
            )

    mid_path = os.path.join(ckpt_dir, "srgan_mid_epoch.pt")
    if os.path.exists(mid_path):
        os.remove(mid_path)

    return {k: v / max(n_batches, 1) for k, v in accum.items()}


# ── validation — batched, no OOM ─────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:  SRGAN,
    loader,
    device: torch.device,
    epoch:  int,
    writer: SummaryWriter,
    logger: logging.Logger,
    split:  str = "val",
) -> dict:
    model.generator.eval()
    model.ema_generator.eval()

    CH_NAMES = ["ECAL", "HCAL", "Tracks"]

    e_errs, maes, psnrs, cf_errs = [], [], [], []
    ch_e_errs = {ch: [] for ch in CH_NAMES}

    for lr, hr in loader:
        lr    = lr.to(device, non_blocking=True)
        sr    = model.infer(lr, use_ema=True)   # already clamped [0,1]
        sr_np = sr.cpu().numpy()
        hr_np = hr.numpy()

        eps = 1e-8

        # Total energy error
        e_sr = sr_np.sum(axis=(1, 2, 3))   # (B,)
        e_hr = hr_np.sum(axis=(1, 2, 3))   # (B,)

        e_errs.append(float(np.mean(
            np.abs(e_sr - e_hr) / (e_hr + eps)
        )))

        # Per-channel energy error (ECAL, HCAL, Tracks)
        for c, ch_name in enumerate(CH_NAMES):
            ch_e_sr = sr_np[:, c].sum(axis=(1, 2))   # (B,)
            ch_e_hr = hr_np[:, c].sum(axis=(1, 2))   # (B,)
            ch_e_errs[ch_name].append(float(np.mean(
                np.abs(ch_e_sr - ch_e_hr) / (np.abs(ch_e_hr) + eps)
            )))

        maes.append(float(np.mean(np.abs(sr_np - hr_np))))
        psnrs.append(float(np.mean(
            10 * np.log10(1.0 / (
                np.mean((sr_np - hr_np)**2, axis=(1, 2, 3)) + eps
            ))
        )))
        # Channel fraction: per-channel energy share
        sr_cf = sr_np.sum(axis=(2, 3)) / (e_sr[:, None] + eps)  # (B, C)
        hr_cf = hr_np.sum(axis=(2, 3)) / (e_hr[:, None] + eps)  # (B, C)
        cf_errs.append(float(np.mean(np.abs(sr_cf - hr_cf))))

    results = {
        "energy_conservation": {
            "mean": float(np.mean(e_errs)),
            "std":  float(np.std(e_errs)),
            "max":  float(np.max(e_errs)),
        },
        "per_channel_energy": {
            ch: {
                "mean": float(np.mean(ch_e_errs[ch])),
                "std":  float(np.std(ch_e_errs[ch])),
            }
            for ch in CH_NAMES
        },
        "pixel_mae": {
            "mean": float(np.mean(maes)),
            "std":  float(np.std(maes)),
        },
        "psnr_db": {
            "mean": float(np.mean(psnrs)),
            "std":  float(np.std(psnrs)),
        },
        "channel_fraction": {
            "mean": float(np.mean(cf_errs)),
            "std":  float(np.std(cf_errs)),
        },
    }

    # TensorBoard scalars
    writer.add_scalar(f"{split}/energy_conservation",
                      results["energy_conservation"]["mean"], epoch)
    for ch in CH_NAMES:
        writer.add_scalar(f"{split}/energy_{ch}",
                          results["per_channel_energy"][ch]["mean"], epoch)
    writer.add_scalar(f"{split}/psnr_db",
                      results["psnr_db"]["mean"],             epoch)
    writer.add_scalar(f"{split}/pixel_mae",
                      results["pixel_mae"]["mean"],           epoch)
    writer.add_scalar(f"{split}/channel_fraction",
                      results["channel_fraction"]["mean"],    epoch)

    pce = results["per_channel_energy"]
    logger.info(
        f"[{split.upper()}] epoch {epoch:03d}"
        f"  EnergyErr={results['energy_conservation']['mean']*100:.4f}%"
        f"  ECAL={pce['ECAL']['mean']*100:.2f}%"
        f"  HCAL={pce['HCAL']['mean']*100:.2f}%"
        f"  Tracks={pce['Tracks']['mean']*100:.2f}%"
        f"  PSNR={results['psnr_db']['mean']:.2f}dB"
        f"  MAE={results['pixel_mae']['mean']:.6f}"
        f"  ChanFrac={results['channel_fraction']['mean']*100:.4f}%"
    )
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def _build_loaders(cfg: dict, logger: logging.Logger, fold_idx: int | None = None):
    """Build dataloaders — standard or k-fold."""
    kfold_cfg = cfg.get("kfold", {})
    use_kfold = kfold_cfg.get("enabled", False)

    if use_kfold:
        n_folds = kfold_cfg.get("n_folds", 5)
        logger.info(f"Building {n_folds}-fold stratified CV dataloaders...")
        all_folds = build_kfold_dataloaders(
            train_path    = cfg["data"]["train"],
            val_path      = cfg["data"]["val"],
            test_path     = cfg["data"]["test"],
            n_folds       = n_folds,
            batch_size    = cfg["training"]["batch_size"],
            train_samples = cfg["data"].get("train_samples"),
            val_samples   = cfg["data"].get("val_samples"),
            test_samples  = cfg["data"].get("test_samples"),
            normalise     = cfg["data"].get("normalise", True),
            num_workers   = cfg["data"].get("num_workers", 0),
            seed          = cfg["training"].get("seed", 42),
            energy_bins   = kfold_cfg.get("energy_bins", 5),
        )
        if fold_idx is not None:
            return all_folds[fold_idx]
        return all_folds
    else:
        logger.info("Building dataloaders...")
        return build_dataloaders(
            train_path    = cfg["data"]["train"],
            val_path      = cfg["data"]["val"],
            test_path     = cfg["data"]["test"],
            batch_size    = cfg["training"]["batch_size"],
            train_samples = cfg["data"].get("train_samples"),
            val_samples   = cfg["data"].get("val_samples"),
            test_samples  = cfg["data"].get("test_samples"),
            max_samples   = cfg["data"].get("max_samples"),
            normalise     = cfg["data"].get("normalise", True),
            num_workers   = cfg["data"].get("num_workers", 0),
            cache_dir     = cfg["data"].get("cache_dir", "data/processed"),
        )


def _train_single(
    cfg: dict,
    train_loader,
    val_loader,
    test_loader,
    resume: bool = False,
    fold_tag: str = "",
) -> dict:
    """Core training loop (used by both standard and k-fold paths)."""

    # ── resolve run ID ────────────────────────────────────────────────────────
    if resume and not fold_tag:
        run_id, saved_epoch = load_run_id()
        if run_id is None:
            print("No previous run found. Starting fresh.")
            resume      = False
            run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_epoch = 0
        else:
            print(f"Resuming run {run_id} from epoch {saved_epoch}")
    else:
        run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
        if fold_tag:
            run_id = f"{run_id}_{fold_tag}"
        saved_epoch = 0

    save_run_id(run_id, saved_epoch)

    log_dir  = os.path.join(cfg["results"]["logs"],        run_id)
    ckpt_dir = os.path.join(cfg["results"]["checkpoints"], run_id)
    plot_dir = os.path.join(cfg["results"]["plots"],       run_id)

    for d in [log_dir, ckpt_dir, plot_dir]:
        os.makedirs(d, exist_ok=True)

    logger = setup_logger(log_dir)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))

    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Run ID     : {run_id}")
    logger.info(f"Saved to   : {RUN_ID_FILE}")
    logger.info(f"Log dir    : {log_dir}")
    logger.info(f"Checkpoint : {ckpt_dir}")
    logger.info(f"Device     : {device}")
    if device.type == "cuda":
        logger.info(f"GPU        : {torch.cuda.get_device_name(0)}")

    # ── reproducibility ───────────────────────────────────────────────────────
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    # ── AMP setup ─────────────────────────────────────────────────────────────
    use_amp = cfg["training"].get("use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    logger.info(f"AMP enabled   : {use_amp}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = SRGAN(cfg).to(device)
    total_g = sum(p.numel() for p in model.generator.parameters())
    total_d = sum(p.numel() for p in model.discriminator.parameters())
    logger.info(f"Generator params     : {total_g:,}")
    logger.info(f"Discriminator params : {total_d:,}")

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch  = 0
    start_batch  = 0
    best_psnr    = -np.inf
    best_metrics = {}

    if resume and not fold_tag:
        ckpt_path, tag = find_best_checkpoint(ckpt_dir)
        if ckpt_path:
            start_epoch, start_batch, best_metrics = load_checkpoint(
                model, ckpt_path, device
            )
            if tag == "last":
                start_epoch += 1
                start_batch  = 0
            best_psnr = best_metrics.get(
                "psnr_db", {}
            ).get("mean", -np.inf)
            logger.info(
                f"Loaded [{tag}] checkpoint"
                f"  epoch={start_epoch}"
                f"  batch={start_batch}"
                f"  best PSNR={best_psnr:.4f} dB"
            )
        else:
            logger.info("No checkpoint found — starting from epoch 0")

    epochs = cfg["training"]["epochs"]

    logger.info("=" * 65)
    logger.info(f"TRAINING START{f' — {fold_tag}' if fold_tag else ''}")
    logger.info(f"  Warmup epochs : {model.warmup_epochs}")
    logger.info(f"  Total epochs  : {epochs}")
    logger.info(f"  Start epoch   : {start_epoch}")
    logger.info(f"  Batch size    : {cfg['training']['batch_size']}")
    actual_lr_g = model.opt_g.param_groups[0]['lr']
    actual_lr_d = model.opt_d.param_groups[0]['lr']
    logger.info(f"  Peak LR G / D : {cfg['training']['lr_g']} / {cfg['training']['lr_d']}")
    logger.info(f"  Init LR G / D : {actual_lr_g:.2e} / {actual_lr_d:.2e}  (warmup ramp active)")
    logger.info("=" * 65)

    _EMERGENCY_STATE["model"]    = model
    _EMERGENCY_STATE["cfg"]      = cfg
    _EMERGENCY_STATE["ckpt_dir"] = ckpt_dir
    _EMERGENCY_STATE["run_id"]   = run_id
    _EMERGENCY_STATE["logger"]   = logger

    skip_batches = start_batch

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):

        _EMERGENCY_STATE["epoch"]     = epoch
        _EMERGENCY_STATE["batch_idx"] = 0

        logger.info(f"Epoch {epoch:03d} - training...")
        train_metrics = train_one_epoch(
            model, train_loader, device,
            epoch, logger, writer, cfg, ckpt_dir,
            skip_batches=skip_batches,
            use_amp=use_amp, scaler=scaler,
        )
        skip_batches = 0

        # save immediately after training — before validation
        save_checkpoint(
            model, epoch, {}, cfg, ckpt_dir, tag="last"
        )
        save_run_id(run_id, epoch + 1)

        # clean temp checkpoints
        for temp_tag in ["interrupt", "mid_epoch"]:
            temp_path = os.path.join(
                ckpt_dir, f"srgan_{temp_tag}.pt"
            )
            if os.path.exists(temp_path):
                os.remove(temp_path)

        logger.info(f"  Checkpoint saved (epoch {epoch})")

        # validate — batched, no OOM
        logger.info(f"Epoch {epoch:03d} — validating...")
        val_results = validate(
            model, val_loader, device, epoch, writer, logger, split="val"
        )

        model.scheduler_step()
        lrs = model.get_lr()
        logger.info(
            f"  LR G={lrs['lr_g']:.2e}  LR D={lrs['lr_d']:.2e}"
        )

        # update checkpoint with val metrics
        save_checkpoint(
            model, epoch, val_results, cfg, ckpt_dir, tag="last"
        )

        # save best
        current_psnr = val_results["psnr_db"]["mean"]
        if current_psnr > best_psnr:
            best_psnr    = current_psnr
            best_metrics = val_results
            save_checkpoint(
                model, epoch, val_results, cfg, ckpt_dir, tag="best"
            )
            logger.info(
                f"  New best PSNR: {best_psnr:.2f} dB (epoch {epoch})"
            )

        # periodic backup every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, epoch, val_results, cfg,
                ckpt_dir, tag=f"epoch_{epoch:03d}",
            )
            logger.info(
                f"  Periodic checkpoint saved (epoch {epoch})"
            )

        writer.add_scalar(
            "epoch/train_g_loss", train_metrics.get("g_loss", 0), epoch
        )
        writer.add_scalar(
            "epoch/train_d_loss", train_metrics.get("d_loss", 0), epoch
        )
        writer.add_scalar("epoch/best_psnr", best_psnr, epoch)
        writer.add_scalar(
            "epoch/lr_g",
            model.opt_g.param_groups[0]["lr"], epoch
        )

    # ── final test evaluation ─────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("FINAL TEST EVALUATION — Dataset-3 (Generalization)")
    logger.info("=" * 65)

    best_ckpt = os.path.join(ckpt_dir, "srgan_best.pt")
    if os.path.exists(best_ckpt):
        load_checkpoint(model, best_ckpt, device)
        logger.info("Loaded best checkpoint for test evaluation")

    test_results = validate(
        model, test_loader, device, epochs, writer, logger, split="test"
    )

    def _serialise(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        return obj

    results_path = os.path.join(log_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(_serialise(test_results), f, indent=2, default=float)

    logger.info(f"Test results saved : {results_path}")
    logger.info(f"Training complete{f' — {fold_tag}' if fold_tag else ''}.")
    writer.close()
    return test_results


def train(cfg: dict, resume: bool = False) -> None:
    """Main entry: standard training or k-fold cross-validation."""
    kfold_cfg = cfg.get("kfold", {})
    use_kfold = kfold_cfg.get("enabled", False)

    # Temporary logger for data loading
    tmp_logger = logging.getLogger("srgan_setup")
    tmp_logger.setLevel(logging.INFO)
    if not tmp_logger.handlers:
        tmp_logger.addHandler(logging.StreamHandler(sys.stdout))

    if use_kfold:
        n_folds = kfold_cfg.get("n_folds", 5)
        fold_loaders = _build_loaders(cfg, tmp_logger)
        all_results = []

        for fold_idx, (tr_loader, vl_loader, te_loader) in enumerate(fold_loaders):
            print(f"\n{'='*65}")
            print(f"  FOLD {fold_idx + 1} / {n_folds}")
            print(f"{'='*65}\n")
            result = _train_single(
                cfg, tr_loader, vl_loader, te_loader,
                resume=False, fold_tag=f"fold{fold_idx}",
            )
            all_results.append(result)

        # ── aggregate k-fold results ──────────────────────────────────
        print("\n" + "=" * 65)
        print("K-FOLD CROSS-VALIDATION SUMMARY")
        print("=" * 65)
        for key in ["energy_conservation", "psnr_db", "pixel_mae", "channel_fraction"]:
            means = [r[key]["mean"] for r in all_results]
            scale = 100.0 if key in ("energy_conservation", "channel_fraction") else 1.0
            unit  = "%" if key in ("energy_conservation", "channel_fraction") else (
                "dB" if key == "psnr_db" else ""
            )
            print(f"  {key:<28}: {np.mean(means)*scale:.4f}{unit} ± {np.std(means)*scale:.4f}{unit}")

        # Per-channel energy across folds
        ch_names = ["ECAL", "HCAL", "Tracks"]
        print(f"\n  Per-channel energy error (across folds):")
        for ch in ch_names:
            ch_means = [r["per_channel_energy"][ch]["mean"] for r in all_results]
            print(f"    {ch:<8}: {np.mean(ch_means)*100:.4f}% ± {np.std(ch_means)*100:.4f}%")
        print("=" * 65)

    else:
        loaders = _build_loaders(cfg, tmp_logger)
        train_loader, val_loader, test_loader = loaders
        _train_single(cfg, train_loader, val_loader, test_loader, resume=resume)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SRGAN for calorimeter jet super-resolution"
    )
    parser.add_argument(
        "--config", type=str, default="configs/srgan.yaml",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Auto-resume from last checkpoint",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume=args.resume)