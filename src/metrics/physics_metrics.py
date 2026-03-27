"""
physics_metrics.py
──────────────────
Physics-motivated evaluation metrics for calorimeter jet super-resolution.

Metrics
───────
    1. Energy Conservation Error  — fractional |E_sr - E_hr| / E_hr
    2. Radial Energy Profile      — mean energy vs distance from jet centre
    3. Pixel-wise MAE             — mean absolute pixel error
    4. PSNR                       — peak signal-to-noise ratio
    5. Channel Fraction Error     — per-channel energy fraction deviation
    6. Wasserstein Distance       — distribution-level comparison per channel

Design choices (for GSoC discussion)
──────────────────────────────────────
    Standard SR metrics (PSNR, SSIM) measure visual quality but ignore
    physics. For CMS reconstruction, energy conservation and shower
    spatial structure are the ground truth constraints. This module
    evaluates both visual and physics correctness so the GAN can be
    assessed as a physics tool, not just an image model.
"""

import numpy as np
import torch
from torch import Tensor
from scipy.stats import wasserstein_distance


# ── tensor → numpy utility ────────────────────────────────────────────────────

def _to_numpy(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# ── 1. Energy Conservation Error ──────────────────────────────────────────────

def energy_conservation_error(sr: np.ndarray, hr: np.ndarray) -> dict:
    """
    Fractional energy conservation error per jet.

        err = |E_sr - E_hr| / (E_hr + eps)

    Parameters
    ----------
    sr : (N, C, H, W)  super-resolved images
    hr : (N, C, H, W)  real HR images

    Returns
    -------
    dict with keys: mean, std, max  (all float)
    """
    sr, hr = _to_numpy(sr), _to_numpy(hr)
    eps    = 1e-8
    e_sr   = sr.sum(axis=(1, 2, 3))     # (N,)
    e_hr   = hr.sum(axis=(1, 2, 3))     # (N,)
    err    = np.abs(e_sr - e_hr) / (np.abs(e_hr) + eps)
    return {
        "mean": float(err.mean()),
        "std":  float(err.std()),
        "max":  float(err.max()),
    }


def per_channel_energy_error(
    sr: np.ndarray,
    hr: np.ndarray,
    ch_names: list | None = None,
) -> dict:
    """
    Per-channel fractional energy conservation error.

    Disentangles energy error across ECAL, HCAL, Tracks so we can
    verify each channel is within expected experimental ranges
    (ECAL ~2.5-5%, HCAL up to ~20%).

    Parameters
    ----------
    sr : (N, C, H, W)
    hr : (N, C, H, W)
    ch_names : channel labels

    Returns
    -------
    dict with per-channel mean/std/max and overall breakdown
    """
    sr, hr = _to_numpy(sr), _to_numpy(hr)
    if ch_names is None:
        ch_names = ["ECAL", "HCAL", "Tracks"]

    eps = 1e-8
    result = {}

    for c, name in enumerate(ch_names[:sr.shape[1]]):
        e_sr = sr[:, c].sum(axis=(1, 2))    # (N,)
        e_hr = hr[:, c].sum(axis=(1, 2))    # (N,)
        err  = np.abs(e_sr - e_hr) / (np.abs(e_hr) + eps)
        result[name] = {
            "mean": float(err.mean()),
            "std":  float(err.std()),
            "max":  float(err.max()),
        }

    return result


# ── 2. Radial Energy Profile ──────────────────────────────────────────────────

def _radial_profile_single(image: np.ndarray, n_bins: int = 25) -> tuple:
    """
    Radial energy profile for a single (H, W) channel image.

    Returns
    -------
    centres : (n_bins,) normalised radial bin centres in [0, 1]
    profile : (n_bins,) mean energy per radial bin
    """
    H, W   = image.shape
    cy, cx = H / 2.0, W / 2.0
    yi, xi = np.indices((H, W))
    r      = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)
    r_max  = np.sqrt(cx ** 2 + cy ** 2)
    bins   = np.linspace(0, r_max, n_bins + 1)
    centres = (bins[:-1] + bins[1:]) / 2.0 / r_max   # normalised

    profile = np.array([
        image[(r >= bins[b]) & (r < bins[b + 1])].mean()
        if np.any((r >= bins[b]) & (r < bins[b + 1])) else 0.0
        for b in range(n_bins)
    ], dtype=np.float32)

    return centres, profile


def radial_energy_profile(
    images:  np.ndarray,
    channel: int = 0,
    n_bins:  int = 25,
) -> dict:
    """
    Mean radial energy profile across a batch of jet images.

    Parameters
    ----------
    images  : (N, C, H, W)
    channel : which detector channel to profile (0=ECAL, 1=HCAL, 2=Tracks)
    n_bins  : number of radial bins

    Returns
    -------
    dict with keys: centres, mean, std  — all (n_bins,) arrays
    """
    images = _to_numpy(images)
    profiles = np.stack([
        _radial_profile_single(images[n, channel], n_bins)[1]
        for n in range(len(images))
    ])
    centres = _radial_profile_single(images[0, channel], n_bins)[0]
    return {
        "centres": centres,
        "mean":    profiles.mean(axis=0),
        "std":     profiles.std(axis=0),
    }


def radial_profile_error(
    sr:      np.ndarray,
    hr:      np.ndarray,
    channel: int = 0,
    n_bins:  int = 25,
) -> dict:
    """
    Mean absolute difference between SR and HR radial profiles.

    Parameters
    ----------
    sr, hr  : (N, C, H, W)
    channel : detector channel index

    Returns
    -------
    dict with keys: mae (scalar), profile_sr, profile_hr (arrays)
    """
    sr, hr  = _to_numpy(sr), _to_numpy(hr)
    prof_sr = radial_energy_profile(sr, channel, n_bins)
    prof_hr = radial_energy_profile(hr, channel, n_bins)
    mae     = float(np.abs(prof_sr["mean"] - prof_hr["mean"]).mean())
    return {
        "mae":        mae,
        "profile_sr": prof_sr,
        "profile_hr": prof_hr,
    }


# ── 3. Pixel-wise MAE ─────────────────────────────────────────────────────────

def pixel_mae(sr: np.ndarray, hr: np.ndarray) -> dict:
    """
    Per-jet pixel-wise mean absolute error.

    Parameters
    ----------
    sr, hr : (N, C, H, W)

    Returns
    -------
    dict with keys: mean, std  (float)
    """
    sr, hr = _to_numpy(sr), _to_numpy(hr)
    err    = np.abs(sr - hr).mean(axis=(1, 2, 3))   # (N,)
    return {"mean": float(err.mean()), "std": float(err.std())}


# ── 4. PSNR ───────────────────────────────────────────────────────────────────

def psnr(sr: np.ndarray, hr: np.ndarray) -> dict:
    """
    Peak signal-to-noise ratio between SR and HR images.

    Parameters
    ----------
    sr, hr : (N, C, H, W)  values assumed in [0, 1] after normalisation

    Returns
    -------
    dict with keys: mean, std  (float, dB)
    """
    sr, hr  = _to_numpy(sr), _to_numpy(hr)
    mse     = ((sr - hr) ** 2).mean(axis=(1, 2, 3))    # (N,)
    max_val = max(hr.max(), 1e-8)
    psnr_db = 10.0 * np.log10(max_val ** 2 / (mse + 1e-8))
    return {"mean": float(psnr_db.mean()), "std": float(psnr_db.std())}


# ── 5. Channel Fraction Error ─────────────────────────────────────────────────

def channel_fraction_error(sr: np.ndarray, hr: np.ndarray) -> dict:
    """
    Mean absolute error in per-channel energy fractions.

    Measures whether the GAN preserves the relative energy balance
    across ECAL, HCAL, and Tracks channels.

    Parameters
    ----------
    sr, hr : (N, C, H, W)

    Returns
    -------
    dict with keys: per_channel (C,), mean (float)
    """
    sr, hr  = _to_numpy(sr), _to_numpy(hr)

    def fractions(x):
        ch_e  = x.sum(axis=(2, 3))                      # (N, C)
        total = ch_e.sum(axis=1, keepdims=True)
        return ch_e / (total + 1e-8)                    # (N, C)

    frac_sr = fractions(sr)
    frac_hr = fractions(hr)
    err     = np.abs(frac_sr - frac_hr).mean(axis=0)   # (C,)

    return {
        "per_channel": err,
        "mean":        float(err.mean()),
    }


# ── 6. Wasserstein Distance ───────────────────────────────────────────────────

def wasserstein_pixel_distance(
    sr:      np.ndarray,
    hr:      np.ndarray,
    channel: int = 0,
) -> float:
    """
    1-D Wasserstein distance between flattened pixel distributions
    of SR and HR for a given channel.

    Captures distribution-level differences that MAE/PSNR miss.

    Parameters
    ----------
    sr, hr  : (N, C, H, W)
    channel : detector channel index

    Returns
    -------
    float — Wasserstein-1 distance
    """
    sr, hr = _to_numpy(sr), _to_numpy(hr)
    p = sr[:, channel].flatten()
    q = hr[:, channel].flatten()
    return float(wasserstein_distance(p, q))


# ── full evaluation suite ─────────────────────────────────────────────────────

def evaluate(
    sr:       np.ndarray,
    hr:       np.ndarray,
    n_bins:   int = 25,
    ch_names: list = None,
) -> dict:
    """
    Run all physics metrics and return a single results dict.

    Parameters
    ----------
    sr       : (N, C, H, W) super-resolved images
    hr       : (N, C, H, W) real HR images
    n_bins   : radial profile bins
    ch_names : channel labels for display

    Returns
    -------
    dict — all metric results
    """
    if ch_names is None:
        ch_names = ["ECAL", "HCAL", "Tracks"]

    results = {}

    results["energy_conservation"]    = energy_conservation_error(sr, hr)
    results["per_channel_energy"]     = per_channel_energy_error(sr, hr, ch_names)
    results["pixel_mae"]              = pixel_mae(sr, hr)
    results["psnr_db"]                = psnr(sr, hr)
    results["channel_fraction"]       = channel_fraction_error(sr, hr)

    results["radial_profiles"] = {
        ch: radial_profile_error(sr, hr, channel=c, n_bins=n_bins)
        for c, ch in enumerate(ch_names)
    }

    results["wasserstein"] = {
        ch: wasserstein_pixel_distance(sr, hr, channel=c)
        for c, ch in enumerate(ch_names)
    }

    return results


def print_report(results: dict, ch_names: list = None) -> None:
    """Pretty-print the evaluation report."""
    if ch_names is None:
        ch_names = ["ECAL", "HCAL", "Tracks"]

    print("=" * 60)
    print("PHYSICS METRICS REPORT")
    print("=" * 60)

    ec = results["energy_conservation"]
    print(f"\nEnergy Conservation Error (total)")
    print(f"  mean : {ec['mean']*100:.4f}%")
    print(f"  std  : {ec['std']*100:.4f}%")
    print(f"  max  : {ec['max']*100:.4f}%")

    pce = results.get("per_channel_energy", {})
    if pce:
        print(f"\nPer-Channel Energy Error")
        for ch_name, vals in pce.items():
            print(f"  {ch_name:<8}: mean={vals['mean']*100:.4f}%  "
                  f"std={vals['std']*100:.4f}%  max={vals['max']*100:.4f}%")

    pm = results["pixel_mae"]
    print(f"\nPixel MAE")
    print(f"  mean : {pm['mean']:.6f}")
    print(f"  std  : {pm['std']:.6f}")

    ps = results["psnr_db"]
    print(f"\nPSNR")
    print(f"  mean : {ps['mean']:.2f} dB")
    print(f"  std  : {ps['std']:.2f} dB")

    cf = results["channel_fraction"]
    print(f"\nChannel Fraction Error")
    for ch, err in zip(ch_names, cf["per_channel"]):
        print(f"  {ch:<8}: {err*100:.4f}%")
    print(f"  mean  : {cf['mean']*100:.4f}%")

    print(f"\nRadial Profile MAE")
    for ch in ch_names:
        print(f"  {ch:<8}: {results['radial_profiles'][ch]['mae']:.6f}")

    print(f"\nWasserstein Distance (pixel distribution)")
    for ch in ch_names:
        print(f"  {ch:<8}: {results['wasserstein'][ch]:.6f}")

    print("=" * 60)


if __name__ == "__main__":
    sr = np.random.rand(16, 3, 125, 125).astype(np.float32)
    hr = np.random.rand(16, 3, 125, 125).astype(np.float32)

    results = evaluate(sr, hr)
    print_report(results)
    print("physics_metrics.py — OK")