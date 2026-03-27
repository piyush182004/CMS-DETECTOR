"""
parquet_loader.py
─────────────────
Efficient data loading for calorimeter jet images.

Architecture
────────────
    Train  : Chunked loading with disk caching (memory-efficient)
    Val    : Pure streaming (no disk I/O)
    Test   : Pure streaming (no disk I/O)

Normalization strategy (energy-preserving)
──────────────────────────────────────────
    PAIRED normalisation: both LR and HR are divided by max(HR) so that
    the energy scale is preserved between the two.  Independent per-image
    normalisation was the root cause of the ~13-14% energy error because
    max(LR) ≠ max(HR), which rescales the total energy differently.
"""

from __future__ import annotations

import os
import logging
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, Dataset

from sklearn.model_selection import StratifiedKFold

__all__ = ["build_dataloaders", "build_kfold_dataloaders"]

log = logging.getLogger(__name__)

CHUNK_SIZE = 2500
STREAM_SIZE = 500


def _extract_image(nested_array) -> np.ndarray:
    """Extract 3-channel image from nested parquet structure."""
    return np.stack([
        np.array([np.array(row, dtype=np.float32) for row in ch], dtype=np.float32)
        for ch in nested_array
    ])


def normalize_pair(lr: np.ndarray, hr: np.ndarray, eps: float = 1e-8):
    """Paired max normalization — SAME scale for LR and HR.

    Both images are divided by max(HR) so that:
        1) Values are in a numerically convenient [0, ~1] range.
        2) The energy ratio  sum(LR) / sum(HR) is PRESERVED exactly.
        3) Inter-channel ratios are preserved in both images.

    This fixes the ~13-14% energy error caused by independent
    normalisation where max(LR) ≠ max(HR) broke the energy scale.
    """
    scale = hr.max()
    if scale > eps:
        return lr / scale, hr / scale
    return lr, hr


def normalize(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-image max normalization — maps to [0, 1].

    ONLY used at inference time when HR is not available.
    For training, use normalize_pair() instead.
    """
    vmax = image.max()
    return image / (vmax + eps) if vmax > eps else image


class ChunkedTrainDataset(IterableDataset):
    """Memory-efficient training dataset with disk caching."""

    def __init__(
        self,
        path: str,
        chunk_dir: str,
        max_samples: int | None = None,
        normalise: bool = True,
        shuffle: bool = True,
    ):
        self.path = path
        self.chunk_dir = chunk_dir
        self.max_samples = max_samples
        self.normalise = normalise
        self.shuffle = shuffle

        os.makedirs(chunk_dir, exist_ok=True)
        self._clean()

        pf = pq.ParquetFile(path)
        self._total = min(pf.metadata.num_rows, max_samples or float("inf"))

    def _clean(self) -> None:
        for f in os.listdir(self.chunk_dir):
            if f.startswith("chunk_") or f.startswith("train_chunk_"):
                os.remove(os.path.join(self.chunk_dir, f))

    def __len__(self) -> int:
        return int(self._total)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        pf = pq.ParquetFile(self.path)
        total, chunk_idx, prev_path = 0, 0, None

        for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
            df = batch.to_pandas()
            n = min(len(df), (self.max_samples or float("inf")) - total)
            if n <= 0:
                break

            lr_arr = np.stack([_extract_image(df["X_jets_LR"].iloc[i]) for i in range(int(n))])
            hr_arr = np.stack([_extract_image(df["X_jets"].iloc[i]) for i in range(int(n))])

            if self.normalise:
                # Paired normalization — divide both by max(HR) to preserve energy scale
                pairs = [normalize_pair(lr_arr[i], hr_arr[i]) for i in range(len(lr_arr))]
                lr_arr = np.stack([p[0] for p in pairs])
                hr_arr = np.stack([p[1] for p in pairs])

            curr_path = os.path.join(self.chunk_dir, f"chunk_{chunk_idx:04d}.pt")
            torch.save({"lr": lr_arr, "hr": hr_arr}, curr_path)

            if prev_path and os.path.exists(prev_path):
                os.remove(prev_path)

            indices = np.random.permutation(int(n)) if self.shuffle else np.arange(int(n))
            chunk = torch.load(curr_path, weights_only=False)

            for i in indices:
                yield torch.from_numpy(chunk["lr"][i]), torch.from_numpy(chunk["hr"][i])

            total += int(n)
            prev_path = curr_path
            chunk_idx += 1

        if prev_path and os.path.exists(prev_path):
            os.remove(prev_path)


class StreamingDataset(IterableDataset):
    """Pure streaming dataset — no disk writes."""

    def __init__(
        self,
        path: str,
        max_samples: int | None = None,
        normalise: bool = True,
    ):
        self.path = path
        self.max_samples = max_samples
        self.normalise = normalise

        pf = pq.ParquetFile(path)
        self._total = min(pf.metadata.num_rows, max_samples or float("inf"))

    def __len__(self) -> int:
        return int(self._total)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        pf = pq.ParquetFile(self.path)
        total = 0

        for batch in pf.iter_batches(batch_size=STREAM_SIZE):
            df = batch.to_pandas()
            n = min(len(df), (self.max_samples or float("inf")) - total)
            if n <= 0:
                break

            for i in range(int(n)):
                lr = _extract_image(df["X_jets_LR"].iloc[i])
                hr = _extract_image(df["X_jets"].iloc[i])

                if self.normalise:
                    lr, hr = normalize_pair(lr, hr)

                yield torch.from_numpy(lr), torch.from_numpy(hr)

            total += int(n)


def build_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 16,
    train_samples: int | None = None,
    val_samples: int | None = None,
    test_samples: int | None = None,
    normalise: bool = True,
    num_workers: int = 0,
    cache_dir: str = "data/processed",
    chunk_dir: str | None = None,
    max_samples: int | None = None,  # Legacy
    **kwargs,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders with separate sample limits."""
    chunk_dir = chunk_dir or os.path.join(cache_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    train_n = train_samples or max_samples
    val_n = val_samples or max_samples
    test_n = test_samples or max_samples

    train_ds = ChunkedTrainDataset(train_path, chunk_dir, train_n, normalise, shuffle=True)
    val_ds = StreamingDataset(val_path, val_n, normalise)
    test_ds = StreamingDataset(test_path, test_n, normalise)

    log.info(f"Dataloaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return DataLoader(train_ds, **kw), DataLoader(val_ds, **kw), DataLoader(test_ds, **kw)


# ── In-memory dataset for k-fold (fits in RAM for small data) ─────────────────

class InMemoryPairedDataset(Dataset):
    """Holds preloaded LR/HR arrays for k-fold splitting."""

    def __init__(self, lr_arr: np.ndarray, hr_arr: np.ndarray):
        self.lr = lr_arr
        self.hr = hr_arr

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        return torch.from_numpy(self.lr[idx]), torch.from_numpy(self.hr[idx])


def _load_all_samples(
    paths: list[str],
    max_samples: int | None,
    normalise: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all LR/HR pairs from one or more parquet files into RAM."""
    all_lr, all_hr = [], []
    total = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
            df = batch.to_pandas()
            n = len(df)
            if max_samples is not None:
                n = min(n, max_samples - total)
            if n <= 0:
                break
            for i in range(int(n)):
                lr = _extract_image(df["X_jets_LR"].iloc[i])
                hr = _extract_image(df["X_jets"].iloc[i])
                if normalise:
                    lr, hr = normalize_pair(lr, hr)
                all_lr.append(lr)
                all_hr.append(hr)
            total += int(n)
            if max_samples is not None and total >= max_samples:
                break
    return np.stack(all_lr), np.stack(all_hr)


def _energy_bins(hr_arr: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Bin samples by total HR energy for stratified splitting."""
    energies = hr_arr.sum(axis=(1, 2, 3))
    bins = np.quantile(energies, np.linspace(0, 1, n_bins + 1))
    labels = np.digitize(energies, bins[1:-1])  # 0 .. n_bins-1
    return labels


def build_kfold_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    n_folds: int = 5,
    batch_size: int = 16,
    train_samples: int | None = None,
    val_samples: int | None = None,
    test_samples: int | None = None,
    normalise: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    energy_bins: int = 5,
    **kwargs,
) -> list[tuple[DataLoader, DataLoader, DataLoader]]:
    """Build stratified k-fold dataloaders.

    Combines train + val data, stratifies by total HR energy into
    n_folds folds.  Each fold yields (train_loader, val_loader, test_loader).
    The test set is shared across all folds (held-out evaluation).

    Returns
    -------
    List of (train_loader, val_loader, test_loader) tuples, one per fold.
    """
    log.info(f"Loading data for {n_folds}-fold cross-validation...")

    # Pool train + val data
    pool_max = None
    if train_samples and val_samples:
        pool_max = train_samples + val_samples
    elif train_samples:
        pool_max = train_samples

    lr_all, hr_all = _load_all_samples([train_path, val_path], pool_max, normalise)
    log.info(f"Pooled data: {len(lr_all)} samples")

    # Stratify by HR total energy
    strat_labels = _energy_bins(hr_all, n_bins=energy_bins)

    # Separate test set
    lr_test, hr_test = _load_all_samples([test_path], test_samples, normalise)
    test_ds = InMemoryPairedDataset(lr_test, hr_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=False, shuffle=False)

    # K-fold splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_loaders = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(lr_all, strat_labels)):
        train_ds = InMemoryPairedDataset(lr_all[train_idx], hr_all[train_idx])
        val_ds   = InMemoryPairedDataset(lr_all[val_idx], hr_all[val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=False, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=False, shuffle=False)

        log.info(f"  Fold {fold_idx}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        fold_loaders.append((train_loader, val_loader, test_loader))

    return fold_loaders