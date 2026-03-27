# Super-Resolution at the CMS Detector — GSoC 2026

**Applicant:** Piyush Mondal  
**Organisation:** ML4Sci / CMS Experiment  
**Task:** Train a Generative Adversarial Network to super-resolve calorimeter jet images from 64×64 (low-resolution) to 125×125 (high-resolution).


## Important Info

For experimenting purpose I have used 5000 train,500 test,500 valid sets of data but to use all make sure to change in "C:\GSOC-CMS-2K26\configs\srgan.yaml" samples to null(also mentioned in comments of the code)

To run the training process make sure to run in the terminal after every installations
run command "python -m src.train --config configs/srgan.yaml"

for the testing of the inference of the trained model
choose the best or last model from "C:\GSOC-CMS-2K26\results\checkpoints"
example if i trained on fold1 epochs 50 
I can use "C:\GSOC-CMS-2K26\results\checkpoints\srgan_epoch_049.pt"

and run using the command
"python .\test-output.py --ckpt "C:\GSOC-CMS-2K26\results\checkpoints\20260327_164141_fold0\srgan_epoch_049.pt""
the plots results will be saved at 
"C:\GSOC-CMS-2K26\results\plots"
which can be viewed

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Model Architecture](#model-architecture)
5. [Loss Functions](#loss-functions)
6. [Physics Metrics](#physics-metrics)
7. [Training Pipeline](#training-pipeline)
8. [Installation](#installation)
9. [How to Train](#how-to-train)
10. [How to Run Inference (test-output.py)](#how-to-run-inference)
11. [Configuration Reference](#configuration-reference)
12. [Design Choices & Discussion](#design-choices--discussion)

---

## Project Overview

The CMS detector at CERN records particle collision events where jets of particles (quarks and gluons) deposit energy across a calorimeter. This project trains an **ESRGAN-style Super-Resolution GAN** to reconstruct high-resolution (125×125) calorimeter images from their low-resolution (64×64) counterparts — preserving physical quantities like total energy, per-channel energy balance, and shower spatial structure.

**Why super-resolution for physics?**  
Standard SR metrics like PSNR measure visual quality. In a physics context, what matters equally is that the model **conserves energy** and reproduces the **shower geometry** of each calorimeter channel (ECAL, HCAL, Tracks). This implementation enforces those constraints directly through physics-aware losses and evaluation metrics.

---

## Dataset

| Property          | Details |
|-------------------|---------|
| **Source**        | [CERNBox Dataset](https://cernbox.cern.ch/s/EYgmOkI9BjwxNqy) |
| **Format**        | Apache Parquet files |
| **Classes**       | Quarks and Gluons impinging on the CMS calorimeter |
| **LR images**     | `X_jets_LR` — 64×64 pixels, 3 channels |
| **HR images**     | `X_jets` — 125×125 pixels, 3 channels |
| **Channels**      | ECAL (Electromagnetic Calorimeter), HCAL (Hadronic Calorimeter), Tracks |

Three Parquet files are used (train / validation / test split):

```
QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272_LR.parquet   ← training
QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540_LR.parquet   ← validation
QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494_LR.parquet   ← test / inference
```

Place these files at the paths specified in `configs/srgan.yaml` (under the `data:` section).

---

## Project Structure

```
GSOC-CMS-2K26/
│
├── configs/
│   └── srgan.yaml              # All training hyperparameters and paths
│
├── src/
│   ├── train.py                # Main training loop (run this to train)
│   ├── data/
│   │   └── parquet_loader.py   # Parquet dataset loader with paired normalisation
│   ├── models/
│   │   ├── generator.py        # RRDB-CA Generator (ESRGAN-style)
│   │   ├── discriminator.py    # PatchGAN Discriminator with spectral norm
│   │   └── gan.py              # SRGAN training wrapper (optimisers, schedulers, EMA)
│   ├── losses/
│   │   └── sr_loss.py          # Physics-aware loss functions (RaGAN-LS + physics)
│   └── metrics/
│       └── physics_metrics.py  # Energy, PSNR, Wasserstein, radial profile metrics
│
├── notebook/
│   ├── baseline_upsampling.ipynb       # Bicubic / bilinear baseline comparison
│   ├── data_check_notebook_fixed.ipynb # Dataset exploration and sanity checks
│   └── physics_metrics.ipynb           # Physics metric visualisations
│
├── test-output.py              # Inference script — loads checkpoint, plots results
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Model Architecture

### Generator — RRDB-CA (`src/models/generator.py`)

Based on **ESRGAN** (Wang et al., ECCV Workshop 2018) with physics-specific modifications.

```
Input  : (B, 3, 64, 64)    ← normalised LR jet image
Output : (B, 3, 125, 125)  ← super-resolved HR jet image
```

| Component | Detail |
|-----------|--------|
| **Head** | Single 3×3 conv for shallow feature extraction |
| **Body** | 6× RRDB blocks (Residual-in-Residual Dense Blocks) |
| **RRDB** | Each block contains 3 Residual Dense Sub-blocks + Channel Attention (SE) |
| **Upsampler** | Sub-pixel convolution (PixelShuffle ×2) → centre-crop from 128→125 |
| **Tail** | 3×3 conv + ReLU (enforces non-negative energy output) |
| **Normalisation** | No BatchNorm (following EDSR/ESRGAN — avoids range distortion) |
| **Activation** | LeakyReLU (avoids dead neurons on sparse calorimeter data) |

**Channel Attention (SE block):** Squeeze-and-Excitation applied after each RRDB group, allowing the network to focus independently on ECAL, HCAL, and Tracks channels — the most physically meaningful distinction in calorimeter data.

### Discriminator — PatchGAN (`src/models/discriminator.py`)

```
Input  : (B, 3, 125, 125)  ← real HR or generated SR image
Output : (B, 1, H', W')    ← patch-wise logit map
```

| Component | Detail |
|-----------|--------|
| **Architecture** | 4-layer PatchGAN (70×70 receptive field) |
| **Normalisation** | Spectral Normalisation on all conv layers |
| **Norm type** | InstanceNorm (more stable than BatchNorm for small batches) |
| **Activation** | LeakyReLU throughout |
| **Output** | No sigmoid — LSGAN-style raw logits |

The **~70×70 pixel receptive field** is well-matched to typical jet shower widths in the 125×125 calorimeter images, making the patch discrimination physically meaningful.

---

## Loss Functions

Defined in `src/losses/sr_loss.py`. The total generator loss is:

```
L_G = L1
    + λ_adv       × L_adv        (RaGAN-LS adversarial)
    + λ_energy    × L_energy      (total energy conservation)
    + λ_ch_energy × L_ch_energy   (per-channel energy: ECAL, HCAL, Tracks)
    + λ_cf        × L_cf          (channel energy fraction balance)
    + λ_freq      × L_freq        (spectral frequency loss)
```

| Loss | Weight | Purpose |
|------|--------|---------|
| `L1` | 1.0 | Pixel-level reconstruction fidelity |
| `L_adv` (RaGAN-LS) | 0.01 | Relativistic average GAN — prevents discriminator collapse |
| `L_energy` | 1.0 | Total energy conservation: `log(E_sr / E_hr)²` |
| `L_ch_energy` | 0.5 | Per-channel energy (ECAL/HCAL/Tracks) independently conserved |
| `L_cf` | 0.5 | Channel fraction: relative energy balance across channels |
| `L_freq` | 0.1 | 2D FFT loss — penalises missing high-frequency details |

**Why RaGAN-LS (Relativistic average GAN)?**  
Standard LSGAN can suffer discriminator collapse where `D_loss → 0` and gradients vanish. RaGAN makes the discriminator predict *relative* realism ("is real MORE realistic than fake?"), providing stable gradients throughout training even when the generator improves.

---

## Physics Metrics

Defined in `src/metrics/physics_metrics.py`. Both standard image metrics and custom physics metrics are computed:

| Metric | What it measures |
|--------|-----------------|
| **PSNR** | Peak signal-to-noise ratio (image quality, dB) |
| **Pixel MAE** | Mean absolute per-pixel error |
| **Energy Conservation Error** | `|E_sr - E_hr| / E_hr` — total energy fidelity |
| **Per-channel Energy Error** | ECAL, HCAL, Tracks energy errors independently |
| **Channel Fraction Error** | Relative energy balance across detector channels |
| **Radial Energy Profile** | Mean energy vs distance from jet centre (shower shape) |
| **Wasserstein Distance** | Distribution-level pixel comparison per channel |

Standard image metrics alone miss whether the generator is physically correct. A blurry image can have low energy error; a sharp image can have poor energy conservation. Both need to be tracked.

---

## Training Pipeline

Defined in `src/train.py`. Key design decisions:

| Feature | Detail |
|---------|--------|
| **Warm-up phase** | 5 epochs of L1-only pre-training before the adversarial loss is enabled |
| **LR schedule** | Linear warm-up → constant hold → cosine decay to `1e-6` |
| **EMA weights** | Exponential moving average (decay=0.999) of generator weights used for inference |
| **AMP** | PyTorch automatic mixed-precision (`use_amp: true`) |
| **Stratified K-Fold CV** | 5-fold cross-validation stratified by total HR energy (handles limited data) |
| **Paired normalisation** | Both LR and HR divided by `max(HR)` — fixes energy scale error |
| **Gradient clipping** | `max_norm=1.0` on generator for stable training |

---

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd GSOC-CMS-2K26
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install numpy pyarrow torch pyyaml scipy tensorboard pandas scikit-learn
   ```

4. **Download the dataset**  
   Download the Parquet files from [CERNBox](https://cernbox.cern.ch/s/EYgmOkI9BjwxNqy) and update the paths in `configs/srgan.yaml` under the `data:` section.

---

## How to Train

Run the training script using the provided config:

```bash
python -m src.train --config configs/srgan.yaml
```

**What happens during training:**
- Epochs 1–5: Generator is pre-trained with L1 + physics losses only (no adversarial loss yet)
- Epochs 6–50: Full GAN training with all losses enabled
- Checkpoints are saved to `results/checkpoints/`
- Training plots are saved to `results/plots/`
- TensorBoard logs are saved to `results/logs/`

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir results/logs
```

---

## How to Run Inference

After training, use `test-output.py` to load the best saved checkpoint and visualise results on the held-out **test set**.

### Basic usage (auto-detects best checkpoint):
```bash
python test-output.py
```

### With explicit config:
```bash
python test-output.py --config configs/srgan.yaml
```

### With a specific checkpoint:
```bash
python test-output.py --ckpt results/checkpoints/<fold>/srgan_best.pt
```

**What the script does:**
1. Loads the best (or last) checkpoint from `results/checkpoints/`
2. Loads 4 sample jets from the **test** Parquet file (unseen during training)
3. Runs the EMA generator in inference mode
4. Prints per-jet metrics to the console:
   - PSNR (dB)
   - Total energy conservation error (%)
   - Channel fraction error (%)
   - Per-channel energy error for ECAL, HCAL, Tracks (%)
5. Saves a side-by-side comparison plot: **LR input → SR output → HR ground truth** for each detector channel
6. Plot is saved to `results/plots/test_output-..._epoch_XXX.png`

**Example console output:**
```
── Per-jet metrics ──────────────────────────
  Jet 0 | PSNR=28.45dB | EnergyErr=3.21% | ChanFrac=1.05%
  Jet 1 | PSNR=27.91dB | EnergyErr=2.87% | ChanFrac=0.98%
  ...
  Mean PSNR        : 28.12 dB
  Mean EnergyErr   : 3.04%
  Mean ChanFrac    : 1.01%

── Per-channel energy error ─────────────────
  ECAL    : 2.54%
  HCAL    : 8.12%
  Tracks  : 1.43%
```

---

## Configuration Reference

All hyperparameters are in `configs/srgan.yaml`:

```yaml
data:
  train_samples: 5000     # jets used for training
  val_samples:   500      # jets used for validation
  test_samples:  500      # jets used for test
  normalise: true         # paired normalisation (LR & HR / max(HR))

model:
  n_feat:    64           # feature channels in generator
  n_groups:  6            # number of RRDB blocks
  n_blocks:  3            # dense sub-blocks per RRDB
  growth:    32           # dense connection growth rate
  reduction: 16           # channel attention squeeze ratio
  ema_decay: 0.999        # EMA decay for inference weights

training:
  epochs:        50
  warmup_epochs:  5       # L1-only pre-training
  hold_epochs:   10       # constant LR before cosine decay
  batch_size:    16
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  use_amp: true           # mixed precision training

loss:
  lambda_adv:       0.01
  lambda_energy:    1.0
  lambda_ch_energy: 0.5
  lambda_cf:        0.5
  lambda_freq:      0.1

kfold:
  enabled:      true
  n_folds:      5         # 5-fold stratified cross-validation
  energy_bins:  5         # stratification by HR energy quantile
```

---

## Design Choices & Discussion

### Why ESRGAN over plain SRGAN?
ESRGAN replaces basic residual blocks with **Residual-in-Residual Dense Blocks (RRDB)**, which provide richer feature reuse and stronger gradient flow. With the small dataset size (5000 training jets), maximum information flow per parameter is important.

### Why paired normalisation?
Early experiments (mentor feedback, March 2026) revealed that normalising LR and HR images **independently** breaks the energy scale — the model has no way to learn the correct absolute energy. Paired normalisation (`both / max(HR)`) fixes this, significantly reducing energy conservation error.

### Why per-channel energy loss?
ECAL, HCAL, and Track channels have very different energy scales and distributions. A single global energy loss can mask channel-level failures. With per-channel loss, the model must independently conserve energy in each physical detector subsystem.

### Why stratified K-Fold CV?
With only ~5000 training jets, a single train/val split risks bias from energy distribution imbalance across folds. Stratified 5-fold CV, where folds are balanced by total HR energy quantile, gives more reliable generalisation estimates.

### Why RaGAN instead of vanilla LSGAN?
Standard LSGAN training often leads to discriminator collapse where `D_loss → 0`. The Relativistic average GAN loss (RaGAN) avoids this by making D compare real vs fake *relatively*, providing useful gradients to both networks throughout training.

### Why frequency loss?
GANs tend to over-smooth outputs in pixel space. The spectral (FFT) loss directly penalises missing high-frequency content, which corresponds to sharp shower edges and fine spatial structure that are physically important for jet reconstruction.

---

## References

- Wang et al., *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks*, ECCV Workshop 2018
- Lim et al., *Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)*, CVPR Workshop 2017
- Jolicoeur-Martineau, *The Relativistic Discriminator: a Key Element Missing from Standard GAN*, ICLR 2019
- Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018
- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks (pix2pix / PatchGAN)*, CVPR 2017
