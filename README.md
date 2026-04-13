# CVDL Phase Image Denoising

A deep-learning pipeline for denoising noisy phase images, built with PyTorch and served via a Streamlit web app.

---

## Overview

This project trains a custom multi-stage CNN to remove noise from grayscale phase images at varying SNR levels (−5 dB to +25 dB). The trained model is then exposed through an interactive Streamlit UI where users can upload noisy images and download clean results.

---

## Model Architecture

The `AdvancedDenoisingModel` is a 4-stage residual network (~589 K parameters):

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `MultiBranchModule` | Parallel standard, dilated (rates 1/2/4), and depthwise convolutions fused with a 1×1 conv |
| 2 | `MBConv` | Mobile inverted bottleneck (expand ×4, no SE) |
| 3 | `ReducedUNet` | 3-level encoder-decoder with skip connections and a CBAM attention bottleneck |
| 4 | `FusedMBConv` + `MultiBranchModule` | Refinement stage; residual added back to input |

The network predicts a residual and adds it to the noisy input (`output = x + f(x)`).

---

## Results

### Test Set (mixed SNR)

| Metric | Value |
|--------|-------|
| MSE | 0.00125 |
| PSNR | **29.01 dB** |
| SSIM | **0.946** |
| Parameters | 588,631 |

### SNR-Conditioned Performance

| Input SNR (dB) | PSNR (dB) | SSIM |
|---------------|-----------|------|
| −5 | 21.72 | 0.862 |
| 0 | 25.62 | 0.907 |
| 5 | 30.01 | 0.934 |
| 10 | 33.92 | 0.953 |
| 15 | 37.04 | 0.971 |
| 20 | 40.24 | 0.984 |
| 25 | 43.24 | 0.992 |

### Training Curves

Training ran for **70 epochs** with a cosine-annealing LR schedule (peak 1 × 10⁻³ → floor 1 × 10⁻⁶) and a **Charbonnier + SSIM** composite loss.

- Final train loss: ~0.0276 | val loss: ~0.0281
- Final train PSNR: ~29.09 dB | val PSNR: ~28.99 dB
- Final train SSIM: ~0.947 | val SSIM: ~0.945
- Best checkpoint saved from **epoch 62**

Training curve plots are saved in `models/training_curves.png` and `models/predictions.png`.

---

## Project Structure

```
cvdl/
├── streamlit_app.py          # Streamlit inference app
├── data/
│   ├── Training/             # Training images
│   └── Validation/           # Validation images
├── models/
│   ├── best_model.pth        # Best checkpoint (epoch 62)
│   ├── final_model.pth       # Final checkpoint used by the app
│   ├── history.json          # Per-epoch loss / PSNR / SSIM / LR
│   ├── test_results.json     # Test-set metrics
│   ├── snr_analysis.json     # Per-SNR metrics
│   ├── training_curves.png   # Loss & PSNR training plots
│   ├── predictions.png       # Visual prediction samples
│   └── snr_analysis.png      # SNR vs metric plots
└── rbc-main/
    └── cvdlcode.ipynb        # Full training & analysis notebook
```

---

## Requirements

- Python 3.11+
- PyTorch (CPU or CUDA)
- Streamlit
- NumPy
- Pillow
- torchmetrics (for training notebook)

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2. Install dependencies
pip install -U pip
pip install streamlit torch torchvision numpy pillow torchmetrics
```

---

## Run the App

```bash
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (default: http://localhost:8501).

**App features:**
- Upload one or more PNG / JPG / JPEG images
- Toggle RGB channel-wise denoising (each channel processed independently)
- Side-by-side noisy vs. denoised preview
- Download denoised images as PNG

> The app expects `models/final_model.pth` to exist. Replace it with `best_model.pth` or your own weights as needed.

---

## Training

Open and run `rbc-main/cvdlcode.ipynb` to reproduce training from scratch.

Key hyperparameters:

| Setting | Value |
|---------|-------|
| Image size | 256 × 256 |
| Batch size | 32 |
| Epochs | 70 |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| LR schedule | Cosine annealing (1×10⁻³ → 1×10⁻⁶) |
| Loss | Charbonnier + SSIM |
| SNR range (train) | −5 dB to +25 dB (random per sample) |
| SNR range (val/test) | −5 dB to +25 dB (fixed at load time) |

---

## Dataset

The dataset consists of **36,864 grayscale phase images** (JPG/JPEG/PNG), split automatically by the notebook:

| Split | Images | Ratio |
|-------|--------|-------|
| Train | 25,804 | 70% |
| Val | 5,530 | 15% |
| Test | 5,530 | 15% |

Place your images under `data/Training/` and `data/Validation/`. The notebook scans both folders, merges them, and re-splits them 70/15/15 automatically.

Gaussian noise is added on-the-fly at a random SNR (−5 dB to +25 dB) for training samples, and at a fixed SNR for validation/test samples to ensure reproducible evaluation.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
