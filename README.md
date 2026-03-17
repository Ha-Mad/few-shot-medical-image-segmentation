# Few-Shot Medical Image Segmentation
### EMS741 Coursework — Meta-Learning Methods Comparison

A comprehensive Jupyter notebook comparing **meta-learning algorithms** for few-shot medical image segmentation using a U-Net backbone. The project evaluates classic and novel methods across 1-shot, 3-shot, and 5-shot settings, including ablation studies on architectural and training improvements.

---

## Methods Compared

| # | Method | Description |
|---|--------|-------------|
| 1 | **Reptile** | First-order meta-learning via repeated SGD across tasks |
| 2 | **FOMAML** | First-Order MAML — gradient-based meta-learning without second-order terms |
| 3 | **Full MAML** | Model-Agnostic Meta-Learning with second-order gradients |
| 4 | **Meta-SGD** | MAML with learned per-parameter learning rates |
| 5 | **iMAML** | Implicit MAML using proximal regularisation |
| 6 | **Task-Weighted Reptile** | Reptile with importance weighting — harder tasks receive larger updates |
| 7 | **Baseline** | Standard U-Net trained from scratch (no meta-learning) |

---

## Improvements & Ablations (FOMAML variants)

| Variant | Change |
|---------|--------|
| FOMAML + Focal Loss | Replaces BCE+Dice with Focal+Dice to handle class imbalance |
| FOMAML + Augmentation | Random flips, rotations, brightness jitter during meta-training |
| FOMAML + Attention U-Net | Swaps standard U-Net for Attention U-Net backbone |
| FOMAML + Focal + Aug | Focal Loss + Augmentation combined |
| FOMAML + Focal + Attention | Focal Loss + Attention U-Net combined |
| FOMAML + Aug + Attention | Augmentation + Attention U-Net combined |
| FOMAML + All | Focal Loss + Augmentation + Attention U-Net (full combination) |

---

## Architecture

**Backbone:** Standard U-Net / Attention U-Net
**Input:** 128×128 grayscale medical images
**Output:** Binary segmentation mask

**Loss Functions:**
- BCE + Dice Loss (default)
- Focal + Dice Loss (improved variants)

---

## Dataset

Images are downloaded automatically from Zenodo at runtime:

```
https://zenodo.org/records/18745413/files/ems741_cw_data.zip
```

The dataset contains labelled medical image tasks used for N-shot evaluation (N = 1, 3, 5).

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision matplotlib numpy Pillow tqdm
```

> Dependencies are also installed automatically in the first notebook cell.

### Run

1. Open `ems741_v4.ipynb` in Jupyter or Google Colab.
2. **Run Section 0 first** — installs packages and downloads data.
3. Run each section independently (Sections 1–15) to train/evaluate a method.
4. Run **Section 16** to generate full comparison plots and save results.

---

## Evaluation

Each method is evaluated on:
- **Dice Score** across 1-shot, 3-shot, and 5-shot settings
- **Full results table** (all methods × all shot counts)
- **Training curves** (validation loss over outer steps)
- **Bar chart comparison** across methods
- **Ablation study** — FOMAML improvement contributions
- **Per-task analysis** — Dice score per anatomical task
- **Qualitative visualisation** — predicted masks vs. ground truth

---

## Configuration

Key hyperparameters (Section 0, cell 5):

```python
IMAGE_SIZE     = (128, 128)
OUTER_STEPS    = 5000
INNER_STEPS    = 16
INNER_LR       = 1e-3
OUTER_LR       = 1e-3
N_SHOT_LIST    = [1, 3, 5]
ADAPT_STEPS    = 100
```

---

## Project Structure

```
few-shot-medical-image-segmentation/
├── ems741_v4.ipynb     # Main notebook (all methods, training, evaluation)
└── README.md
```

Trained model checkpoints (`.pth` files) and results (`.csv`, plots) are saved locally when cells are executed.

---

## License

This project is submitted as coursework for **EMS741** and is intended for academic use.
