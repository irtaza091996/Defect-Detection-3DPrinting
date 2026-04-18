# Development of AI Solutions for Monitoring of Additive Extrusion Processes Utilizing Optical Coherence Tomography Data

Deep learning-based defect detection in Optical Coherence Tomography (OCT)
B-scans of 3D-printed samples using U-Net and DeepLabv3+ segmentation models.

Developed as part of a Master's Thesis at TU Dresden,
Faculty of Mechanical Science and Engineering.

---

## Overview

Additive manufacturing (FDM/extrusion) processes are prone to defects such as
internal voids and surface artifacts that compromise structural integrity. Manual
inspection of OCT scan data is impractical at scale, so this project develops an
automated AI pipeline that performs **real-time, pixel-level semantic segmentation**
of OCT B-scans to classify each pixel into one of three classes:

| Class | Colour in output | Description |
|---|---|---|
| Background | Black | Everything outside the printed material |
| Edge Artifacts | Green | Surface-layer scattering at the print boundary |
| Internal Defects | Red | Voids inside the printed structure |

The segmentation output is designed to feed a **closed-loop feedback controller**
that can pause or adjust the 3D printer in real time when defects are detected, thus
saving material cost and print time.

<img width="370" height="200" alt="image" src="https://github.com/user-attachments/assets/c399137f-44e0-439b-b787-3c2edb4fdb53" />

---

The figure below illustrates the end-to-end pipeline, from OCT scanning and data
preprocessing to model training, evaluation, and real-time defect feedback.

<img width="905" height="472" alt="image" src="https://github.com/user-attachments/assets/970848bb-bd7b-4772-a96b-c7eecfce2c33" />

---

## Results Summary

### Segmentation Metrics (per class)

| Model | Class | Precision | Recall | Dice | Accuracy |
|---|---|---|---|---|---|
| U-Net *(overall ~96%)* | Background | 1.00 | 1.00 | 1.00 | 0.99 |
| | Edge Artifacts | 0.97 | 0.97 | 0.97 | 0.91 |
| | Internal Defects | 0.92 | 0.90 | 0.91 | 0.91 |
| DeepLabv3+ *(overall ~94%)* | Background | 1.00 | 1.00 | 1.00 | 0.98 |
| | Edge Artifacts | 0.96 | 0.95 | 0.95 | 0.90 |
| | Internal Defects | 0.88 | 0.86 | 0.87 | 0.89 |

### Inference Speed

| Model | Avg. Time / Image | FPS | Convergence |
|---|---|---|---|
| U-Net | 0.0048 s | ~208 | Epoch 16 |
| DeepLabv3+ | 0.0046 s | ~217 | Epoch 29 |

Both models run well above the real-time threshold (≥25 FPS) on a single GPU.
DeepLabv3+ is ~3-4% faster at inference, but U-Net converges faster and achieves
better defect detection accuracy, thus making **U-Net the recommended choice** for
critical AM quality-control environments.

U-Net Training:

<img width="456" height="300" alt="image" src="https://github.com/user-attachments/assets/2fe048d9-0f94-4453-b742-923f7d72e383" />

U-Net Confusion Matrix:

<img width="376" height="300" alt="image" src="https://github.com/user-attachments/assets/1eff7a22-e766-4d13-989a-bf066a0eb3a7" />

U-Net Evaluation Metrics:

<img width="477" height="300" alt="image" src="https://github.com/user-attachments/assets/5ba899b6-a5b0-4801-aad1-3e2aa2c45f18" />

Inference Test on Unseen Data:

<img width="527" height="492" alt="image" src="https://github.com/user-attachments/assets/ecfe4fa8-fd75-4890-b91a-85897a3060e6" />

---

## Repository Structure

```
├── src/
│   ├── config.py            # Central constants (model params, class names, colours)
│   ├── dataset.py           # OCTSegmentationDataset class and load_dataset()
│   └── utils.py             # Shared utilities: metrics, plotting, mask decoding
├── Preprocessing/
│   ├── tiff_processing.py   # Step 1 — extract B-scan frames from 3D OCT TIFF volumes
│   └── preprocessing.py     # Step 2 — convert binary outlier maps to multi-class masks
├── Models/
│   ├── U-Net/
│   │   └── u_net.py         # U-Net training script
│   └── DeepLabv3Plus/
│       └── deeplabv3plus.py # DeepLabV3+ training script
├── inference/
│   └── inference_test.py    # Benchmarking and overlay visualisation
├── requirements.txt
└── README.md
```

---

## Getting Started

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended (models were trained on NVIDIA A100 80 GB via Google Colab; CPU fallback is supported but slow)

### 1. Clone the repository

```bash
git clone https://github.com/irtaza091996/Defect-Detection-3DPrinting.git
cd Defect-Detection-3DPrinting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

> **Note on paths:** All scripts accept a `--data-dir` (or equivalent) argument.
> Edit this to point at your local dataset root before running.

### Step 1 — Extract B-Scans from TIFF Volumes

```bash
python Preprocessing/tiff_processing.py \
    --tiff-dir /path/to/tiff \
    --out-dir  /path/to/output
```

<img width="610" height="150" alt="image" src="https://github.com/user-attachments/assets/6473b027-32f9-4790-b16e-8e76a414a3e3" />

### Step 2 — Generate Multi-Class Segmentation Masks

```bash
python Preprocessing/preprocessing.py --data-dir /path/to/Data
```

### Step 3 — Train a Model

```bash
# U-Net (early-stopped at epoch 16 in the original experiments)
python Models/U-Net/u_net.py --data-dir /path/to/Data --epochs 20

# DeepLabV3+ (early-stopped at epoch 29 in the original experiments)
python Models/DeepLabv3Plus/deeplabv3plus.py --data-dir /path/to/Data --epochs 30
```

All hyperparameters (`--lr`, `--batch-size`, `--patience`, `--num-workers`, `--save-path`) are configurable via CLI flags. Run with `--help` for the full list.

### Step 4 — Run Inference

```bash
python inference/inference_test.py \
    --test-dir Test/ \
    --unet-weights    Models/U-Net/Weights_U-Net.pth \
    --deeplab-weights Models/DeepLabv3Plus/Weights_Deeplabv3+.pth
```

---

## Dataset

9 printed samples were scanned using OCT, yielding 8,131 grayscale B-scans with
corresponding multi-class segmentation masks. 80/20 train/validation split.

**The dataset is not publicly available.** Contact the author for access inquiries.

| Sample | B-Scans | Width × Height (px) | Material |
|---|---|---|---|
| A-09_1_layer | 999 | 260 × 131 | PA12 |
| A-09_2_layer | 999 | 317 × 158 | PA12 |
| A-09_3_layer | 999 | 281 × 177 | PA12 |
| T-16_1_layer | 998 | 295 × 138 | PA12 |
| T-16_2_layer | 999 | 305 × 174 | PA12 |
| T-16_3_layer | 998 | 271 × 220 | PA12 |
| X5Y4_1_layer | 714 | 295 × 140 | Raise3D PLA Red |
| X5Y4_2_layer | 714 | 303 × 150 | Raise3D PLA Red |
| X5Y4_3_layer | 711 | 305 × 187 | Raise3D PLA Red |

Binary outlier maps (used as the basis for ground truth mask generation) were
produced via statistical methods in a prior study at TU Dresden.

---

## Methods

| Component | U-Net | DeepLabv3+ |
|---|---|---|
| Encoder backbone | ResNet-18 (no pre-trained weights) | ResNet-18 (no pre-trained weights) |
| Input | Grayscale OCT B-scans, resized to 256 × 320 px | ← same |
| Output classes | Background, Edge Artifacts, Internal Defects | ← same |
| Loss function | Cross-Entropy Loss | ← same |
| Optimizer | Adam — LR: 1e-3 (tuned via Optuna) | ← same |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 3) | ← same |
| Batch size | 4 | ← same |
| Early stopping | Patience 5 | ← same |
| Actual epochs run | 16 | 29 |
| Metrics | Dice, IoU, Precision, Recall, Accuracy | ← same |
| Training hardware | NVIDIA A100 80 GB (Google Colab) | ← same |

> Hyperparameters (learning rate, optimizer, loss function) were selected using
> **Optuna** Bayesian optimisation. The same ResNet-18 backbone was deliberately
> chosen for both models to enable a fair comparison.

---

## Future Work

As discussed in the thesis, several directions remain open:

- **3D segmentation**: Extend the pipeline from 2D B-scans to full volumetric OCT data for richer spatial context.
- **Additional defect classes**: Incorporate cracks, delamination, and thermal distortions (currently only voids and edge artifacts are labelled).
- **Ground truth quality**: Improve the statistical outlier maps (e.g., via semi-supervised learning) to reduce label noise and misclassification.
- **Closed-loop deployment**: Integrate the segmentation output with a real-time printer feedback controller to trigger corrective actions automatically.

---

## Citation

If you use this work, please cite:

```
Muhammad Irtaza Khan.
"Development of AI Solutions for Monitoring of Additive Extrusion Processes
Utilizing Optical Coherence Tomography Data."
Master's Thesis, TU Dresden, Faculty of Mechanical Science and Engineering, 2025.
```

---

## Author

Muhammad Irtaza Khan

TU Dresden - Faculty of Mechanical Science and Engineering


