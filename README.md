# Development of AI Solutions for Monitoring of Additive Extrusion Processes Utilizing Optical Coherence Tomography Data

Deep learning-based defect detection in Optical Coherence Tomography (OCT)
B-scans of 3D-printed samples using U-Net and DeepLabv3+ segmentation models.

Developed as part of a Master's Thesis at TU Dresden,
Faculty of Mechanical Science and Engineering.

---

## Overview

This project develops AI solutions to monitor additive extrusion (FDM)
processes in real time. OCT B-scans of printed samples are segmented using
two deep learning architectures to classify pixels into three classes:
- Background
- Edge Artifacts (surface defects)
- Internal Defects (voids)

<img width="370" height="200" alt="image" src="https://github.com/user-attachments/assets/c399137f-44e0-439b-b787-3c2edb4fdb53" />

---

The figure below illustrates the end-to-end pipeline — from OCT
scanning and data preprocessing to model training, evaluation,
and real-time defect feedback.

<img width="905" height="472" alt="image" src="https://github.com/user-attachments/assets/970848bb-bd7b-4772-a96b-c7eecfce2c33" />

---

## Results Summary

### Segmentation Metrics

| Model      | Class           | Dice | IoU  | Precision | Recall |
|------------|-----------------|------|------|-----------|--------|
| U-Net      | Background      | 0.98 | 0.96 | 0.98      | 0.98   |
| U-Net      | Edge Artifacts  | 0.94 | 0.89 | 0.95      | 0.93   |
| U-Net      | Internal Defects| 0.90 | 0.82 | 0.92      | 0.90   |
| DeepLabv3+ | Background      | 0.96 | 0.93 | 0.96      | 0.96   |
| DeepLabv3+ | Edge Artifacts  | 0.91 | 0.84 | 0.91      | 0.91   |
| DeepLabv3+ | Internal Defects| 0.86 | 0.75 | 0.88      | 0.86   |

### Inference Speed

| Model      | Avg. Time / Image | FPS   |
|------------|-------------------|-------|
| U-Net      | ~0.005 s          | ~200  |
| DeepLabv3+ | ~0.008 s          | ~125  |

Both models run well above the real-time threshold (≥25 FPS) on a single GPU.

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
- CUDA-capable GPU recommended (CPU fallback supported)

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
# U-Net
python Models/U-Net/u_net.py --data-dir /path/to/Data --epochs 20

# DeepLabV3+
python "Models/DeepLabv3Plus/deeplabv3plus.py" --data-dir /path/to/Data --epochs 30
```

All hyperparameters (`--lr`, `--batch-size`, `--patience`, `--save-path`) are configurable via CLI flags. Run with `--help` for the full list.

### Step 4 — Run Inference

```bash
python inference/inference_test.py" \
    --test-dir Test/ \
    --unet-weights    Models/U-Net/Weights_U-Net.pth \
    --deeplab-weights "Models/DeepLabv3Plus/Weights_Deeplabv3+.pth"
```

---

## Dataset

9 printed samples were scanned using OCT, yielding 8,131 grayscale B-scans
with corresponding multi-class segmentation masks. 80/20 train/validation split.

The dataset is not publicly available. Contact the author for access inquiries.

| Sample        | B-Scans | Width × Height (px) |
|---------------|---------|----------------------|
| A-09_1_layer  | 999     | 260 × 131            |
| A-09_2_layer  | 999     | 317 × 158            |
| A-09_3_layer  | 999     | 281 × 177            |
| T-16_1_layer  | 998     | 295 × 138            |
| T-16_2_layer  | 999     | 305 × 174            |
| T-16_3_layer  | 998     | 271 × 220            |
| X5Y4_1_layer  | 714     | 295 × 140            |
| X5Y4_2_layer  | 714     | 303 × 150            |
| X5Y4_3_layer  | 711     | 305 × 187            |

---

## Methods

| Component       | Detail                                  |
|-----------------|-----------------------------------------|
| Models          | U-Net, DeepLabV3+ (ResNet-18 backbone)  |
| Input           | Grayscale OCT B-scans, resized 256×320  |
| Classes         | Background, Edge Artifacts, Defects     |
| Loss Function   | Cross-Entropy Loss                      |
| Optimizer       | Adam (LR: 1e-3)                         |
| LR Scheduler    | ReduceLROnPlateau (factor 0.5, patience 3) |
| Early Stopping  | Patience 5 epochs                       |
| Metrics         | Dice, IoU, Precision, Recall, F1        |

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
TU Dresden
