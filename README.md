# OCT Defect Segmentation for Additive Manufacturing

Deep learning-based defect detection in Optical Coherence Tomography (OCT) 
B-scans of 3D-printed samples using U-Net and DeepLabv3+ segmentation models.

Developed as part of a Master's Thesis at TU Dresden,
Faculty of Mechanical Science and Engineering (May 2025).

---

## Overview

This project develops AI solutions to monitor additive extrusion (FDM) 
processes in real time. OCT B-scans of printed samples are segmented using 
two deep learning architectures to classify pixels into three classes:
- Background
- Edge Artifacts (surface)  
- Internal Defects (voids)

---

## Results Summary

| Model       | Accuracy | Defect Precision | Defect Recall |
|-------------|----------|-----------------|---------------|
| U-Net       | ~96%     | 0.92            | 0.90          |
| DeepLabv3+  | ~94%     | 0.88            | 0.86          |

U-Net converges in 16 epochs vs 29 for DeepLabv3+.  
DeepLabv3+ is ~3-4% faster at inference (215.8 FPS vs 209.1 FPS).

---

## Repository Structure
```
├── Preprocessing/       # Data preparation and mask generation scripts
├── Models/
│   ├── U-Net/           # U-Net architecture, training notebook, weights
│   └── DeepLabv3+/      # DeepLabv3+ architecture, training notebook, weights
├── Inference Test/      # Scripts for running inference and benchmarking
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Defect-Detection-3DPrinting.git
cd Defect-Detection-3DPrinting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download model weights
Download pretrained weights and place them in the appropriate model folder:
- `Models/U-Net/Weights_U-Net.pth`
- `Models/DeepLabv3+/Weights_Deeplabv3+.pth`

[Download from Google Drive](#) ← add your link here

---

## Usage

### Preprocessing
Run `Preprocessing/preprocessing.py` or open the notebook to convert 3D OCT 
volumes into B-scans and generate multi-class segmentation masks.

### Training
Open the respective notebook in `Models/U-Net/` or `Models/DeepLabv3+/` 
and run all cells. Hyperparameters (epochs, LR, batch size) are set at the top.

### Inference
```bash
python "Inference Test/inference_test.py"
```
Or open `inference_test.ipynb` for an interactive walkthrough with visualizations.

---

## Dataset

9 printed samples were scanned using OCT, yielding 8,131 grayscale B-scans 
with corresponding multi-class segmentation masks. 80/20 train/validation split.

The dataset is not publicly available. Contact the author for access inquiries.

---

## Methods

- **Models:** U-Net (ResNet-18 backbone), DeepLabv3+ (ResNet-18 backbone)
- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** Adam (LR: 1e-3)
- **Metrics:** Dice, IoU, Precision, Recall

---

## Author

Muhammad Irtaza Khan  
TU Dresden — Faculty of Computer Science  
