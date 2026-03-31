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
- Edge Artifacts (surface)  
- Internal Defects (voids)
<img width="370" height="200" alt="image" src="https://github.com/user-attachments/assets/c399137f-44e0-439b-b787-3c2edb4fdb53" />


---
The figure below illustrates the end-to-end pipeline — from OCT 
scanning and data preprocessing to model training, evaluation, 
and real-time defect feedback.

<img width="905" height="472" alt="image" src="https://github.com/user-attachments/assets/970848bb-bd7b-4772-a96b-c7eecfce2c33" />



---

## Results Summary

| Model       | Accuracy | Defect Precision | Defect Recall |
|-------------|----------|-----------------|---------------|
| U-Net       | ~96%     | 0.92            | 0.90          |
| DeepLabv3+  | ~94%     | 0.88            | 0.86          |


U-Net Training:

<img width="456" height="300" alt="image" src="https://github.com/user-attachments/assets/2fe048d9-0f94-4453-b742-923f7d72e383" />


U-Net Confusion Matrix:

<img width="376" height="300" alt="image" src="https://github.com/user-attachments/assets/1eff7a22-e766-4d13-989a-bf066a0eb3a7" />


U-Net Evaluation Matrix:

<img width="477" height="300" alt="image" src="https://github.com/user-attachments/assets/5ba899b6-a5b0-4801-aad1-3e2aa2c45f18" />


Inference Test on Unseen Data:

<img width="527" height="492" alt="image" src="https://github.com/user-attachments/assets/ecfe4fa8-fd75-4890-b91a-85897a3060e6" />


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



---

## Usage

### Preprocessing
Run `Preprocessing/preprocessing.py` or open the notebook to convert 3D OCT 
volumes into B-scans and generate multi-class segmentation masks.

<img width="610" height="150" alt="image" src="https://github.com/user-attachments/assets/6473b027-32f9-4790-b16e-8e76a414a3e3" />

### Training
Open the respective notebook in `Models/U-Net/` or `Models/DeepLabv3+/` 
and run all cells. Hyperparameters (epochs, LR, batch size) are set at the top.

### Inference
Open the respective notebook in `Inference Test/inference_test.py` 
and run all cells.



---

## Dataset

9 printed samples were scanned using OCT, yielding 8,131 grayscale B-scans 
with corresponding multi-class segmentation masks. 80/20 train/validation split.

The dataset is not publicly available. Contact the author for access inquiries.

Dataset Information:
| Sample       | Number of B-scans | Dimensions |
|-------------|-----|-----------------|
| A-09_1_layer | 999 | 260x131 | 
| A-09_2_layer | 999 | 317x158 |
| A-09_3_layer | 999 | 281x177 | 
| T-16_1_layer | 998 | 295x138 |
| T-16_2_layer | 999 | 305x174 | 
| T-16_3_layer | 998 | 271x220 |
| X5Y4_1_layer | 714 | 295x140 | 
| X5Y4_2_layer | 714 | 303x150 |
| X5Y4_3_layer | 711 | 305x187 | 
---

## Methods

- **Models:** U-Net (ResNet-18 backbone), DeepLabv3+ (ResNet-18 backbone)
- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** Adam (LR: 1e-3)
- **Metrics:** Dice, IoU, Precision, Recall

---

## Author

Muhammad Irtaza Khan  
TU Dresden 
