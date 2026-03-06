# -*- coding: utf-8 -*-
"""U_Net.ipynb

# **U-Net Segmentation Model**

"""

"""**Importing Necessary Libraries**"""

#Installing segmentation_models_pytorch and dependencies
!pip install -q segmentation-models-pytorch
!pip install -q albumentations==1.3.0
!pip install -q --upgrade scikit-image

import os
from glob import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

"""**B-scan Processing**"""

#Defining dataset
class OCTSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
        mask = np.loadtxt(self.mask_paths[idx], dtype=np.uint8)

        #Resizing both image and mask to a fixed size (256x320)
        target_h, target_w = 256, 320
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        #Converting to tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  #Shape: [1, H, W]
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

#Collecting all paths
base_path = "/Data/"
samples = [
    'A-09_1_layer', 'A-09_2_layer', 'A-09_3_layer',
    'T-16_1_layer', 'T-16_2_layer', 'T-16_3_layer',
    'X5Y4_1_layer', 'X5Y4_2_layer', 'X5Y4_3_layer'
]

all_image_paths, all_mask_paths = [], []
for sample in samples:
    img_dir = os.path.join(base_path, sample, f'Timing_{sample}')
    mask_dir = os.path.join(base_path, sample, 'Multimasks_corrected_txt')
    all_image_paths.extend(sorted(glob(os.path.join(img_dir, '*.png'))))
    all_mask_paths.extend(sorted(glob(os.path.join(mask_dir, '*.txt'))))

#Data Split
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    all_image_paths, all_mask_paths, test_size=0.2, random_state=42
)

train_dataset = OCTSegmentationDataset(train_imgs, train_masks)
val_dataset = OCTSegmentationDataset(val_imgs, val_masks)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

"""**Defining Model Parameters**"""

import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=1,
    classes=3
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

"""**Model Training**"""

from tqdm import tqdm

#Training loop
num_epochs = 20
patience = 5
early_stop_counter = 0
best_val_loss = float("inf")
save_path = "/Weights_U-Net.pth"
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    print(f"\n Epoch {epoch+1}/{num_epochs}")

    #Training phase
    model.train()
    train_loss = 0.0
    pbar_train = tqdm(train_loader, desc="Training", leave=False)
    for images, masks in pbar_train:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        pbar_train.set_postfix(batch_loss=loss.item())

    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    #Validation phase
    model.eval()
    val_loss = 0.0
    pbar_val = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, masks in pbar_val:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            pbar_val.set_postfix(batch_loss=loss.item())

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    #Epoch summary
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    scheduler.step(avg_val_loss)

    #Defining Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print("Best model saved.")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"Early Stop Counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

"""**Loss Plots**"""

plt.figure(figsize=(8, 5))
max_epoch_to_display = 16
plt.plot(range(max_epoch_to_display + 1), train_losses[:max_epoch_to_display + 1], label='Training Loss')
plt.plot(range(max_epoch_to_display + 1), val_losses[:max_epoch_to_display + 1], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.xlim(0, max_epoch_to_display)
plt.tight_layout()
plt.show()

"""**Overlay**"""

import random
#Picking a random sample
idx = random.randint(0, len(val_dataset)-1)
image, true_mask = val_dataset[idx]

#Preparing input
input_tensor = image.unsqueeze(0).to(device)

#Inference
with torch.no_grad():
    output = model(input_tensor)
    pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

#Converting to numpy
image_np = image.squeeze(0).cpu().numpy()
true_mask_np = true_mask.cpu().numpy()

#Plotting
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.imshow(image_np, cmap='gray')
plt.title("Input B-scan")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(true_mask_np, cmap='jet', vmin=0, vmax=2)
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=2)
plt.title("Predicted Mask")
plt.axis("off")
plt.tight_layout()
plt.show()

def decode_segmentation_mask(mask, num_classes=3):
    colors = {
        0: [0, 0, 0],
        1: [0, 255, 0],
        2: [255, 0, 0],
    }
    h, w = mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        overlay[mask == c] = colors[c]
    return overlay

#Generating overlay
overlay_rgb = decode_segmentation_mask(pred_mask)
image_rgb = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
blended = cv2.addWeighted(image_rgb, 0.6, overlay_rgb, 0.4, 0)

#Displaying overlay
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("B-scan")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(blended)
plt.title("Predicted Mask Overlay")
plt.axis("off")
plt.tight_layout()
plt.show()

"""**Generating Segmentation Metrics**"""

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def compute_metrics(y_true, y_pred, num_classes=3):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    metrics = {}
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        y_pred_c = (y_pred == c).astype(int)
        dice = (2 * np.sum(y_true_c * y_pred_c)) / (np.sum(y_true_c) + np.sum(y_pred_c) + 1e-6)
        iou = jaccard_score(y_true_c, y_pred_c, zero_division=0)
        precision = precision_score(y_true_c, y_pred_c, zero_division=0)
        recall = recall_score(y_true_c, y_pred_c, zero_division=0)
        f1 = f1_score(y_true_c, y_pred_c, zero_division=0)
        metrics[f'class_{c}'] = {'Dice': dice, 'IoU': iou, 'Precision': precision, 'Recall': recall, 'F1-score': f1}
    return metrics

#Accumulating predictions and ground truths
all_preds, all_trues = [], []
with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        trues = masks.numpy()
        all_preds.append(preds)
        all_trues.append(trues)

all_preds = np.concatenate(all_preds, axis=0)
all_trues = np.concatenate(all_trues, axis=0)

#Computing
metrics = compute_metrics(all_trues, all_preds)
for cls, m in metrics.items():
    print(f"\n {cls.upper()}")
    for k, v in m.items():
        print(f"   {k}: {v:.4f}")

"""**Confusion Matrix**"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred_flat = all_preds.flatten()
y_true_flat = all_trues.flatten()

cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1, 2], normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Background", "Edge Artifacts", "Defects"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', values_format=".2f")
plt.title("Confusion Matrix")
plt.show()

"""**Segmentation Evaluation Metrics**"""

classes = ['Background', 'Edge Artifacts', 'Defects']
dice_vals = [metrics[f'class_{i}']['Dice'] for i in range(3)]
iou_vals = [metrics[f'class_{i}']['IoU'] for i in range(3)]
precision_vals = [metrics[f'class_{i}']['Precision'] for i in range(3)]
recall_vals = [metrics[f'class_{i}']['Recall'] for i in range(3)]

x = np.arange(len(classes))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - 1.5*width, dice_vals, width, label='Dice')
rects2 = ax.bar(x - 0.5*width, iou_vals, width, label='IoU')
rects3 = ax.bar(x + 0.5*width, precision_vals, width, label='Precision')
rects4 = ax.bar(x + 1.5*width, recall_vals, width, label='Recall')

ax.set_ylabel('Scores')
ax.set_title('Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

def annotate_bars(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

for rect_set in [rects1, rects2, rects3, rects4]:
    annotate_bars(rect_set)

plt.tight_layout()
plt.show()