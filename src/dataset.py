import os
from glob import glob

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

SAMPLES = [
    'A-09_1_layer', 'A-09_2_layer', 'A-09_3_layer',
    'T-16_1_layer', 'T-16_2_layer', 'T-16_3_layer',
    'X5Y4_1_layer', 'X5Y4_2_layer', 'X5Y4_3_layer',
]

TARGET_H, TARGET_W = 256, 320


class OCTSegmentationDataset(Dataset):
    """PyTorch Dataset for grayscale OCT B-scans with multi-class segmentation masks."""

    def __init__(self, image_paths: list[str], mask_paths: list[str]):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
        mask = np.loadtxt(self.mask_paths[idx], dtype=np.uint8)

        image = cv2.resize(image, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

        return (
            torch.tensor(image, dtype=torch.float32).unsqueeze(0),  # [1, H, W]
            torch.tensor(mask, dtype=torch.long),
        )


def load_dataset(
    data_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list, list, list, list]:
    """Collect all image/mask paths from data_dir and return an 80/20 train/val split."""
    all_image_paths, all_mask_paths = [], []
    for sample in SAMPLES:
        img_dir = os.path.join(data_dir, sample, f'Timing_{sample}')
        mask_dir = os.path.join(data_dir, sample, 'Multimasks_corrected_txt')
        all_image_paths.extend(sorted(glob(os.path.join(img_dir, '*.png'))))
        all_mask_paths.extend(sorted(glob(os.path.join(mask_dir, '*.txt'))))

    return train_test_split(
        all_image_paths, all_mask_paths,
        test_size=test_size,
        random_state=random_state,
    )
