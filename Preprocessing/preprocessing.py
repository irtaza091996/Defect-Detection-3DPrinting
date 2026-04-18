"""Convert binary outlier maps into multi-class segmentation masks (ground truth).

Each raw binary mask (.txt) is processed per sample using morphological operations
(Opening or Closing), spline gap-filling, and neighbourhood-label voting to produce
three-class masks:
  0 = Background
  1 = Internal Defects
  2 = Edge Artifacts

Usage:
    python Preprocessing/preprocessing.py --data-dir Data/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import make_interp_spline
from scipy.ndimage import binary_dilation, binary_erosion

# (sample_name, morphological_operation, dilation_struct, erosion_struct)
SAMPLE_CONFIGS = [
    ('A-09_1_layer', 'Opening', (1, 10), (2, 20)),
    ('A-09_2_layer', 'Closing', (1, 40), (2, 40)),
    ('A-09_3_layer', 'Closing', (1, 40), (2, 40)),
    ('T-16_1_layer', 'Opening', (1, 10), (2, 20)),
    ('T-16_2_layer', 'Opening', (1, 10), (2, 20)),
    ('T-16_3_layer', 'Closing', (1, 40), (2, 40)),
    ('X5Y4_1_layer', 'Opening', (1, 10), (2, 20)),
    ('X5Y4_2_layer', 'Opening', (1, 10), (2, 20)),
    ('X5Y4_3_layer', 'Opening', (1,  5), (2,  5)),
]


def assign_label(neighborhood: np.ndarray) -> int:
    """Assign a label to an unlabelled pixel via majority vote of its 3×3 neighbours."""
    center = neighborhood[4]
    if center != 0:
        return center
    window = np.delete(neighborhood, 4)
    return 1 if np.sum(window == 1) >= np.sum(window == 2) else 2


def apply_morphology(mask: np.ndarray, operation: str, param1: tuple, param2: tuple) -> np.ndarray:
    if operation == 'Opening':
        return binary_erosion(binary_dilation(mask, structure=np.ones(param1)), structure=np.ones(param2))
    return binary_erosion(binary_dilation(mask, structure=np.ones(param2)), structure=np.ones(param1))


def fill_column_gaps(closed: np.ndarray) -> np.ndarray:
    """Use linear spline interpolation to fill columns with no active pixels."""
    for col in np.where(np.sum(closed, axis=0) == 0)[0]:
        left, right = col - 1, col + 1
        while left >= 0 and np.sum(closed[:, left]) == 0:
            left -= 1
        while right < closed.shape[1] and np.sum(closed[:, right]) == 0:
            right += 1
        if left >= 0 and right < closed.shape[1]:
            y_l = np.min(np.where(closed[:, left] == 1))
            y_r = np.min(np.where(closed[:, right] == 1))
            x_new = np.linspace(left, right, num=(right - left + 1))
            y_smooth = make_interp_spline([left, right], [y_l, y_r], k=1)(x_new)
            closed[np.round(y_smooth).astype(int), np.round(x_new).astype(int)] = 1
    return closed


def build_edge_mask(red_mask: np.ndarray, morph_op: str, param1: tuple, param2: tuple) -> np.ndarray:
    """Derive the edge-artifact (blue) mask from the raw binary mask."""
    quarter_height = 3 * red_mask.shape[0] // 4
    upper = red_mask[:quarter_height, :].copy()

    closed = apply_morphology(upper, morph_op, param1, param2)
    closed = fill_column_gaps(closed)

    blue_mask = red_mask.copy()
    blue_mask[:quarter_height, :] = closed

    # Propagate edge label downward per column
    for col in range(blue_mask.shape[1]):
        for row in range(blue_mask.shape[0]):
            if blue_mask[row, col] == 1:
                blue_mask[row:, col] = 1
                break

    # Remove thin noise strips at the top of each column
    for col in range(blue_mask.shape[1]):
        ones = np.where(blue_mask[:, col] == 1)[0]
        if len(ones) >= 25:
            blue_mask[ones[:25], col] = 0

    return blue_mask


def neighbourhood_voting_correction(binary_mask: np.ndarray, multi_mask: np.ndarray) -> np.ndarray:
    """Fix pixels that are foreground in binary_mask but unassigned in multi_mask."""
    corrected = multi_mask.copy()
    padded = np.pad(multi_mask, pad_width=1, mode='constant', constant_values=0)
    for i, j in np.argwhere((binary_mask == 1) & (multi_mask == 0)):
        corrected[i, j] = assign_label(padded[i:i + 3, j:j + 3].flatten())
    return corrected


def process_sample(sample_name: str, morph_op: str, param1: tuple, param2: tuple,
                   base_dir: str, out_txt_dir: str, out_img_dir: str):
    txt_dir = os.path.join(base_dir, sample_name, 'txt_binary')
    mask_dir = os.path.join(base_dir, sample_name, 'Masks')

    for filename in sorted(os.listdir(txt_dir)):
        if not filename.endswith('.txt'):
            continue

        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(txt_dir, filename)
        img_path = os.path.join(mask_dir, base_name + '.png')

        if not os.path.exists(img_path):
            print(f'  Skipping {filename} — image not found')
            continue

        red_mask = np.loadtxt(txt_path, dtype=np.uint8)
        blue_mask = build_edge_mask(red_mask, morph_op, param1, param2)

        multi_mask = np.zeros_like(red_mask)
        multi_mask[(red_mask == 1) & (blue_mask == 1)] = 2
        multi_mask[(red_mask == 1) & (blue_mask == 0)] = 1

        corrected = neighbourhood_voting_correction(red_mask, multi_mask)

        # Geometric correction: shift one pixel to the right
        corrected = np.hstack([
            np.zeros((corrected.shape[0], 1), dtype=np.uint8),
            corrected[:, :-1],
        ])

        np.savetxt(os.path.join(out_txt_dir, base_name + '.txt'), corrected, fmt='%d')

        colored = np.zeros((*corrected.shape, 3), dtype=np.uint8)
        colored[corrected == 1] = [0, 255, 0]
        colored[corrected == 2] = [255, 0, 0]
        colored = np.hstack([
            np.zeros((colored.shape[0], 1, 3), dtype=np.uint8),
            colored[:, :-1, :],
        ])
        Image.fromarray(colored).save(os.path.join(out_img_dir, base_name + '.png'))

    print(f'  Done: {sample_name}')


def main(args):
    out_txt_dir = os.path.join(args.data_dir, 'Multimasks_corrected_txt')
    out_img_dir = os.path.join(args.data_dir, 'Multimasks_corrected')
    os.makedirs(out_txt_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)

    for sample_name, morph_op, param1, param2 in SAMPLE_CONFIGS:
        print(f'Processing {sample_name} ({morph_op})...')
        process_sample(sample_name, morph_op, param1, param2, args.data_dir, out_txt_dir, out_img_dir)

    print('\nAll samples processed successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate multi-class segmentation masks from binary OCT outlier maps')
    parser.add_argument('--data-dir', default='Data/', help='Path to dataset root directory')
    main(parser.parse_args())
