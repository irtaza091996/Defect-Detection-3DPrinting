"""Run inference benchmarking and overlay visualisation for U-Net and DeepLabV3+.

Usage:
    python "Inference Test/inference_test.py" --test-dir Test/ \
        --unet-weights Models/U-Net/Weights_U-Net.pth \
        --deeplab-weights "Models/DeepLabv3+/Weights_Deeplabv3+.pth"
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import segmentation_models_pytorch as smp

# Allow imports from the repo root (src/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import decode_segmentation_mask

TARGET_H, TARGET_W = 256, 320


def load_models(unet_weights: str, deeplab_weights: str, device: torch.device):
    unet = smp.Unet('resnet18', encoder_weights=None, in_channels=1, classes=3).to(device)
    unet.load_state_dict(torch.load(unet_weights, map_location=device))
    unet.eval()

    deeplab = smp.DeepLabV3Plus('resnet18', encoder_weights=None, in_channels=1, classes=3).to(device)
    deeplab.load_state_dict(torch.load(deeplab_weights, map_location=device))
    deeplab.eval()

    return unet, deeplab


def preprocess_image(path: str, device: torch.device) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
    img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    return torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)


def warmup(models: list, dummy: torch.Tensor, iterations: int = 10):
    with torch.no_grad():
        for _ in range(iterations):
            for model in models:
                model(dummy)


def benchmark(models: dict, test_images: list, device: torch.device) -> dict:
    times = {name: [] for name in models}
    with torch.no_grad():
        for path in tqdm(test_images, desc='Benchmarking'):
            img = preprocess_image(path, device)
            for name, model in models.items():
                start = time.perf_counter()
                model(img)
                times[name].append(time.perf_counter() - start)
    return times


def print_benchmark_results(times: dict):
    print()
    for name, t in times.items():
        avg = np.mean(t)
        print(f'{name:12s}  avg: {avg:.6f}s/image  |  FPS: {1/avg:.2f}')


def plot_benchmark(times: dict):
    avg_times = {name: np.mean(t) for name, t in times.items()}
    fps_values = {name: 1.0 / t for name, t in avg_times.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(avg_times.keys(), avg_times.values(), color=['skyblue', 'orange'])
    ax1.set_ylabel('Time per Image (s)')
    ax1.set_title('Inference Time per Image')
    for i, (name, val) in enumerate(avg_times.items()):
        ax1.text(i, val + 0.0005, f'{val:.4f}s', ha='center')

    ax2.bar(fps_values.keys(), fps_values.values(), color=['skyblue', 'orange'])
    ax2.set_ylabel('Frames Per Second (FPS)')
    ax2.set_title('Inference Speed (FPS)')
    for i, (name, val) in enumerate(fps_values.items()):
        ax2.text(i, val + 5, f'{val:.1f} FPS', ha='center')

    plt.tight_layout()
    plt.show()


def visualise_prediction(model, sample_path: str, device: torch.device):
    original = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = original.shape

    img_tensor = preprocess_image(sample_path, device)
    with torch.no_grad():
        pred_mask = torch.argmax(model(img_tensor).squeeze(), dim=0).cpu().numpy()

    pred_mask_resized = cv2.resize(
        pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )
    overlay = decode_segmentation_mask(pred_mask_resized)
    image_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)

    print(f'Original image size: {orig_w}×{orig_h}')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original B-Scan')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(blended)
    plt.title('Overlay: Prediction on Original Size')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    unet, deeplab = load_models(args.unet_weights, args.deeplab_weights, device)
    models = {'U-Net': unet, 'DeepLabV3+': deeplab}

    test_images = sorted(glob(str(Path(args.test_dir) / '*.png')))
    if not test_images:
        raise FileNotFoundError(f'No PNG images found in {args.test_dir}')

    warmup(list(models.values()), preprocess_image(test_images[0], device))

    times = benchmark(models, test_images, device)
    print_benchmark_results(times)
    plot_benchmark(times)
    visualise_prediction(deeplab, test_images[1], device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference benchmark for U-Net and DeepLabV3+')
    parser.add_argument('--test-dir', default='Test', help='Directory containing test PNG images')
    parser.add_argument('--unet-weights', default='Models/U-Net/Weights_U-Net.pth')
    parser.add_argument('--deeplab-weights', default='Models/DeepLabv3+/Weights_Deeplabv3+.pth')
    main(parser.parse_args())
