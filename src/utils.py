import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)

CLASS_NAMES = ['Background', 'Edge Artifacts', 'Defects']
CLASS_COLORS = {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 0]}


def decode_segmentation_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a class-index mask to an RGB colour overlay."""
    overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        overlay[mask == cls] = color
    return overlay


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> dict:
    """Return per-class Dice, IoU, Precision, Recall, and F1."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    metrics = {}
    for c in range(num_classes):
        yt = (y_true_flat == c).astype(int)
        yp = (y_pred_flat == c).astype(int)
        dice = (2 * np.sum(yt * yp)) / (np.sum(yt) + np.sum(yp) + 1e-6)
        metrics[CLASS_NAMES[c]] = {
            'Dice':      dice,
            'IoU':       jaccard_score(yt, yp, zero_division=0),
            'Precision': precision_score(yt, yp, zero_division=0),
            'Recall':    recall_score(yt, yp, zero_division=0),
            'F1':        f1_score(yt, yp, zero_division=0),
        }
    return metrics


def print_metrics(metrics: dict) -> None:
    for cls, m in metrics.items():
        print(f'\n{cls.upper()}')
        for k, v in m.items():
            print(f'  {k}: {v:.4f}')


def plot_loss_curves(train_losses: list, val_losses: list) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[0, 1, 2], normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap='Blues', values_format='.2f'
    )
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_evaluation_metrics(metrics: dict) -> None:
    classes = list(metrics.keys())
    metric_keys = ['Dice', 'IoU', 'Precision', 'Recall']
    x = np.arange(len(classes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, key in enumerate(metric_keys):
        vals = [metrics[c][key] for c in classes]
        rects = ax.bar(x + (i - 1.5) * width, vals, width, label=key)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f'{h:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom',
            )

    ax.set_ylabel('Score')
    ax.set_title('Segmentation Evaluation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_overlay(image_np: np.ndarray, pred_mask: np.ndarray, title: str = 'Predicted Mask Overlay') -> None:
    overlay_rgb = decode_segmentation_mask(pred_mask)
    image_rgb = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(image_rgb, 0.6, overlay_rgb, 0.4, 0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('B-Scan')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(blended)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
