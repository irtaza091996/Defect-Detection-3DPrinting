"""Train a U-Net model on OCT B-scan segmentation data.

Usage:
    python Models/U-Net/u_net.py --data-dir /path/to/Data
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

# Allow imports from the repo root (src/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset import OCTSegmentationDataset, load_dataset
from src.utils import (
    compute_metrics,
    plot_confusion_matrix,
    plot_evaluation_metrics,
    plot_loss_curves,
    plot_overlay,
    print_metrics,
)


def build_model(device: torch.device) -> nn.Module:
    return smp.Unet(
        encoder_name='resnet18',
        encoder_weights=None,
        in_channels=1,
        classes=3,
    ).to(device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one train or validation epoch. Pass optimizer=None for eval mode."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    desc = 'Training' if training else 'Validating'
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for images, masks in tqdm(loader, desc=desc, leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, val_imgs, val_masks, args, device):
    """Compute and display segmentation metrics on the full validation set."""
    val_dataset = OCTSegmentationDataset(val_imgs, val_masks)
    loader = DataLoader(val_dataset, batch_size=args.batch_size)
    all_preds, all_trues = [], []

    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            preds = torch.argmax(model(images.to(device)), dim=1).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(masks.numpy())

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)

    metrics = compute_metrics(all_trues, all_preds)
    print_metrics(metrics)
    plot_confusion_matrix(all_trues, all_preds)
    plot_evaluation_metrics(metrics)

    # Visualise a random validation sample
    idx = random.randint(0, len(val_dataset) - 1)
    image, _ = val_dataset[idx]
    with torch.no_grad():
        pred = torch.argmax(
            model(image.unsqueeze(0).to(device)).squeeze(), dim=0
        ).cpu().numpy()
    plot_overlay(image.squeeze(0).numpy(), pred)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_imgs, val_imgs, train_masks, val_masks = load_dataset(args.data_dir)

    train_loader = DataLoader(
        OCTSegmentationDataset(train_imgs, train_masks),
        batch_size=args.batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        OCTSegmentationDataset(val_imgs, val_masks),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
    )

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        avg_train = run_epoch(model, train_loader, criterion, device, optimizer)
        avg_val = run_epoch(model, val_loader, criterion, device)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f'Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}')
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), args.save_path)
            print('Best model saved.')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f'Early stop counter: {early_stop_counter}/{args.patience}')
            if early_stop_counter >= args.patience:
                print('Early stopping triggered.')
                break

    plot_loss_curves(train_losses, val_losses)
    evaluate(model, val_imgs, val_masks, args, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Net on OCT B-scans')
    parser.add_argument('--data-dir', required=True, help='Path to dataset root directory')
    parser.add_argument('--save-path', default='Weights_U-Net.pth', help='Where to save best weights')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    train(parser.parse_args())
