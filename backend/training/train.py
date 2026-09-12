"""Train the baseline or proposed model on a local ImageFolder dataset.

This script never downloads data. Keep the final test directory isolated.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backend.model import build_model
from backend.training.data import RetinalFolderDataset
from backend.training.losses import FocalLoss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["baseline", "proposed"], default="proposed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RetinalFolderDataset(args.data_root / "train", args.variant)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    counts = torch.bincount(torch.tensor(dataset.targets), minlength=len(dataset.classes)).float()
    class_weights = counts.sum() / counts.clamp_min(1)
    class_weights = class_weights / class_weights.mean()
    model = build_model(args.variant, len(dataset.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = FocalLoss(class_weights=class_weights.to(device))
    model.train()
    for _ in range(args.epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": dataset.classes, "variant": args.variant}, args.output)


if __name__ == "__main__":
    main()
