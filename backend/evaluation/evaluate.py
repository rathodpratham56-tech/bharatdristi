"""Evaluate a supplied checkpoint on an isolated ImageFolder test set."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.evaluation.metrics import classification_metrics
from backend.model import build_model, load_checkpoint
from backend.training.data import RetinalFolderDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["baseline", "proposed"], default="baseline")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RetinalFolderDataset(args.data_dir, args.variant)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model, loaded, warning = load_checkpoint(build_model(args.variant, len(dataset.classes)), args.checkpoint, device)
    if not loaded:
        raise RuntimeError(warning)
    labels, predictions, probabilities = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            output = model(images.to(device))
            probabilities.extend(torch.softmax(output, dim=1).cpu().numpy())
            predictions.extend(output.argmax(dim=1).cpu().tolist())
            labels.extend(targets.tolist())
    result = classification_metrics(labels, predictions, np.asarray(probabilities), dataset.classes)
    result["variant"] = args.variant
    result["checkpoint"] = str(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
