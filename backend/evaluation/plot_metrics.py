"""Plot metrics from real JSON outputs only."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    proposed = json.loads(args.proposed.read_text(encoding="utf-8"))
    names = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall_sensitivity", "macro_f1"]
    x = list(range(len(names)))
    plt.figure(figsize=(9, 5))
    plt.bar([i - 0.2 for i in x], [baseline[name] for name in names], width=0.4, label="Baseline ResNet50")
    plt.bar([i + 0.2 for i in x], [proposed[name] for name in names], width=0.4, label="Proposed ResNet50 + CBAM")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
