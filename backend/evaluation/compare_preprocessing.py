"""Run baseline and improved evaluations separately; never invent comparison values."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--improved", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = {
        "baseline": json.loads(args.baseline.read_text(encoding="utf-8")),
        "improved": json.loads(args.improved.read_text(encoding="utf-8")),
        "note": "Values are copied from the supplied evaluation outputs; no result is generated without real evaluation files.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
