"""Create reproducible grouped manifests from a user-provided CSV.

Expected columns: image_path,label,patient_id. Patient IDs are required for
leakage-safe grouped splitting; without them the output records the limitation.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def make_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, str]:
    if "patient_id" in frame.columns and frame["patient_id"].notna().all():
        groups = frame[["patient_id", "label"]].drop_duplicates("patient_id")
        groups = groups.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(groups)
        train_ids = set(groups.iloc[: int(n * 0.70)]["patient_id"])
        valid_ids = set(groups.iloc[int(n * 0.70): int(n * 0.85)]["patient_id"])
        frame = frame.copy()
        frame["split"] = frame["patient_id"].map(lambda value: "train" if value in train_ids else "validation" if value in valid_ids else "test")
        return frame, "patient_grouped"
    frame = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(frame)
    frame["split"] = ["train" if i < int(n * 0.70) else "validation" if i < int(n * 0.85) else "test" for i in range(n)]
    return frame, "image_level_patient_id_unavailable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    required = {"image_path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    output, method = make_split(frame, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_frame in output.groupby("split"):
        split_frame.to_csv(args.output_dir / f"{split}.csv", index=False)
    (args.output_dir / "split_metadata.json").write_text(json.dumps({"seed": args.seed, "method": method, "rows": len(output)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
