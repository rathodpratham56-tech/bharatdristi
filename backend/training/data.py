from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from backend.preprocessing import basic_preprocess_image, preprocess_image


class RetinalFolderDataset(Dataset):
    def __init__(self, root: Path, variant: str = "baseline"):
        self.root = Path(root)
        self.classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.samples = [(path, self.class_to_index[path.parent.name]) for path in self.root.rglob("*") if path.suffix.lower() in extensions and path.parent.name in self.class_to_index]
        self.variant = variant

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {path}")
        _, tensor, _ = preprocess_image(image) if self.variant == "proposed" else basic_preprocess_image(image)
        return tensor.squeeze(0), label
