from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 5) -> nn.Module:
	model = models.resnet50(weights=None)
	model.fc = nn.Linear(model.fc.in_features, num_classes)
	return model


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> tuple[nn.Module, bool, str | None]:
	if not path.exists():
		return model, False, f"Checkpoint not found: {path}"
	checkpoint = torch.load(path, map_location=device)
	state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
	state = {key.removeprefix("module."): value for key, value in state.items()}
	missing, unexpected = model.load_state_dict(state, strict=False)
	if missing or unexpected:
		return model, False, f"Checkpoint mismatch; missing={len(missing)}, unexpected={len(unexpected)}"
	model.to(device).eval()
	return model, True, None

__all__ = ["build_model", "load_checkpoint"]
