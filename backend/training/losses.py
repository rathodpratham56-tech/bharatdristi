import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.class_weights if self.class_weights.numel() else None
        cross_entropy = F.cross_entropy(logits, targets, weight=weights, reduction="none")
        return (((1 - torch.exp(-cross_entropy)) ** self.gamma) * cross_entropy).mean()
