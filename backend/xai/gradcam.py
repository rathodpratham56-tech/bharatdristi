from typing import Any

import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: Any, _inputs: Any, output: torch.Tensor) -> None:
        self.activations = output

    def _save_gradient(self, _module: Any, _inputs: Any, outputs: Any) -> None:
        self.gradients = outputs[0]

    def generate(self, tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(tensor)
        output[:, target_class].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))[0].detach().cpu().numpy()
        cam -= cam.min()
        maximum = cam.max()
        return cam / maximum if maximum > 1e-8 else np.zeros_like(cam)

    def close(self) -> None:
        self.forward_hook.remove()
        self.backward_hook.remove()


def generate_gradcam(model: torch.nn.Module, tensor: torch.Tensor, target_class: int, source_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_layer = model.layer4[-1] if hasattr(model, "layer4") else model.layer4[-1]
    generator = GradCAM(model, target_layer)
    try:
        cam = generator.generate(tensor, target_class)
    finally:
        generator.close()
    height, width = source_bgr.shape[:2]
    cam = cv2.resize(cam, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(source_bgr, 0.55, heatmap, 0.45, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
