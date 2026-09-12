from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class QualityConfig:
    min_brightness: float = 20.0
    max_brightness: float = 245.0
    min_blur_score: float = 2.0
    min_retinal_area: float = 0.20


@dataclass(frozen=True)
class QualityResult:
    acceptable: bool
    message: str
    blur_score: float
    brightness: float
    retinal_area: float
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _retinal_area(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    threshold = max(8.0, float(np.percentile(gray, 15)))
    mask = gray > threshold
    return float(np.mean(mask))


def check_image_quality(image_bgr: np.ndarray, config: QualityConfig | None = None) -> QualityResult:
    config = config or QualityConfig()
    if image_bgr is None or image_bgr.size == 0:
        return QualityResult(False, "Invalid image.", 0.0, 0.0, 0.0, ("invalid_image",))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    retinal_area = _retinal_area(image_bgr)
    reasons: list[str] = []
    if brightness < config.min_brightness:
        reasons.append("too_dark")
    if brightness > config.max_brightness:
        reasons.append("overexposed")
    if blur_score < config.min_blur_score:
        reasons.append("too_blurry")
    if retinal_area < config.min_retinal_area:
        reasons.append("insufficient_retinal_area")

    if reasons:
        return QualityResult(
            False,
            "Image quality is insufficient for reliable screening. Please upload a clearer retinal image.",
            blur_score,
            brightness,
            retinal_area,
            tuple(reasons),
        )
    return QualityResult(
        True,
        f"Acceptable (clarity {blur_score:.1f}, retinal area {retinal_area:.2f})",
        blur_score,
        brightness,
        retinal_area,
        (),
    )


def _crop_retinal_region(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > 8
    if not np.any(mask):
        return image_bgr
    ys, xs = np.where(mask)
    margin = max(4, int(0.02 * max(image_bgr.shape[:2])))
    x1, x2 = max(0, xs.min() - margin), min(image_bgr.shape[1], xs.max() + margin + 1)
    y1, y2 = max(0, ys.min() - margin), min(image_bgr.shape[0], ys.max() + margin + 1)
    return image_bgr[y1:y2, x1:x2]


def preprocess_image(image_bgr: np.ndarray, input_size: int = 224) -> tuple[np.ndarray, Any, dict[str, Any]]:
    cropped = _crop_retinal_region(image_bgr)
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_bgr = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(enhanced_rgb)
    tensor = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(image).unsqueeze(0)
    metadata = {
        "input_size": input_size,
        "operations": ["retinal_region_crop", "LAB_CLAHE", "resize", "ImageNet_normalization"],
        "visualization_note": "Preprocessing is intended to improve visibility while preserving retinal structures; it is not a clinical enhancement claim.",
    }
    return enhanced_rgb, tensor, metadata
