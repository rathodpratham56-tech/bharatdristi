from dataclasses import dataclass
from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    model_path: Path = ROOT_DIR / "bharatdrishti_resnet50_best.pth"
    input_size: int = 224
    num_classes: int = 5
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.40"))
    min_brightness: float = float(os.getenv("MIN_BRIGHTNESS", "20.0"))
    max_brightness: float = float(os.getenv("MAX_BRIGHTNESS", "245.0"))
    min_blur_score: float = float(os.getenv("MIN_BLUR_SCORE", "2.0"))
    min_retinal_area: float = float(os.getenv("MIN_RETINAL_AREA", "0.20"))


settings = Settings()
