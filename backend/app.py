import base64
import os

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from .config import settings
    from .model import build_model, load_checkpoint
    from .preprocessing import QualityConfig, check_image_quality, preprocess_image
    from .xai import generate_gradcam
except ImportError:
    from config import settings
    from model import build_model, load_checkpoint
    from preprocessing import QualityConfig, check_image_quality, preprocess_image
    from xai import generate_gradcam


app = FastAPI(title="BharatDrishti Clinical AI Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")
model_variant = settings.model_variant if settings.model_variant in {"baseline", "proposed"} else "baseline"
checkpoint_path = settings.proposed_model_path if model_variant == "proposed" else settings.model_path
model = build_model(model_variant, settings.num_classes).to(device)
model, model_loaded, model_warning = load_checkpoint(model, checkpoint_path, device)
model.eval()

GRADE_INFO = {
    0: {"grade": "Grade 0 (No DR)", "desc": "No clinically significant diabetic retinopathy detected by this model.", "lesions": "No significant DR-related abnormality identified by the model.", "rec": "Routine eye screening advised.", "ref": "No immediate referral indicated based on this screening result."},
    1: {"grade": "Grade 1 (Mild)", "desc": "Mild non-proliferative diabetic retinopathy classification.", "lesions": "Features associated with mild disease may have contributed to the classification.", "rec": "Regular ophthalmic follow-up recommended.", "ref": "Routine ophthalmology follow-up."},
    2: {"grade": "Grade 2 (Moderate)", "desc": "Moderate non-proliferative diabetic retinopathy classification.", "lesions": "Features associated with moderate disease may have contributed to the classification.", "rec": "Ophthalmology consultation recommended.", "ref": "Ophthalmology or tele-ophthalmology consultation."},
    3: {"grade": "Grade 3 (Severe)", "desc": "Severe non-proliferative diabetic retinopathy classification.", "lesions": "Features associated with severe disease may have contributed to the classification.", "rec": "Prompt specialist evaluation recommended.", "ref": "Urgent ophthalmology or tele-ophthalmology referral."},
    4: {"grade": "Grade 4 (Proliferative)", "desc": "Proliferative diabetic retinopathy classification.", "lesions": "Features associated with proliferative disease may have contributed to the classification.", "rec": "Prompt retinal specialist evaluation recommended.", "ref": "Urgent retina specialist referral."},
}


def image_to_base64(image_rgb: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("Failed to encode image.")
    return base64.b64encode(encoded).decode("utf-8")


def screening_interpretation(pred_class: int, confidence: float) -> str:
    if pred_class >= 2 or confidence < settings.min_confidence + 0.15:
        return "Concerning or uncertain screening result; professional ophthalmological consultation is recommended."
    return "Screening result is suitable for review; it is not a definitive diagnosis."


@app.post("/analyze-fundus")
async def analyze_fundus(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            return {"status": "Error", "message": "Uploaded file is empty."}
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"status": "Error", "message": "Unable to decode uploaded image."}

        quality = check_image_quality(image, QualityConfig(
            min_brightness=settings.min_brightness,
            max_brightness=settings.max_brightness,
            min_blur_score=settings.min_blur_score,
            min_retinal_area=settings.min_retinal_area,
        ))
        quality_payload = quality.as_dict()
        quality_payload["reasons"] = list(quality.reasons)
        if not quality.acceptable:
            return {"status": "Error", "iqa_message": quality.message, "image_quality": quality_payload}

        processed_rgb, input_tensor, preprocessing_metadata = preprocess_image(image, settings.input_size)
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(input_tensor), dim=1)[0]
            confidence, predicted = torch.max(probabilities, dim=0)
        pred_class = int(predicted.item())
        confidence_value = float(confidence.item())
        confidence_text = f"{confidence_value * 100:.2f}%"
        if confidence_value < settings.min_confidence:
            return {
                "status": "Error",
                "iqa_message": "AI confidence is too low. The image could not be classified reliably.",
                "confidence_score": confidence_text,
                "image_quality": quality_payload,
                "preprocessing": preprocessing_metadata,
            }

        result = GRADE_INFO.get(pred_class, GRADE_INFO[0])
        overlay_rgb, pure_heatmap_rgb = generate_gradcam(model, input_tensor.requires_grad_(True), pred_class, image)
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {
            "status": "Success",
            "iqa_metric": quality.message,
            "image_quality": quality_payload,
            "preprocessing": preprocessing_metadata,
            "model": {
                "variant": model_variant,
                "name": "ResNet50" if model_variant == "baseline" else "ResNet50 + CBAM attention",
                "checkpoint_loaded": model_loaded,
                "warning": model_warning,
            },
            "prediction": {
                "grade": result["grade"],
                "confidence_score": confidence_text,
                "description": result["desc"],
                "detected_lesions": result["lesions"],
                "recommendation": result["rec"],
                "referral": result["ref"],
                "screening_interpretation": screening_interpretation(pred_class, confidence_value),
            },
            "clahe_image": f"data:image/png;base64,{image_to_base64(processed_rgb)}",
            "heatmap_image": f"data:image/png;base64,{image_to_base64(overlay_rgb)}",
            "pure_heatmap_image": f"data:image/png;base64,{image_to_base64(pure_heatmap_rgb)}",
            "original_image": f"data:image/png;base64,{image_to_base64(original_rgb)}",
            "xai_note": "Highlighted regions indicate areas that contributed most to the model prediction. Grad-CAM does not prove a medical diagnosis.",
            "disclaimer": "This AI system is intended for screening/research assistance and does not replace professional ophthalmological diagnosis.",
        }
    except Exception as exc:
        print(f"Analysis error: {exc}")
        return {"status": "Error", "message": "The screening request could not be completed."}


@app.get("/")
def root():
    return {
        "status": "BharatDrishti AI Engine running",
        "model": "ResNet50" if model_variant == "baseline" else "ResNet50 + CBAM attention",
        "classes": settings.num_classes,
        "gradcam": "Enabled",
        "checkpoint_loaded": model_loaded,
        "warning": model_warning,
    }
