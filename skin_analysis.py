from pathlib import Path

import cv2
import numpy as np


def assess_image_quality(image_rgb: np.ndarray) -> dict:
    """Return basic technical image-quality indicators.

    These indicators are intended for capture guidance only. They are not
    clinical quality measures and have not been validated for diagnosis.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with three colour channels.")

    if image_rgb.size == 0:
        raise ValueError("The supplied image is empty.")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    sharpness_indicator = min(100.0, sharpness / 2.0)
    exposure_penalty = 0.0
    if brightness < 40:
        exposure_penalty = min(100.0, (40.0 - brightness) * 1.5)
    elif brightness > 220:
        exposure_penalty = min(100.0, (brightness - 220.0) * 1.5)

    score = int(
        max(
            0,
            min(
                100,
                round(0.8 * sharpness_indicator + 0.2 * (100.0 - exposure_penalty)),
            ),
        )
    )

    if sharpness >= 120:
        sharpness_label = "High"
    elif sharpness >= 45:
        sharpness_label = "Moderate"
    else:
        sharpness_label = "Low"

    if brightness < 70:
        brightness_label = "Dark"
    elif brightness > 190:
        brightness_label = "Bright"
    else:
        brightness_label = "Moderate"

    usable = score >= 35 and 35 <= brightness <= 220
    message = (
        "The image appears suitable for an initial software demonstration."
        if usable
        else "Image quality may affect the analysis. Try even lighting and a steady camera."
    )

    return {
        "score": score,
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "sharpness_label": sharpness_label,
        "brightness_label": brightness_label,
        "usable": usable,
        "message": message,
    }


def make_preview_overlay(image_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Create a non-diagnostic colour/contrast preview.

    The highlighted pixels are heuristic candidates only. This function is
    not an acne detector and must not be interpreted as one.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with three colour channels.")

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    red = image_rgb[:, :, 0].astype(np.int16)
    green = image_rgb[:, :, 1].astype(np.int16)

    red_candidate = (
        (red - green > 15)
        & (hsv[:, :, 1] > 45)
        & (hsv[:, :, 2] > 35)
    )

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    local_average = cv2.GaussianBlur(gray, (0, 0), sigmaX=7)
    dark_candidate = (
        local_average.astype(np.int16) - gray.astype(np.int16) > 10
    )

    candidate_mask = red_candidate | dark_candidate
    kernel = np.ones((3, 3), dtype=np.uint8)
    candidate_mask = cv2.morphologyEx(
        candidate_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel,
    ).astype(bool)

    highlighted = image_rgb.copy()
    highlighted[candidate_mask] = np.array([255, 0, 180], dtype=np.uint8)
    overlay = cv2.addWeighted(image_rgb, 0.70, highlighted, 0.30, 0)

    candidate_area_pct = (
        100.0 * float(np.count_nonzero(candidate_mask)) / candidate_mask.size
    )

    return overlay, {"candidate_area_pct": candidate_area_pct}


def load_detector(model_path: str | Path):
    """Load trained YOLO-compatible weights for inference."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights were not found: {path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required to load the trained detection model. "
            "Install the packages listed in requirements.txt."
        ) from exc

    return YOLO(str(path))


def predict_detections(
    model,
    image_rgb: np.ndarray,
    confidence: float = 0.35,
) -> dict:
    """Run object detection and return an annotated RGB image and predictions."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be between 0 and 1.")

    results = model.predict(
        source=image_rgb,
        conf=confidence,
        verbose=False,
    )

    annotated = image_rgb.copy()
    detections = []

    if not results:
        return {"annotated_image": annotated, "detections": detections}

    result = results[0]
    names = result.names

    if result.boxes is not None:
        for box in result.boxes:
            coordinates = box.xyxy[0].cpu().numpy().astype(int)
            score = float(box.conf[0].cpu().item())
            class_id = int(box.cls[0].cpu().item())

            x1, y1, x2, y2 = coordinates.tolist()
            class_name = (
                names.get(class_id, str(class_id))
                if isinstance(names, dict)
                else names[class_id]
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 180, 0),
                2,
            )
            label = f"{class_name} {score:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 180, 0),
                1,
                cv2.LINE_AA,
            )

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(class_name),
                    "confidence": score,
                    "box": [x1, y1, x2, y2],
                }
            )

    return {"annotated_image": annotated, "detections": detections}
