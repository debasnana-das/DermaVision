import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from guidance import get_educational_guidance
from skin_analysis import (
    assess_image_quality,
    load_detector,
    make_preview_overlay,
    predict_detections,
)

APP_TITLE = "DermaVision"
DEFAULT_MODEL_PATH = Path(__file__).with_name("acne_detector.pt")


@st.cache_resource(show_spinner="Loading the detection model...")
def get_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        return None
    return load_detector(path)


def create_report(quality: dict, detections: list[dict] | None, mode: str) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "DERMAVISION - IMAGE ANALYSIS SUMMARY",
        f"Generated: {generated_at}",
        "",
        f"Analysis mode: {mode}",
        f"Image quality indicator: {quality['score']}/100",
        f"Sharpness: {quality['sharpness_label']}",
        f"Brightness: {quality['brightness_label']}",
        "",
    ]

    if detections is None:
        lines.extend([
            "No trained detection model was available.",
            "No acne-lesion predictions were generated.",
            "The displayed preview is based on simple image-processing heuristics only.",
        ])
    else:
        lines.append(f"Detected candidate lesions: {len(detections)}")
        for index, item in enumerate(detections, start=1):
            lines.append(
                f"{index}. {item['class_name']} "
                f"(confidence: {item['confidence']:.3f})"
            )

    lines.extend([
        "",
        "LIMITATIONS",
        "This application is an educational software prototype. Its output is not a medical diagnosis,",
        "does not identify nutritional deficiencies, and must not be used to select or prescribe treatment.",
        "Image quality, lighting, skin tone, camera characteristics, and other factors may affect results.",
        "Consult a qualified healthcare professional for medical concerns.",
    ])
    return "\n".join(lines)


st.set_page_config(page_title=APP_TITLE, page_icon="D", layout="wide")

st.title(APP_TITLE)
st.caption("Image-based facial skin feature analysis and monitoring")

st.warning(
    "DermaVision is an educational prototype, not a medical diagnostic system. "
    "It cannot identify vitamin deficiencies or prescribe treatment. "
    "Consult a qualified healthcare professional for medical concerns."
)

with st.sidebar:
    st.header("Analysis settings")
    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Minimum confidence threshold used by the trained detection model.",
    )
    st.caption("Uploaded photographs are processed in memory and are not intentionally saved by this application.")

uploaded_file = st.file_uploader(
    "Upload a front-facing facial photograph",
    type=["jpg", "jpeg", "png", "webp"],
)

model_path = os.environ.get("DERMAVISION_MODEL_PATH", str(DEFAULT_MODEL_PATH))
model = get_model(model_path)

if uploaded_file is None:
    st.info("Upload a photograph to begin.")
    st.markdown(
        """
        **Application workflow**

        1. Validate the uploaded image.
        2. Assess basic image quality.
        3. Analyse visible features using the available model.
        4. Display annotated results and general educational information.
        """
    )
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.asarray(image, dtype=np.uint8)
    except Exception:
        st.error("The selected file could not be opened as an image. Please upload a valid image.")
        st.stop()

    quality = assess_image_quality(image_array)

    image_col, result_col = st.columns(2)
    with image_col:
        st.subheader("Uploaded photograph")
        st.image(image_array, use_container_width=True)

    detections = None
    analysis_mode = "Image-processing preview"

    if model is not None:
        prediction = predict_detections(model, image_array, confidence=confidence)
        detections = prediction["detections"]
        result_image = prediction["annotated_image"]
        analysis_mode = "Trained object-detection model"
    else:
        result_image, preview_stats = make_preview_overlay(image_array)

    with result_col:
        st.subheader("Analysis result")
        st.image(result_image, use_container_width=True)
        if model is None:
            st.warning(
                "Trained model weights were not found. This preview highlights broad colour and "
                "contrast regions only; it does not detect or count acne lesions."
            )
        else:
            st.success("Trained detection model loaded.")

    st.subheader("Image quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Capture indicator", f"{quality['score']}/100")
    q2.metric("Sharpness", quality["sharpness_label"])
    q3.metric("Brightness", quality["brightness_label"])
    st.write(quality["message"])

    if detections is not None:
        st.subheader("Detection summary")
        st.metric("Detected candidate lesions", len(detections))
        if detections:
            st.dataframe(
                [
                    {
                        "Feature": item["class_name"],
                        "Confidence": round(item["confidence"], 3),
                    }
                    for item in detections
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("The model did not return detections above the selected confidence threshold.")
    else:
        st.subheader("Image-processing preview")
        st.metric("Highlighted pixel area", f"{preview_stats['candidate_area_pct']:.2f}%")
        st.caption(
            "This percentage is a heuristic image statistic, not a lesion count, skin-condition "
            "measurement, or severity score."
        )

    st.subheader("General educational information")
    for item in get_educational_guidance():
        st.markdown(f"- {item}")

    if detections is not None:
        if "history" not in st.session_state:
            st.session_state.history = []

        st.subheader("Session progress record")
        if st.button("Add this result to the session record"):
            st.session_state.history.append(
                {
                    "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "detected_candidates": len(detections),
                    "image_quality_indicator": quality["score"],
                }
            )
            st.success("The measurement has been added to this session.")

        if st.session_state.history:
            st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)

    report_text = create_report(quality, detections, analysis_mode)
    st.download_button(
        "Download analysis summary",
        data=report_text.encode("utf-8"),
        file_name="dermavision_analysis.txt",
        mime="text/plain",
    )
