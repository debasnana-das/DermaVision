import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import streamlit as st
from PIL import Image
import requests
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

from guidance import get_educational_guidance
from skin_analysis import (
    assess_image_quality,
    load_detector,
    make_preview_overlay,
    predict_detections,
)

APP_TITLE = "DermaVision"
DEFAULT_MODEL_PATH = Path(__file__).with_name("acne_detector.pt")
ACNE04_DATASET_URL = "https://huggingface.co/datasets/Layered-Labs/ACNE04"


@st.cache_data(ttl=3600, show_spinner=False)
def get_acne04_samples() -> list[dict]:
    """Fetch one real sample per severity from ACNE04's held-out test partition."""
    samples = []
    for severity in ("mild", "moderate", "severe", "very_severe"):
        response = requests.get(
            "https://datasets-server.huggingface.co/filter",
            params={
                "dataset": "Layered-Labs/ACNE04",
                "config": "default",
                "split": "train",
                "where": f'"split"=\'test\' AND "severity"=\'{severity}\'',
                "offset": 0,
                "length": 1,
            },
            timeout=(5, 30),
        )
        response.raise_for_status()
        rows = response.json().get("rows", [])
        if not rows:
            continue
        row = rows[0].get("row", {})
        image_info = row.get("image", {})
        image_url = image_info.get("src") if isinstance(image_info, dict) else None
        if not image_url:
            continue
        samples.append({
            "label": f"{severity.replace('_', ' ').title()} — {row.get('file_name', 'ACNE04 sample')}",
            "severity": severity.replace("_", " "),
            "file_name": str(row.get("file_name", "ACNE04 sample")),
            "image_url": image_url,
            "lesion_count": row.get("num_lesions_annotated"),
            "split": row.get("split", "test"),
        })
    if not samples:
        raise RuntimeError("No ACNE04 test samples were returned by the dataset viewer.")
    return samples


@st.cache_data(ttl=3600, show_spinner=False)
def get_acne04_image(image_url: str) -> bytes:
    """Fetch a single image from a Hugging Face dataset-viewer URL."""
    from urllib.parse import urlparse

    parsed = urlparse(image_url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not (
        host == "huggingface.co" or host.endswith(".huggingface.co")
    ):
        raise ValueError("Unexpected image host returned by the ACNE04 dataset viewer.")
    response = requests.get(image_url, timeout=(5, 30))
    response.raise_for_status()
    if not response.headers.get("Content-Type", "").startswith("image/"):
        raise ValueError("The ACNE04 sample URL did not return an image.")
    if len(response.content) > 15 * 1024 * 1024:
        raise ValueError("The ACNE04 sample exceeds the 15 MB demo limit.")
    return response.content


@st.cache_resource(show_spinner="Loading the detection model...")
def get_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        return None, None

    try:
        return load_detector(path), None
    except Exception as exc:
        return None, str(exc)


def create_report(
    quality: dict,
    detections: list[dict] | None,
    mode: str,
    educational_guidance: list[str],
) -> str:
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
        "GENERAL EDUCATIONAL INFORMATION",
        *[f"- {item}" for item in educational_guidance],
        "",
        "LIMITATIONS",
        "This application is an educational software prototype. Its output is not a medical diagnosis,",
        "does not identify nutritional deficiencies, and must not be used to select or prescribe treatment.",
        "Image quality, lighting, skin tone, camera characteristics, and other factors may affect results.",
        "Consult a qualified healthcare professional for medical concerns.",
    ])
    return "\n".join(lines)


def create_pdf_report(report_text: str) -> bytes:
    """Create a downloadable PDF using the same content as the text report."""
    buffer = BytesIO()
    page_width, _ = A4
    left_margin = right_margin = 18 * mm

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=right_margin,
        leftMargin=left_margin,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="DermaVision - Image Analysis Summary",
        author="DermaVision",
        subject="Educational facial image analysis summary",
    )

    title_style = ParagraphStyle(
        "DermaVisionTitle",
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DermaVisionSubtitle",
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#667085"),
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "DermaVisionSection",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "DermaVisionBody",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#283444"),
        spaceAfter=3,
    )
    meta_style = ParagraphStyle(
        "DermaVisionMeta",
        parent=body_style,
        fontSize=8.5,
        textColor=colors.HexColor("#667085"),
    )

    story = [
        Paragraph("DERMAVISION - IMAGE ANALYSIS SUMMARY", title_style),
        Paragraph(
            "Image-based facial skin feature analysis and monitoring",
            subtitle_style,
        ),
        HRFlowable(
            width="100%",
            thickness=0.8,
            color=colors.HexColor("#D7DEE8"),
            spaceAfter=10,
        ),
    ]

    for index, line in enumerate(report_text.splitlines()):
        clean_line = line.strip()
        if not clean_line:
            story.append(Spacer(1, 3))
        elif index == 0 and clean_line.upper().startswith("DERMAVISION"):
            continue
        elif clean_line.upper() in {"GENERAL EDUCATIONAL INFORMATION", "LIMITATIONS"}:
            story.append(Paragraph(escape(clean_line), section_style))
        elif clean_line.startswith("Generated:"):
            story.append(Paragraph(escape(clean_line), meta_style))
        elif clean_line.startswith("- "):
            story.append(
                Paragraph(escape(clean_line[2:]), body_style, bulletText="-")
            )
        else:
            story.append(Paragraph(escape(clean_line), body_style))

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7DEE8"))
        canvas.setLineWidth(0.5)
        canvas.line(left_margin, 14 * mm, page_width - right_margin, 14 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawString(
            left_margin,
            9 * mm,
            "DermaVision | Educational prototype - not a medical diagnosis",
        )
        canvas.drawRightString(
            page_width - right_margin,
            9 * mm,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buffer.getvalue()


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
        min_value=0.05,
        max_value=0.90,
        value=0.16,
        step=0.01,
        help=(
            "Minimum confidence threshold used by the trained detection model. "
            "Lower values show more candidate boxes and may increase false detections."
        ),
    )
    st.caption("Uploaded photographs are processed in memory and are not intentionally saved by this application.")

st.subheader("Choose an image to analyse")
image_source_mode = st.radio(
    "Image source",
    ["Try an ACNE04 sample", "Upload your own photograph"],
    horizontal=True,
)
uploaded_file = None
image_caption = "Uploaded photograph"
image_source_details = "User-provided photograph"
selected_sample = None

if image_source_mode == "Try an ACNE04 sample":
    st.caption(
        "Samples are retrieved from the ACNE04 held-out test split when selected. "
        "They are not stored in this GitHub repository."
    )
    st.markdown(f"Dataset source: [ACNE04 on Hugging Face]({ACNE04_DATASET_URL})")
    try:
        sample_catalog = get_acne04_samples()
        selected_label = st.selectbox(
            "Select a sample photograph",
            [sample["label"] for sample in sample_catalog],
        )
        selected_sample = next(
            sample for sample in sample_catalog if sample["label"] == selected_label
        )
        sample_bytes = get_acne04_image(selected_sample["image_url"])
        uploaded_file = BytesIO(sample_bytes)
        image_caption = "Selected ACNE04 sample"
        image_source_details = (
            f"ACNE04 held-out test sample; severity label: {selected_sample['severity']}; "
            f"filename: {selected_sample['file_name']}"
        )
    except Exception as exc:
        st.warning(
            "The ACNE04 sample gallery is temporarily unavailable. "
            "You can still switch to upload mode or open the dataset directly."
        )
        st.caption(f"Gallery error: {exc}")
else:
    uploaded_file = st.file_uploader(
        "Upload a front-facing facial photograph",
        type=["jpg", "jpeg", "png", "webp"],
    )

model_path = os.environ.get("DERMAVISION_MODEL_PATH", str(DEFAULT_MODEL_PATH))
model, model_error = get_model(model_path)

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
        st.subheader(image_caption)
        st.image(image_array, use_container_width=True)
        if selected_sample is not None:
            st.caption(
                f"Dataset severity label: {selected_sample['severity']} | "
                f"Annotated boxes: {selected_sample['lesion_count']} | "
                f"Split: {selected_sample['split']}"
            )

    detections = None
    analysis_mode = "Image-processing preview"

    if model is not None:
        prediction = predict_detections(model, image_array, confidence=confidence)
        detections = prediction["detections"]
        result_image = prediction["annotated_image"]
        analysis_mode = "Trained object-detection model"
    else:
        result_image, preview_stats = make_preview_overlay(image_array)
        if model_error:
            st.warning(
                "The configured model weights could not be loaded. "
                "The application is showing the non-diagnostic image-processing preview instead."
            )

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
    educational_guidance = get_educational_guidance()
    for item in educational_guidance:
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

    report_text = create_report(quality, detections, analysis_mode, educational_guidance)
    report_pdf = create_pdf_report(report_text)

    text_download_col, pdf_download_col = st.columns(2)
    with text_download_col:
        st.download_button(
            "Download analysis summary (TXT)",
            data=report_text.encode("utf-8"),
            file_name="dermavision_analysis.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with pdf_download_col:
        st.download_button(
            "Download analysis summary (PDF)",
            data=report_pdf,
            file_name="dermavision_analysis.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
