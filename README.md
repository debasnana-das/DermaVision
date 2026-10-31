# DermaVision

DermaVision is an image-based facial skin analysis application developed using computer vision and machine learning. The initial model-development focus is the localisation of visible acne-like features. Image-quality assessment, measurement summaries and general skincare information are supporting functions.

DermaVision is intended for educational and demonstration purposes. It is not a medical diagnostic tool and does not replace assessment by a qualified healthcare professional.

## Features

- Upload facial photographs in JPG, JPEG, PNG or WEBP format.
- Assess basic image sharpness and brightness.
- Display an exploratory image-processing preview when trained model weights are unavailable.
- Run a trained object-detection model and display annotated predictions when model weights are provided.
- Review general skincare information and safety guidance.
- Download an analysis summary as a plain-text file or a formatted PDF.

The exploratory preview uses simple colour and contrast heuristics. It does not detect or count confirmed acne lesions. Lesion detection is available only after a suitable model has been trained and its weights have been configured.

## Technology

- Python
- Streamlit
- OpenCV
- PyTorch
- Ultralytics
- NumPy
- Pillow
- ReportLab (PDF report generation)

## Repository Structure

```text
dermavision/
├── app.py
├── skin_analysis.py
├── guidance.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

Training datasets, personal photographs, generated outputs and model-weight files are kept outside the source repository.

## Installation

Python 3.10 or later is recommended.

Create and activate a virtual environment.

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application from the repository root.

```bash
streamlit run app.py
```

When trained model weights are not configured, the application displays a clearly labelled image-processing preview. This preview is not an acne detector.

To run inference with trained weights, set the `DERMAVISION_MODEL_PATH` environment variable to the model file before starting Streamlit.

**Windows PowerShell**

```powershell
$env:DERMAVISION_MODEL_PATH = "C:\path\to\best.pt"
streamlit run app.py
```

**Linux or macOS**

```bash
export DERMAVISION_MODEL_PATH="/path/to/best.pt"
streamlit run app.py
```

The model file must be compatible with the Ultralytics inference interface used by `skin_analysis.py`.

## Model Training

The training entry point can prepare ACNE04 from its Hugging Face dataset repository and convert its lesion bounding boxes to YOLO format. The canonical fold-0 test partition is preserved as a held-out test set; ten percent of the official training partition is reserved for validation using a fixed random seed.

The `datasets` package is required only for dataset preparation. Install it separately on the training computer:

```bash
pip install "datasets>=2.19,<4.0"
```

A YOLO-format dataset should have separate training and validation images and labels. Each label file uses the format:

```text
class_id x_center y_center width height
```

Coordinates must be normalised to the range 0–1. The dataset configuration file must specify the training path, validation path and class names.

Example configuration:

```yaml
path: /absolute/path/to/prepared_dataset
train: images/train
val: images/val
names:
  0: acne_lesion
```

First, prepare the dataset locally:

```bash
python train.py --prepare-acne04
```

This downloads the dataset and writes the prepared images, labels and configuration under `data/acne04_yolo/`. The original dataset is approximately 1.13 GB; additional disk space is required for extracted images and training outputs. The output directory is excluded from Git.

Then train the initial detector:

```bash
python train.py --data data/acne04_yolo/data.yaml --epochs 100 --imgsz 640 --batch 8
```

To use a GPU, install a PyTorch build compatible with the CUDA version and hardware available on the training computer. For a CPU-only run, specify `--device cpu`; training may be considerably slower.

The training script saves the best weights under the configured output directory, normally:

```text
outputs/acne_detection/weights/best.pt
```

Use the resulting file for local inference by setting `DERMAVISION_MODEL_PATH`. Training performance must be evaluated on held-out data before the model is used for demonstration.

## Dataset and Model Use

No dataset or trained model weights are included in this repository. The current preparation script uses the `Layered-Labs/ACNE04` Hugging Face dataset release and expects the `image`, `file_name`, `lesion_boxes` and `split` fields. ACNE04 is intended for academic and research use only. Its original licensing conditions prohibit commercial use without permission from the original authors. Review the dataset card and original source terms before downloading or using it. Cite both the dataset release and the original ACNE04 paper in academic reports. The dataset must be prepared separately on the training computer.

The application looks for `acne_detector.pt` in the repository root by default. Alternatively, set `DERMAVISION_MODEL_PATH` to the model file path through the hosting environment. The original training dataset is not required for inference. After an image is analysed, the user can download the same analysis summary as either a TXT file or a formatted PDF.

## Deployment

For Streamlit Community Cloud:

1. Place the selected trained checkpoint in the repository root and name it `acne_detector.pt`.
2. Confirm that `app.py`, `requirements.txt`, and `acne_detector.pt` are committed to the `main` branch.
3. Sign in at https://share.streamlit.io and connect the GitHub account that owns this repository.
4. Select **Create app**, then choose `debasnana-das/DermaVision`, branch `main`, and entry point `app.py`.
5. Deploy the application and wait for the dependency installation and startup checks to finish.
6. Open the assigned `streamlit.app` URL and test it with a suitable photograph.

Streamlit Community Cloud must be authorised to access the private GitHub repository before deployment. A private application can be restricted to named viewers; invited viewers must authenticate. If the application is configured as public, anyone with its URL may access it. Review the sharing setting before distributing the link.

The hosted application performs inference using the trained model weights. It does not download or require the original training dataset.

## Development Status

The repository contains the Streamlit application, image-analysis utilities, educational guidance, dependency list and a YOLO-compatible training entry point.

The ACNE04 download and dataset-conversion workflow is implemented in `train.py`. An initial YOLOv8n acne-lesion detection model has been trained and its checkpoint is included as `acne_detector.pt`. Initial evaluation indicates limited detection and localisation performance. The model is an experimental academic prototype and is not clinically validated.

## Limitations and Responsible Use

- Model predictions depend on image quality, dataset coverage, annotation quality and model performance.
- Lighting, camera characteristics, skin tone, image angle, makeup and facial hair may affect image analysis.
- A facial photograph cannot establish vitamin or mineral deficiencies.
- The application must not be used to diagnose a medical condition, prescribe treatment or make clinical decisions.
- Users should consult a qualified healthcare professional for persistent, painful, worsening or otherwise concerning skin changes.
- Uploaded photographs should be handled with appropriate consent and privacy safeguards.

## Licence

The project source code is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the `LICENSE` file for the complete licence text.

Third-party datasets, pretrained weights and dependencies may have separate licence and usage conditions. The project licence does not replace or override those conditions. ACNE04 is not included in this repository; review its original dataset terms before use or redistribution.


## Dataset References

- ACNE04 dataset release: https://huggingface.co/datasets/Layered-Labs/ACNE04
- Original ACNE04 source repository: https://github.com/xpwu95/LDL
- Wu et al., "Joint Acne Image Grading and Counting via Label Distribution Learning", ICCV 2019.
