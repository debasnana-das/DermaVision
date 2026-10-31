# DermaVision

DermaVision is a computer vision project for image-based analysis of visible facial skin features and monitoring changes across photographs. The initial development focus is acne-feature detection, with image-quality assessment and educational skincare information as supporting components.

The system is intended for educational and demonstrative use. It is not a medical diagnostic tool and does not replace assessment by a qualified healthcare professional.

## Objectives

- Assess whether an uploaded facial photograph is suitable for analysis.
- Detect and localise supported visible skin features using image processing and machine learning.
- Present measured results with appropriate limitations.
- Compare measurements from photographs captured under reasonably consistent conditions.
- Provide general, evidence-based skincare information.

## Planned Components

1. **Image input and quality assessment** — image upload, validation and checks for basic capture quality.
2. **Skin-feature analysis** — an initial acne-feature detection model, followed by additional feature categories only where suitable data and validation are available.
3. **Measurement and visualisation** — annotated images and quantitative summaries of model predictions.
4. **Progress monitoring** — comparison of measurements across photographs while accounting for differences in capture conditions.
5. **Educational guidance** — general information and safety guidance, without diagnosis or personalised prescriptions.

## Technology

The planned implementation uses:

- Python
- Streamlit
- OpenCV
- PyTorch

The final dependencies and versions will be recorded in `requirements.txt` as the implementation is established.

## Repository Contents

The repository will contain application source code, model-training and evaluation code, configuration, and project documentation.

Training datasets, personal photographs, generated outputs, and large model-weight files are maintained separately from the source repository. Dataset access instructions, attribution, and applicable usage conditions will be documented before each dataset is used. Model weights required by the hosted application will be provided through the deployment environment or an appropriate external storage location.

## Development Status

The repository has been created and the initial project documentation is in place. Application development, dataset preparation, model training, evaluation, and deployment are pending.

Features will be identified as complete only after implementation and testing.

## Intended Workflow

1. Prepare and validate permitted training data.
2. Develop the image-processing and model-training pipeline.
3. Evaluate the model on data held out from training.
4. Integrate the validated inference pipeline into the Streamlit application.
5. Deploy the application with its required model weights.
6. Verify the hosted application using images not used during model training.

## Limitations and Responsible Use

- Predictions depend on image quality, dataset coverage, annotation quality, and model performance.
- Differences in lighting, camera characteristics, skin tone, image angle, makeup, and facial hair may affect results.
- A photograph cannot establish vitamin or mineral deficiencies.
- The application must not be used to diagnose a condition, prescribe treatment, or make clinical decisions.
- Users should consult a qualified healthcare professional for persistent, painful, worsening, or otherwise concerning skin changes.

## Running the Application

Local installation and execution instructions will be added after the application entry point and dependency list are finalised.

## Licence and Data Use

No software licence has been assigned to this repository at this stage. Dataset and model licences are separate from the repository licence and must be checked before redistribution or deployment.
