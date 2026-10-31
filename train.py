import argparse
from pathlib import Path
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the DermaVision acne-feature object detector."
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to a dataset configuration file in YOLO format.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Initial model weights or model configuration.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default=None, help="Examples: cpu, 0, or 0,1.")
    parser.add_argument("--project", type=Path, default=Path("outputs"))
    parser.add_argument("--name", default="acne_detection")
    return parser.parse_args()


def validate_dataset_config(config_path: Path) -> None:
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install the dependencies in requirements.txt."
        ) from exc

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The dataset configuration must contain a YAML mapping.")

    missing = [key for key in ("train", "val") if key not in config]
    if missing:
        raise ValueError(
            "The dataset configuration is missing required entries: "
            + ", ".join(missing)
        )

    if "names" not in config and "nc" not in config:
        raise ValueError(
            "The dataset configuration must define class names using 'names' "
            "or the number of classes using 'nc'."
        )


def main():
    args = parse_arguments()

    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("Epochs, image size, and batch size must be positive.")

    validate_dataset_config(args.data)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Ultralytics is not installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)

    model = YOLO(args.model)

    training_options = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": True,
        "save": True,
        "plots": True,
    }
    if args.device is not None:
        training_options["device"] = args.device

    model.train(**training_options)

    best_weights = args.project / args.name / "weights" / "best.pt"
    if best_weights.is_file():
        print(f"Training completed. Best model weights: {best_weights.resolve()}")
        print(
            "Set DERMAVISION_MODEL_PATH to this file when running the application."
        )
    else:
        print(
            "Training finished, but the expected best.pt file was not found. "
            "Check the training output directory."
        )


if __name__ == "__main__":
    main()
