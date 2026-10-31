import argparse
import random
import shutil
import sys
from pathlib import Path

from PIL import Image


DEFAULT_DATASET_ID = "Layered-Labs/ACNE04"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare the ACNE04 dataset or train the DermaVision detector."
    )
    parser.add_argument(
        "--prepare-acne04",
        action="store_true",
        help="Download ACNE04 and convert its lesion boxes to YOLO format.",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Hugging Face dataset identifier used with --prepare-acne04.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/acne04_yolo"),
        help="Local output directory for the prepared dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory when preparing the dataset.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to a YOLO dataset configuration file.",
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


def _open_dataset_image(value):
    """Convert a Hugging Face image value into an RGB PIL image."""
    from io import BytesIO

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")

    raise ValueError("The dataset row does not contain a readable image.")


def _get_yolo_boxes(box_data, image_width, image_height):
    """Convert parallel pixel-coordinate box arrays to normalized YOLO rows."""
    if not isinstance(box_data, dict):
        return []

    required = ("xmin", "ymin", "xmax", "ymax")
    if any(key not in box_data or box_data[key] is None for key in required):
        return []

    coordinate_arrays = [box_data[key] for key in required]
    lengths = {len(values) for values in coordinate_arrays}
    if len(lengths) != 1:
        raise ValueError("Lesion bounding-box coordinate arrays have different lengths.")

    yolo_rows = []
    for x_min, y_min, x_max, y_max in zip(*coordinate_arrays):
        x_min = max(0.0, min(float(x_min), float(image_width)))
        x_max = max(0.0, min(float(x_max), float(image_width)))
        y_min = max(0.0, min(float(y_min), float(image_height)))
        y_max = max(0.0, min(float(y_max), float(image_height)))

        if x_max <= x_min or y_max <= y_min:
            continue

        x_center = ((x_min + x_max) / 2.0) / image_width
        y_center = ((y_min + y_max) / 2.0) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        yolo_rows.append(
            f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return yolo_rows


def prepare_acne04(dataset_id: str, output_dir: Path, overwrite: bool = False) -> None:
    """Download ACNE04 and create train/validation/test YOLO folders.

    The dataset's official fold-0 train/test membership is preserved.
    Ten percent of the official training partition is reserved for validation.
    The official test partition is not used during training.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for data preparation. "
            "Install it with: pip install datasets"
        ) from exc

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already contains files. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    dataset = load_dataset(dataset_id, split="train")

    required_columns = {"image", "file_name", "lesion_boxes", "split"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(
            "The selected dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    records = []
    seen_names = set()

    for index, row in enumerate(dataset):
        official_split = str(row["split"]).strip().lower()
        if official_split not in {"train", "test"}:
            raise ValueError(f"Unexpected dataset split value: {official_split}")

        image = _open_dataset_image(row["image"])
        image_width, image_height = image.size
        if image_width < 1 or image_height < 1:
            raise ValueError(f"Image {index} has invalid dimensions.")

        original_name = str(row.get("file_name") or f"image_{index:04d}.jpg")
        image_name = Path(original_name).stem.replace(" ", "_") + ".jpg"

        if image_name in seen_names:
            raise ValueError(f"Duplicate image filename found: {image_name}")
        seen_names.add(image_name)

        yolo_boxes = _get_yolo_boxes(
            row["lesion_boxes"],
            image_width,
            image_height,
        )
        records.append(
            {
                "row_index": index,
                "image_name": image_name,
                "official_split": official_split,
                "yolo_boxes": yolo_boxes,
            }
        )

    official_train = [i for i, row in enumerate(records) if row["official_split"] == "train"]
    official_test = [i for i, row in enumerate(records) if row["official_split"] == "test"]

    if not official_train or not official_test:
        raise ValueError("The dataset must contain both official train and test partitions.")

    shuffled_train = official_train.copy()
    random.Random(42).shuffle(shuffled_train)
    validation_count = max(1, round(len(shuffled_train) * 0.10))
    validation_indices = set(shuffled_train[:validation_count])
    train_indices = set(official_train) - validation_indices

    split_indices = {
        "train": train_indices,
        "val": validation_indices,
        "test": set(official_test),
    }

    for split_name, indices in split_indices.items():
        image_dir = output_dir / "images" / split_name
        label_dir = output_dir / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for record_index in sorted(indices):
            record = records[record_index]
            image_path = image_dir / record["image_name"]
            label_path = label_dir / f"{Path(record['image_name']).stem}.txt"

            source_image = _open_dataset_image(dataset[record["row_index"]]["image"])
            source_image.save(image_path, format="JPEG", quality=95)
            label_path.write_text(
                "\n".join(record["yolo_boxes"]),
                encoding="utf-8",
            )

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write the dataset configuration.") from exc

    config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "acne_lesion"},
    }
    config_path = output_dir / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    total_boxes = sum(len(record["yolo_boxes"]) for record in records)
    print("Dataset preparation completed.")
    print(f"Dataset rows processed: {len(records)}")
    print(f"Training images: {len(train_indices)}")
    print(f"Validation images: {len(validation_indices)}")
    print(f"Held-out test images: {len(official_test)}")
    print(f"Valid lesion boxes written: {total_boxes}")
    print(f"YOLO configuration: {config_path.resolve()}")
    print("The dataset has been written locally and is excluded from Git by .gitignore.")


def validate_dataset_config(config_path: Path) -> None:
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {config_path}")

    import yaml

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
            "The dataset configuration must define class names using 'names' or 'nc'."
        )


def train_model(args) -> None:
    if args.data is None:
        raise ValueError("Provide --data with a YOLO dataset configuration.")

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
        print("Set DERMAVISION_MODEL_PATH to this file when running the application.")
        print("The held-out test partition has not been used for training or validation.")
    else:
        print(
            "Training finished, but the expected best.pt file was not found. "
            "Check the training output directory."
        )


def main():
    args = parse_arguments()

    if args.prepare_acne04:
        prepare_acne04(args.dataset_id, args.output_dir, args.overwrite)
        return

    train_model(args)


if __name__ == "__main__":
    main()
