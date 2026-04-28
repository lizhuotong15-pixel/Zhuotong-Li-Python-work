from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


DEFAULT_IMAGE_DIR = Path(r"C:\Users\LZT\Downloads\brutalism_dataset_200\images")
DEFAULT_LABELS = [
    "a photo of a brutalist building",
    "a photo of a modern building",
    "a photo of a classical building",
    "a photo of an interior space",
    "a photo of an urban streetscape",
]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot CLIP image classification over a folder of images."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Folder containing the images to classify.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="Candidate text labels used by CLIP.",
    )
    parser.add_argument(
        "--model-name",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images processed per batch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("clip_predictions.csv"),
        help="CSV file to save the predictions.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many top labels to keep for each image.",
    )
    return parser.parse_args()


def iter_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def classify_images(
    image_paths: list[Path],
    labels: list[str],
    model_name: str,
    batch_size: int,
    top_k: int,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    rows: list[dict[str, object]] = []

    for batch_paths in batched(image_paths, batch_size):
        batch_images: list[Image.Image] = []
        valid_paths: list[Path] = []

        for path in batch_paths:
            try:
                batch_images.append(load_rgb_image(path))
                valid_paths.append(path)
            except (UnidentifiedImageError, OSError) as exc:
                rows.append(
                    {
                        "image_name": path.name,
                        "image_path": str(path),
                        "predicted_label": "error",
                        "predicted_score": None,
                        "top_predictions": f"Failed to read image: {exc}",
                    }
                )

        if not batch_images:
            continue

        inputs = processor(
            text=labels,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu()

        for path, prob_vector in zip(valid_paths, probs):
            ranked = sorted(
                zip(labels, prob_vector.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )
            top_predictions = ranked[:top_k]
            rows.append(
                {
                    "image_name": path.name,
                    "image_path": str(path),
                    "predicted_label": top_predictions[0][0],
                    "predicted_score": round(top_predictions[0][1], 6),
                    "top_predictions": " | ".join(
                        f"{label}: {score:.4f}" for label, score in top_predictions
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    image_paths = iter_images(args.image_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in: {args.image_dir}")

    top_k = max(1, min(args.top_k, len(args.labels)))
    results = classify_images(
        image_paths=image_paths,
        labels=args.labels,
        model_name=args.model_name,
        batch_size=max(1, args.batch_size),
        top_k=top_k,
    )
    results.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Processed {len(results)} images.")
    print(f"Saved results to: {args.output.resolve()}")
    print("Candidate labels:")
    for label in args.labels:
        print(f"- {label}")


if __name__ == "__main__":
    main()
