"""
Extract image features using a pretrained ResNet18 model.

This script processes flow regime images, extracts feature vectors
using a pretrained ResNet18 backbone, and saves the extracted
features to disk for later classifier training.

Workflow:

    Images
        -> Image preprocessing
        -> ResNet18 feature extraction
        -> Save feature vectors
"""

from pathlib import Path

import yaml
import torch

from flowregime.feature_extractor import (
    load_resnet18,
    get_transform,
    save_features_for_class,
)


def main():

    # Load project configuration
    with open(
        "configs/baseline.yaml",
        "r",
    ) as f:
        cfg = yaml.safe_load(f)

    # Directory containing class folders with images
    data_dir = Path(
        cfg["data"]["root_dir"]
    )

    # Directory used to store extracted features
    # Verify that this matches the intended repository structure.
    output_dir = Path(
        cfg["outputs"]["features_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Select GPU if available, otherwise CPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # Load pretrained ResNet18 feature extractor
    model = (
        load_resnet18()
        .to(device)
    )

    # Create image preprocessing pipeline
    transform = get_transform(
        cfg["images"]["image_size"]
    )

    # Locate all flow-regime class directories
    class_dirs = sorted([
        d
        for d in data_dir.iterdir()
        if d.is_dir()
    ])

    # Extract and save features for each class
    for class_dir in class_dirs:

        save_features_for_class(
            class_dir=class_dir,
            output_dir=output_dir,
            model=model,
            transform=transform,
            batch_size=cfg["features"]["batch_size"],
            device=device,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()