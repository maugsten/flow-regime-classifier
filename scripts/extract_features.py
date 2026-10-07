from pathlib import Path

import yaml
import torch

from flowregime.feature_extractor import (
    load_resnet18,
    get_transform,
    save_features_for_class,
)


def main():

    with open(
        "configs/baseline.yaml",
        "r",
    ) as f:

        cfg = yaml.safe_load(f)

    data_dir = Path(
        cfg["data_dir"]
    )

    output_dir = Path(
        cfg["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    model = (
        load_resnet18()
        .to(device)
    )

    transform = get_transform(
        cfg["image_size"]
    )

    class_dirs = sorted([
        d
        for d in data_dir.iterdir()
        if d.is_dir()
    ])

    for class_dir in class_dirs:

        save_features_for_class(
            class_dir=class_dir,
            output_dir=output_dir,
            model=model,
            transform=transform,
            batch_size=cfg["batch_size"],
            device=device,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()