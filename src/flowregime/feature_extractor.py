from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)
import torchvision.transforms as transforms


def load_resnet18():

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    feature_extractor = nn.Sequential(
        *list(model.children())[:-1]
    )

    feature_extractor.eval()

    return feature_extractor


def get_transform(image_size=224):

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])


def extract_feature_batch(
    image_paths,
    model,
    transform,
    device,
):

    images = []

    for path in image_paths:

        img = (
            Image.open(path)
            .convert("RGB")
        )

        img = transform(img)

        images.append(img)

    images = torch.stack(images).to(device)

    with torch.inference_mode():

        features = model(images)

        features = torch.flatten(
            features,
            1
        )

    return features.cpu()


def save_features_for_class(
    class_dir,
    output_dir,
    model,
    transform,
    batch_size,
    device,
):

    class_name = class_dir.name

    output_class_dir = output_dir / class_name

    output_class_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_dirs = sorted(
        d for d in class_dir.iterdir()
        if d.is_dir()
    )

    print(
        f"{class_name}: "
        f"{len(experiment_dirs)} experiments"
    )

    for exp_dir in experiment_dirs:

        image_files = sorted(
            list(exp_dir.glob("*.png"))
            + list(exp_dir.glob("*.jpg"))
            + list(exp_dir.glob("*.jpeg"))
        )

        print(
            f"    {exp_dir.name}: "
            f"{len(image_files)} images"
        )

        output_exp_dir = (
            output_class_dir
            / exp_dir.name
        )

        output_exp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for start in range(
            0,
            len(image_files),
            batch_size,
        ):

            batch_files = image_files[
                start:start + batch_size
            ]

            features = extract_feature_batch(
                batch_files,
                model,
                transform,
                device,
            )

            for img_file, feat in zip(
                batch_files,
                features,
            ):

                save_path = (
                    output_exp_dir
                    / f"{img_file.stem}.pt"
                )

                torch.save(
                    feat,
                    save_path,
                )