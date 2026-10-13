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

    # Load the default pretrained ImageNet weights.
    weights = ResNet18_Weights.DEFAULT

    # Load a ResNet18 model with the pretrained weights.
    model = resnet18(weights=weights)

    # Remove the final classification layer so that ResNet18
    # returns a 512-dimensional feature representation.
    feature_extractor = nn.Sequential(
        *list(model.children())[:-1]
    )

    # Set the model to evaluation mode because it is only
    # used for feature extraction, not training.
    feature_extractor.eval()

    return feature_extractor


def get_transform(image_size=224):

    # Define the preprocessing applied to each image
    # before it is passed through ResNet18.
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=0.5, # Hey, should we do this?
            std=0.5,
        ),
    ])


def extract_feature_batch(
    image_paths,
    model,
    transform,
    device,
):

    # Store the preprocessed images in the current batch.
    images = []

    for path in image_paths:

        # Open the image and ensure that it has three RGB channels.
        img = (
            Image.open(path)
            .convert("RGB")
        )

        # Resize, convert, and normalize the image.
        img = transform(img)

        images.append(img)

    # Combine the images into one batch and move it
    # to the selected CPU or GPU device.
    images = torch.stack(images).to(device)

    # Gradients are not needed because the model is not being trained.
    with torch.inference_mode():

        # Extract features for all images in the batch.
        features = model(images)

        # Change the feature shape from
        # (batch_size, 512, 1, 1) to (batch_size, 512).
        features = torch.flatten(
            features,
            1,
        )

    # Move the features back to the CPU before saving them.
    return features.cpu()


def save_features_for_class(
    class_dir,
    output_dir,
    model,
    transform,
    batch_size,
    device,
):

    # Use the directory name as the flow-regime class name.
    class_name = class_dir.name

    # Create an output directory for the class.
    output_class_dir = (
        output_dir
        / class_name
    )

    output_class_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Find all experiment directories belonging to the class.
    experiment_dirs = sorted(
        d
        for d in class_dir.iterdir()
        if d.is_dir()
    )

    print(
        f"{class_name}: "
        f"{len(experiment_dirs)} experiments"
    )

    # Process each experiment separately.
    for exp_dir in experiment_dirs:

        # Find and sort all supported image files in the experiment.
        image_files = sorted(
            list(exp_dir.glob("*.png"))
            + list(exp_dir.glob("*.jpg"))
            + list(exp_dir.glob("*.jpeg"))
        )

        print(
            f"    {exp_dir.name}: "
            f"{len(image_files)} images"
        )

        # Create a matching output directory for the experiment.
        output_exp_dir = (
            output_class_dir
            / exp_dir.name
        )

        output_exp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Process images in batches to avoid loading the
        # entire experiment into memory at once.
        for start in range(
            0,
            len(image_files),
            batch_size,
        ):

            # Select the image files in the current batch.
            batch_files = image_files[
                start:start + batch_size
            ]

            # Extract one feature vector for each image.
            features = extract_feature_batch(
                batch_files,
                model,
                transform,
                device,
            )

            # Save each image feature as a separate .pt file.
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