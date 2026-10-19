# from pathlib import Path

# import torch
# import torch.nn as nn
# from PIL import Image
# from torchvision.models import (
#     resnet18,
#     ResNet18_Weights,
# )
# import torchvision.transforms as transforms


# def load_resnet18():

#     # Load the default pretrained ImageNet weights.
#     weights = ResNet18_Weights.DEFAULT

#     # Load a ResNet18 model with the pretrained weights.
#     model = resnet18(weights=weights)

#     # Remove the final classification layer so that ResNet18
#     # returns a 512-dimensional feature representation.
#     feature_extractor = nn.Sequential(
#         *list(model.children())[:-1]
#     )

#     # Set the model to evaluation mode because it is only
#     # used for feature extraction, not training.
#     feature_extractor.eval()

#     return feature_extractor


# def get_transform(image_size=224):

#     # Define the preprocessing applied to each image
#     # before it is passed through ResNet18.
#     return transforms.Compose([
#         transforms.Resize(
#             (image_size, image_size)
#         ),
#         transforms.Grayscale(
#             num_output_channels=3
#         ),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=0.5, # Hey, should we do this?
#             std=0.5,
#         ),
#     ])


# def extract_feature_batch(
#     image_paths,
#     model,
#     transform,
#     device,
# ):

#     # Store the preprocessed images in the current batch.
#     images = []

#     for path in image_paths:

#         # Open the image and ensure that it has three RGB channels.
#         img = (
#             Image.open(path)
#             .convert("RGB")
#         )

#         # Resize, convert, and normalize the image.
#         img = transform(img)

#         images.append(img)

#     # Combine the images into one batch and move it
#     # to the selected CPU or GPU device.
#     images = torch.stack(images).to(device)

#     # Gradients are not needed because the model is not being trained.
#     with torch.inference_mode():

#         # Extract features for all images in the batch.
#         features = model(images)

#         # Change the feature shape from
#         # (batch_size, 512, 1, 1) to (batch_size, 512).
#         features = torch.flatten(
#             features,
#             1,
#         )

#     # Move the features back to the CPU before saving them.
#     return features.cpu()


# def save_features_for_class(
#     class_dir,
#     output_dir,
#     model,
#     transform,
#     batch_size,
#     device,
# ):

#     # Use the directory name as the flow-regime class name.
#     class_name = class_dir.name

#     # Create an output directory for the class.
#     output_class_dir = (
#         output_dir
#         / class_name
#     )

#     output_class_dir.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     # Find all experiment directories belonging to the class.
#     experiment_dirs = sorted(
#         d
#         for d in class_dir.iterdir()
#         if d.is_dir()
#     )

#     print(
#         f"{class_name}: "
#         f"{len(experiment_dirs)} experiments"
#     )

#     # Process each experiment separately.
#     for exp_dir in experiment_dirs:

#         # Find and sort all supported image files in the experiment.
#         image_files = sorted(
#             list(exp_dir.glob("*.png"))
#             + list(exp_dir.glob("*.jpg"))
#             + list(exp_dir.glob("*.jpeg"))
#         )

#         print(
#             f"    {exp_dir.name}: "
#             f"{len(image_files)} images"
#         )

#         # Create a matching output directory for the experiment.
#         output_exp_dir = (
#             output_class_dir
#             / exp_dir.name
#         )

#         output_exp_dir.mkdir(
#             parents=True,
#             exist_ok=True,
#         )

#         # Process images in batches to avoid loading the
#         # entire experiment into memory at once.
#         for start in range(
#             0,
#             len(image_files),
#             batch_size,
#         ):

#             # Select the image files in the current batch.
#             batch_files = image_files[
#                 start:start + batch_size
#             ]

#             # Extract one feature vector for each image.
#             features = extract_feature_batch(
#                 batch_files,
#                 model,
#                 transform,
#                 device,
#             )

#             # Save each image feature as a separate .pt file.
#             for img_file, feat in zip(
#                 batch_files,
#                 features,
#             ):

#                 save_path = (
#                     output_exp_dir
#                     / f"{img_file.stem}.pt"
#                 )

#                 torch.save(
#                     feat,
#                     save_path,
#                 )

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)
import torchvision.transforms as transforms


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}


def load_resnet18():
    """
    Load a pretrained ResNet18 and remove its final
    classification layer.

    The resulting model returns one 512-dimensional
    feature vector for each input image.
    """

    # Load the pretrained ImageNet weights.
    weights = ResNet18_Weights.DEFAULT

    # Create ResNet18 using the pretrained weights.
    model = resnet18(
        weights=weights
    )

    # Remove the final fully connected classification layer.
    #
    # The remaining network returns features with shape:
    #
    #     (batch_size, 512, 1, 1)
    feature_extractor = nn.Sequential(
        *list(model.children())[:-1]
    )

    # ResNet18 is used only for feature extraction.
    feature_extractor.eval()

    return feature_extractor


def get_transform(image_size=224):
    """
    Create the image preprocessing pipeline used before
    passing images through ResNet18.
    """

    return transforms.Compose(
        [
            # Resize every image to the input size expected
            # by the current feature extraction pipeline.
            transforms.Resize(
                (image_size, image_size)
            ),

            # Convert the flow images to grayscale while
            # retaining three channels for ResNet18.
            transforms.Grayscale(
                num_output_channels=3
            ),

            # Convert the PIL image to a PyTorch tensor.
            transforms.ToTensor(),

            # Scale values from [0, 1] to approximately
            # [-1, 1].
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )


def find_image_files(folder):
    """
    Find supported image files directly inside one folder.

    The function supports uppercase and lowercase file
    extensions because the suffix is converted to lowercase
    before comparison.
    """

    image_files = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_IMAGE_EXTENSIONS
    ]

    return sorted(
        image_files,
        key=lambda path: path.name,
    )


def extract_feature_batch(
    image_paths,
    model,
    transform,
    device,
):
    """
    Extract one 512-dimensional ResNet18 feature vector
    for each image in a batch.
    """

    images = []

    for path in image_paths:

        # Use a context manager to ensure that the image file
        # is closed immediately after loading.
        with Image.open(path) as image:

            image = image.convert("RGB")
            image = transform(image)

        images.append(image)

    # Combine all images into one tensor:
    #
    #     (batch_size, channels, height, width)
    images = torch.stack(
        images
    ).to(device)

    # Gradients are unnecessary because ResNet18 is not
    # being trained.
    with torch.inference_mode():

        features = model(
            images
        )

        # Convert:
        #
        #     (batch_size, 512, 1, 1)
        #
        # into:
        #
        #     (batch_size, 512)
        features = torch.flatten(
            features,
            1,
        )

    # Store cached features on the CPU so that they can later
    # be loaded regardless of the available device.
    return features.cpu()


def save_features_for_variant(
    variant_dir,
    output_variant_dir,
    model,
    transform,
    batch_size,
    device,
):
    """
    Extract and save features for one image variant.

    Examples of variants:
        original
        brightness_plus
        brightness_minus
        contrast_plus
        contrast_minus
        gaussian_blur
    """

    image_files = find_image_files(
        variant_dir
    )

    if not image_files:

        print(
            f"        WARNING: No images in "
            f"{variant_dir}"
        )

        return 0

    output_variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Process images in batches to avoid loading an entire
    # sequence into memory at once.
    for start in range(
        0,
        len(image_files),
        batch_size,
    ):

        batch_files = image_files[
            start:start + batch_size
        ]

        features = extract_feature_batch(
            image_paths=batch_files,
            model=model,
            transform=transform,
            device=device,
        )

        # Save one feature tensor for each image.
        for image_file, feature in zip(
            batch_files,
            features,
        ):

            save_path = (
                output_variant_dir
                / f"{image_file.stem}.pt"
            )

            torch.save(
                feature,
                save_path,
            )

    return len(image_files)


def save_features_for_class(
    class_dir,
    output_dir,
    model,
    transform,
    batch_size,
    device,
):
    """
    Extract features for every experiment and every image
    variant belonging to one flow-regime class.

    Expected input structure:

        data_augmented/
        └── class_name/
            └── Exp_XXXX/
                ├── original/
                ├── brightness_plus/
                ├── brightness_minus/
                ├── contrast_plus/
                ├── contrast_minus/
                └── gaussian_blur/

    Matching output structure:

        features_augmented/
        └── class_name/
            └── Exp_XXXX/
                ├── original/
                ├── brightness_plus/
                ├── brightness_minus/
                ├── contrast_plus/
                ├── contrast_minus/
                └── gaussian_blur/
    """

    class_name = class_dir.name

    output_class_dir = (
        output_dir
        / class_name
    )

    output_class_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Find experiment folders directly inside the class
    # folder. Variant folders are not interpreted as
    # experiments.
    experiment_dirs = sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_dir()
        ],
        key=lambda path: path.name,
    )

    print(
        f"{class_name}: "
        f"{len(experiment_dirs)} experiments"
    )

    total_images = 0
    total_variants = 0

    for experiment_dir in experiment_dirs:

        output_experiment_dir = (
            output_class_dir
            / experiment_dir.name
        )

        output_experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Each directory inside the experiment is one image
        # sequence variant.
        variant_dirs = sorted(
            [
                path
                for path in experiment_dir.iterdir()
                if path.is_dir()
            ],
            key=lambda path: (
                path.name != "original",
                path.name,
            ),
        )

        if not variant_dirs:

            print(
                f"    WARNING: No variant folders in "
                f"{experiment_dir}"
            )

            continue

        print(
            f"    {experiment_dir.name}: "
            f"{len(variant_dirs)} variants"
        )

        for variant_dir in variant_dirs:

            output_variant_dir = (
                output_experiment_dir
                / variant_dir.name
            )

            image_count = (
                save_features_for_variant(
                    variant_dir=variant_dir,
                    output_variant_dir=output_variant_dir,
                    model=model,
                    transform=transform,
                    batch_size=batch_size,
                    device=device,
                )
            )

            print(
                f"        {variant_dir.name}: "
                f"{image_count} images"
            )

            if image_count > 0:
                total_variants += 1
                total_images += image_count

    print(
        f"    Saved {total_images} features "
        f"from {total_variants} variants"
    )