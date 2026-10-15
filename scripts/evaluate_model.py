from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from flowregime.dataset import (
    FlowRegimeDataset,
    collate_fn,
)

from flowregime.model import (
    LSTMClassifier,
)

from flowregime.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
)


def main():

    # Load the project configuration.
    with open(
        "configs/baseline.yaml",
        "r",
    ) as f:
        cfg = yaml.safe_load(f)

    # Select the fastest available computation device.
    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(
        f"Using device: {device}"
    )

    # Read input and output paths from the configuration.
    features_dir = Path(
        cfg["outputs"]["features_dir"]
    )

    splits_file = Path(
        cfg["outputs"]["splits_file"]
    )

    best_model_path = Path(
        cfg["outputs"]["best_model"]
    )

    confusion_matrix_path = Path(
        cfg["outputs"]["confusion_matrix"]
    )

    # Create the output directory for the confusion matrix.
    confusion_matrix_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Check that the trained model exists before evaluation.
    if not best_model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: "
            f"{best_model_path}"
        )

    # Read class names from the integer-to-name mapping in baseline.yaml.
    # Sorting ensures that the labels follow the order 0, 1, 2, ...
    class_labels = [
        class_name
        for _, class_name in sorted(
            cfg["labels"].items(),
            key=lambda item: int(item[0]),
        )
    ]

    # Create the test dataset from experiments assigned
    # to the test split.
    test_dataset = FlowRegimeDataset(
        features_dir=features_dir,
        split_csv=splits_file,
        split="test",
    )

    # Group test experiments into batches.
    # Test data are not shuffled because no training occurs.
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=cfg["dataset"]["shuffle_test"],
        num_workers=cfg["dataset"]["num_workers"],
        collate_fn=collate_fn,
    )

    print(
        f"Test experiments: {len(test_dataset)}"
    )

    # Recreate the same LSTM architecture used during training.
    model = LSTMClassifier(
        input_size=cfg["model"]["input_dim"],
        hidden_size=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_classes=cfg["model"]["output_dim"],
    ).to(device)

    # Load the parameters from the model with the best
    # validation accuracy.
    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
            weights_only=True,
        )
    )

    # Evaluate the trained model on the test dataset.
    (
        accuracy,
        balanced_accuracy,
        matrix,
        report,
    ) = evaluate_model(
        model,
        test_loader,
        class_labels,
    )

    # Print the test results.
    print()
    print("=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    print(
        f"Accuracy: "
        f"{100 * accuracy:.1f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{100 * balanced_accuracy:.1f}%"
    )

    print()
    print("Confusion Matrix")
    print(matrix)

    print()
    print("Classification Report")
    print(report)

    # Save a visual representation of the confusion matrix.
    plot_confusion_matrix(
        matrix,
        class_labels,
        confusion_matrix_path,
    )

    print(
        f"\nConfusion matrix saved to "
        f"{confusion_matrix_path}"
    )


if __name__ == "__main__":
    main()