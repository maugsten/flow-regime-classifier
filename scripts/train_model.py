from pathlib import Path
import random

import numpy as np
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

from flowregime.training import (
    create_loss,
    create_optimizer,
    train_one_epoch,
    evaluate,
)


def main():

    # Load the project configuration.
    with open(
        "configs/baseline.yaml",
        "r",
    ) as f:
        cfg = yaml.safe_load(f)

    # Read and set the random seed for reproducible training.
    seed = cfg["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    # Create the model output directory if it does not exist.
    best_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create the training dataset using the experiments
    # assigned to the training split.
    train_dataset = FlowRegimeDataset(
        features_dir=features_dir,
        split_csv=splits_file,
        split="train",
    )

    # Create the validation dataset using the experiments
    # assigned to the validation split.
    val_dataset = FlowRegimeDataset(
        features_dir=features_dir,
        split_csv=splits_file,
        split="val",
    )

    # Create the training data loader.
    #
    # The data loader groups experiments into batches and shuffles
    # their order before each training epoch.
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=cfg["dataset"]["shuffle_train"],
        num_workers=cfg["dataset"]["num_workers"],
        collate_fn=collate_fn,
    )

    # Create the validation data loader.
    #
    # Validation data are not shuffled because they are only
    # used to evaluate the model.
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=cfg["dataset"]["shuffle_val"],
        num_workers=cfg["dataset"]["num_workers"],
        collate_fn=collate_fn,
    )

    # Select the fastest available computation device.
    #
    # CUDA is used for supported NVIDIA GPUs, while MPS is used
    # for supported Apple GPUs. The CPU is used as a fallback.
    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(
        f"Using device: {device}"
    )

    print(
        f"Training experiments: {len(train_dataset)}"
    )

    print(
        f"Validation experiments: {len(val_dataset)}"
    )

    # Create the LSTM classifier using the architecture
    # specified in baseline.yaml.
    model = LSTMClassifier(
        input_size=cfg["model"]["input_dim"],
        hidden_size=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_classes=cfg["model"]["output_dim"],
    ).to(device)

    # Create the classification loss function.
    criterion = create_loss()

    # Create the optimizer using the settings in baseline.yaml.
    optimizer = create_optimizer(
        model=model,
        optimizer_name=cfg["training"]["optimizer"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Read the number of training epochs from the configuration.
    n_epochs = cfg["training"]["epochs"]

    # Store the highest validation accuracy obtained so far.
    best_val_acc = 0.0

    # Train and validate the model for the configured
    # number of epochs.
    for epoch in range(n_epochs):

        # Train the model for one complete pass through
        # the training dataset.
        loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        # Evaluate the model on the validation dataset.
        val_acc = evaluate(
            model,
            val_loader,
        )

        # Save the model parameters whenever validation
        # accuracy improves.
        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                best_model_path,
            )

        # Print the results for the current epoch.
        print(
            f"Epoch {epoch + 1:3d}/{n_epochs} "
            f"Loss: {loss:.4f} "
            f"ValAcc: {100 * val_acc:.1f}% "
            f"BestValAcc: {100 * best_val_acc:.1f}%"
        )

    # Print the final training summary.
    print(
        f"\nBest validation accuracy: "
        f"{100 * best_val_acc:.1f}%"
    )

    print(
        f"Best model saved to {best_model_path}"
    )


if __name__ == "__main__":
    main()