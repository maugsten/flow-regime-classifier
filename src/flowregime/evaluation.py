from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model,
    dataloader,
    class_labels,
):

    # Enable evaluation behavior in the model.
    model.eval()

    # Identify the device containing the model parameters.
    device = next(
        model.parameters()
    ).device

    all_labels = []
    all_predictions = []

    # Disable gradient calculations during evaluation.
    with torch.no_grad():

        for (
            x_batch,
            y_batch,
            lengths,
        ) in dataloader:

            # Move the input sequences to the model's device.
            x_batch = x_batch.to(device)

            # Calculate one score for each flow-regime class.
            outputs = model(
                x_batch,
                lengths,
            )

            # Select the class with the highest score.
            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            # Store the true labels and model predictions on the CPU.
            all_labels.extend(
                y_batch.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # Use an explicit label order so all configured classes
    # appear in the results, even if one is never predicted.
    label_indices = list(
        range(len(class_labels))
    )

    # Calculate the fraction of correctly classified experiments.
    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    # Calculate accuracy while giving each class equal importance.
    balanced_accuracy = (
        balanced_accuracy_score(
            all_labels,
            all_predictions,
        )
    )

    # Count predictions for every true and predicted class pair.
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=label_indices,
    )

    # Calculate precision, recall, and F1-score for each class.
    report = classification_report(
        all_labels,
        all_predictions,
        labels=label_indices,
        target_names=class_labels,
        digits=3,
        zero_division=0,
    )

    return (
        accuracy,
        balanced_accuracy,
        matrix,
        report,
    )


def plot_confusion_matrix(
    matrix,
    class_labels,
    save_path,
):

    # Convert the output location to a Path object.
    save_path = Path(save_path)

    # Create the output directory if it does not exist.
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Plot the confusion matrix with the configured class names.
    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    # Save the figure and close it to release memory.
    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()