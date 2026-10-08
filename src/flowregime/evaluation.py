import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

IDX_TO_CLASS = {
    0: "annular",
    1: "churn",
    2: "dispersed",
    3: "slug",
    4: "stratified",
    5: "wavy",
}

def evaluate_model(
    model,
    dataloader,
):

    model.eval()

    device = next(
        model.parameters()
    ).device

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for (
            x_batch,
            y_batch,
            lengths,
        ) in dataloader:

            x_batch = x_batch.to(device)

            outputs = model(
                x_batch,
                lengths,
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_labels.extend(
                y_batch.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            all_labels,
            all_predictions,
        )
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        digits=3,
    )

    return (
        accuracy,
        balanced_accuracy,
        matrix,
        report,
    )

def plot_confusion_matrix(
    matrix,
    save_path,
):

    labels = [
        IDX_TO_CLASS[i]
        for i in range(len(IDX_TO_CLASS))
    ]

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()