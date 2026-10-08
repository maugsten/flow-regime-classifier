import torch

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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    test_dataset = (
        FlowRegimeDataset(
            features_dir="outputs/features",
            split_csv="outputs/splits.csv",
            split="test",
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = (
        LSTMClassifier()
        .to(device)
    )

    model.load_state_dict(
        torch.load(
            "outputs/models/best_model.pt",
            map_location=device,
        )
    )

    (
        accuracy,
        balanced_accuracy,
        matrix,
        report,
    ) = evaluate_model(
        model,
        test_loader,
    )

    print()
    print("=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    print(
        f"Accuracy: "
        f"{100*accuracy:.1f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{100*balanced_accuracy:.1f}%"
    )

    print()
    print("Confusion Matrix")
    print(matrix)

    print()
    print("Classification Report")
    print(report)

    plot_confusion_matrix(
        matrix,
        "outputs/confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
