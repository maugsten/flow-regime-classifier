import torch

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

    train_dataset = FlowRegimeDataset(
        features_dir="outputs/features",
        split_csv="outputs/splits.csv",
        split="train",
    )

    val_dataset = FlowRegimeDataset(
        features_dir="outputs/features",
        split_csv="outputs/splits.csv",
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = LSTMClassifier().to(
        device
    )

    criterion = create_loss()

    optimizer = create_optimizer(
        model
    )

    n_epochs = 20

    best_val_acc = 0.0

    for epoch in range(n_epochs):

        loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_acc = evaluate(
            model,
            val_loader,
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                "outputs/models/best_model.pt",
            )

        print(
            f"Epoch {epoch+1:3d} "
            f"Loss: {loss:.4f} "
            f"ValAcc: {100*val_acc:.1f}%"
        )


if __name__ == "__main__":
    main()