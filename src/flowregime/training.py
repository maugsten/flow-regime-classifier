import torch
import torch.nn as nn


def create_loss():

    return nn.CrossEntropyLoss()


def create_optimizer(
    model,
    learning_rate=1e-3,
):

    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
):

    model.train()

    device = next(
        model.parameters()
    ).device

    running_loss = 0.0

    for (
        x_batch,
        y_batch,
        lengths,
    ) in train_loader:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        output = model(
            x_batch,
            lengths,
        )

        loss = criterion(
            output,
            y_batch,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return (
        running_loss
        / len(train_loader)
    )

def evaluate(
    model,
    dataloader,
):

    model.eval()

    device = next(
        model.parameters()
    ).device

    correct = 0
    total = 0

    with torch.no_grad():

        for (
            x_batch,
            y_batch,
            lengths,
        ) in dataloader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(
                x_batch,
                lengths,
            )

            predictions = torch.argmax(
                output,
                dim=1,
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

    return correct / total