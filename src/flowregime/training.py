import torch
import torch.nn as nn


def create_loss():

    # Cross-entropy loss compares the model's class scores
    # with the correct integer class labels.
    return nn.CrossEntropyLoss()


def create_optimizer(
    model,
    optimizer_name,
    learning_rate,
    weight_decay,
):

    # Convert the optimizer name to lowercase so values such
    # as "Adam" and "adam" are treated identically.
    optimizer_name = optimizer_name.lower()

    # Create the optimizer specified in baseline.yaml.
    if optimizer_name == "adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    # Stop execution if the configuration specifies an
    # optimizer that has not been implemented.
    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}"
    )


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
):

    # Enable training behavior in the model.
    model.train()

    # Detect which device currently holds the model parameters.
    device = next(
        model.parameters()
    ).device

    # Accumulate the loss across all batches.
    running_loss = 0.0

    for (
        x_batch,
        y_batch,
        lengths,
    ) in train_loader:

        # Move the feature sequences and labels to the same
        # device as the model.
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Clear gradients calculated for the previous batch.
        optimizer.zero_grad()

        # Pass the feature sequences through the LSTM classifier.
        output = model(
            x_batch,
            lengths,
        )

        # Calculate the classification loss.
        loss = criterion(
            output,
            y_batch,
        )

        # Calculate gradients for all trainable model parameters.
        loss.backward()

        # Update the model parameters using the optimizer.
        optimizer.step()

        # Add the current batch loss to the running total.
        running_loss += loss.item()

    # Return the average loss per batch.
    return (
        running_loss
        / len(train_loader)
    )


def evaluate(
    model,
    dataloader,
):

    # Enable evaluation behavior in the model.
    model.eval()

    # Detect which device currently holds the model parameters.
    device = next(
        model.parameters()
    ).device

    correct = 0
    total = 0

    # Disable gradient calculations during evaluation.
    with torch.no_grad():

        for (
            x_batch,
            y_batch,
            lengths,
        ) in dataloader:

            # Move the feature sequences and labels to the
            # same device as the model.
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Calculate one score for each flow-regime class.
            output = model(
                x_batch,
                lengths,
            )

            # Select the class with the highest model score.
            predictions = torch.argmax(
                output,
                dim=1,
            )

            # Count correctly classified experiments.
            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

    # Return classification accuracy.
    return correct / total