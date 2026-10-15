import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
)


class LSTMClassifier(nn.Module):
    """
    Classify a variable-length sequence of image features.

    The LSTM processes the sequence of ResNet18 feature vectors.
    The final hidden state is used to predict the flow regime.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
    ):

        super().__init__()

        # Process the sequence of ResNet18 feature vectors.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Convert the final LSTM hidden state into one score
        # for each flow-regime class.
        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(
        self,
        x,
        lengths,
    ):

        # Pack the padded sequences so the LSTM ignores padding
        # added to shorter experiments.
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Process the sequences and retrieve the hidden states.
        _, (hidden, _) = self.lstm(
            packed
        )

        # Select the final hidden state from the last LSTM layer.
        last_hidden = hidden[-1]

        # Produce one unnormalized score for each flow regime.
        output = self.classifier(
            last_hidden
        )

        return output