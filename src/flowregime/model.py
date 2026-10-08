import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
)


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size=512,
        hidden_size=128,
        num_layers=2,
        num_classes=6,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(
        self,
        x,
        lengths,
    ):

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(
            packed
        )

        last_hidden = hidden[-1]

        output = self.classifier(
            last_hidden
        )

        return output