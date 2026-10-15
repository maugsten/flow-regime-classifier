from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# Map each flow-regime name to the integer label used by the model.
CLASS_TO_IDX = {
    "annular": 0,
    "churn": 1,
    "dispersed": 2,
    "slug": 3,
    "stratified": 4,
    "wavy": 5,
}


class FlowRegimeDataset(Dataset):
    """
    Load sequences of cached image features for flow-regime experiments.

    Each dataset item represents one complete experiment and returns:
        - A sequence of 512-dimensional image features
        - The integer label for the flow-regime class
    """

    def __init__(
        self,
        features_dir,
        split_csv,
        split,
    ):

        # Root directory containing the cached ResNet18 features.
        self.features_dir = Path(features_dir)

        # Load the experiment split table.
        df = pd.read_csv(split_csv)

        # Retain only experiments belonging to the requested split,
        # such as "train", "val", or "test".
        self.df = df[
            df["Split"] == split
        ].reset_index(drop=True)

    def __len__(self):

        # One dataset entry corresponds to one experiment.
        return len(self.df)

    def __getitem__(self, idx):

        # Read the class and experiment name for this dataset entry.
        row = self.df.iloc[idx]

        class_name = row["Class"]
        experiment = row["Experiment"]

        # Build the path to the experiment's cached feature files.
        exp_dir = (
            self.features_dir
            / class_name
            / experiment
        )

        # Load feature files in sorted order to preserve the
        # temporal order of the original image sequence.
        feature_files = sorted(
            exp_dir.glob("*.pt")
        )

        features = []

        # Load the 512-dimensional feature vector for each image.
        for feature_file in feature_files:
            features.append(
                torch.load(
                    feature_file,
                    weights_only=False,
                )
            )

        # Combine the individual image features into one sequence
        # with shape (sequence_length, feature_dimension).
        features = torch.stack(features)

        # Convert the class name into the integer label expected
        # by the classification loss.
        label = CLASS_TO_IDX[class_name]

        return features, label


def collate_fn(batch):
    """
    Combine experiments of different sequence lengths into one batch.

    Shorter sequences are padded with zeros so that all sequences in
    a batch have the same length.
    """

    # Extract the feature sequence from each experiment in the batch.
    sequences = [
        item[0]
        for item in batch
    ]

    # Combine the integer class labels into one tensor.
    labels = torch.tensor(
        [item[1] for item in batch],
        dtype=torch.long,
    )

    # Store each experiment's original sequence length so the LSTM
    # can distinguish real time steps from added padding.
    lengths = torch.tensor(
        [len(sequence) for sequence in sequences]
    )

    # Pad shorter feature sequences with zeros to match the length
    # of the longest sequence in the current batch.
    #
    # Resulting shape:
    #     (batch_size, longest_sequence, feature_dimension)
    padded = pad_sequence(
        sequences,
        batch_first=True,
    )

    return padded, labels, lengths