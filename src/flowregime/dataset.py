from pathlib import Path

import torch
from torch.utils.data import Dataset
import pandas as pd


CLASS_TO_IDX = {
    "annular": 0,
    "churn": 1,
    "dispersed": 2,
    "slug": 3,
    "stratified": 4,
    "wavy": 5,
}


class FlowRegimeDataset(Dataset):

    def __init__(
        self,
        features_dir,
        split_csv,
        split,
    ):

        self.features_dir = Path(features_dir)

        df = pd.read_csv(split_csv)

        self.df = df[
            df["Split"] == split
        ].reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        class_name = row["Class"]
        experiment = row["Experiment"]

        exp_dir = (
            self.features_dir
            / class_name
            / experiment
        )

        feature_files = sorted(
            exp_dir.glob("*.pt")
        )

        features = []

        for f in feature_files:
            features.append(
                torch.load(
                    f,
                    weights_only=False,
                )
            )

        features = torch.stack(features)

        label = CLASS_TO_IDX[class_name]

        return features, label
    

# Hvis videoene ikke har samme lengde:
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):

    sequences = [
        item[0]
        for item in batch
    ]

    labels = torch.tensor(
        [item[1] for item in batch],
        dtype=torch.long,
    )

    lengths = torch.tensor(
        [len(seq) for seq in sequences]
    )

    padded = pad_sequence(
        sequences,
        batch_first=True,
    )

    return padded, labels, lengths
