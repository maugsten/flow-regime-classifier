# from pathlib import Path

# import pandas as pd
# import torch
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import Dataset


# # Map each flow-regime name to the integer label used by the model.
# CLASS_TO_IDX = {
#     "annular": 0,
#     "churn": 1,
#     "dispersed": 2,
#     "slug": 3,
#     "stratified": 4,
#     "wavy": 5,
# }


# class FlowRegimeDataset(Dataset):
#     """
#     Load sequences of cached image features for flow-regime experiments.

#     Each dataset item represents one complete experiment and returns:
#         - A sequence of 512-dimensional image features
#         - The integer label for the flow-regime class
#     """

#     def __init__(
#         self,
#         features_dir,
#         split_csv,
#         split,
#     ):

#         # Root directory containing the cached ResNet18 features.
#         self.features_dir = Path(features_dir)

#         # Load the experiment split table.
#         df = pd.read_csv(split_csv)

#         # Retain only experiments belonging to the requested split,
#         # such as "train", "val", or "test".
#         self.df = df[
#             df["Split"] == split
#         ].reset_index(drop=True)

#     def __len__(self):

#         # One dataset entry corresponds to one experiment.
#         return len(self.df)

#     def __getitem__(self, idx):

#         # Read the class and experiment name for this dataset entry.
#         row = self.df.iloc[idx]

#         class_name = row["Class"]
#         experiment = row["Experiment"]

#         # Build the path to the experiment's cached feature files.
#         exp_dir = (
#             self.features_dir
#             / class_name
#             / experiment
#         )

#         # Load feature files in sorted order to preserve the
#         # temporal order of the original image sequence.
#         feature_files = sorted(
#             exp_dir.glob("*.pt")
#         )

#         features = []

#         # Load the 512-dimensional feature vector for each image.
#         for feature_file in feature_files:
#             features.append(
#                 torch.load(
#                     feature_file,
#                     weights_only=False,
#                 )
#             )

#         # Combine the individual image features into one sequence
#         # with shape (sequence_length, feature_dimension).
#         features = torch.stack(features)

#         # Convert the class name into the integer label expected
#         # by the classification loss.
#         label = CLASS_TO_IDX[class_name]

#         return features, label


# def collate_fn(batch):
#     """
#     Combine experiments of different sequence lengths into one batch.

#     Shorter sequences are padded with zeros so that all sequences in
#     a batch have the same length.
#     """

#     # Extract the feature sequence from each experiment in the batch.
#     sequences = [
#         item[0]
#         for item in batch
#     ]

#     # Combine the integer class labels into one tensor.
#     labels = torch.tensor(
#         [item[1] for item in batch],
#         dtype=torch.long,
#     )

#     # Store each experiment's original sequence length so the LSTM
#     # can distinguish real time steps from added padding.
#     lengths = torch.tensor(
#         [len(sequence) for sequence in sequences]
#     )

#     # Pad shorter feature sequences with zeros to match the length
#     # of the longest sequence in the current batch.
#     #
#     # Resulting shape:
#     #     (batch_size, longest_sequence, feature_dimension)
#     padded = pad_sequence(
#         sequences,
#         batch_first=True,
#     )

#     return padded, labels, lengths

from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# Map flow-regime names to the integer labels used by the
# classification model.
CLASS_TO_IDX = {
    "annular": 0,
    "churn": 1,
    "dispersed": 2,
    "slug": 3,
    "stratified": 4,
    "wavy": 5,
}


def normalize_experiment_name(value):
    """
    Convert experiment values from splits.csv to the folder
    name format used in the feature directory.

    Examples:
        8075       -> Exp_8075
        8075.0     -> Exp_8075
        "Exp_8075" -> Exp_8075
    """

    value = str(value).strip()

    if value.lower().startswith("exp_"):

        number = value.split(
            "_",
            maxsplit=1,
        )[1]

        return f"Exp_{number}"

    try:

        number = int(
            float(value)
        )

        return f"Exp_{number}"

    except ValueError as error:

        raise ValueError(
            f"Could not interpret experiment value: "
            f"{value}"
        ) from error


class FlowRegimeDataset(Dataset):
    """
    Load cached ResNet18 feature sequences.

    Each dataset item represents one sequence variant from
    one experiment.

    Training:
        Every available variant is included as a separate
        training sequence.

        Example:
            Exp_8075/original
            Exp_8075/brightness_plus
            Exp_8075/gaussian_blur

    Validation and testing:
        Only the original sequence is included.

        Example:
            Exp_8123/original

    All augmented variants remain assigned to the same
    experiment split, preventing train-validation-test
    leakage.
    """

    def __init__(
        self,
        features_dir,
        split_csv,
        split,
    ):
        self.features_dir = Path(
            features_dir
        )

        self.split = (
            str(split)
            .strip()
            .lower()
        )

        valid_splits = {
            "train",
            "val",
            "test",
        }

        if self.split not in valid_splits:

            raise ValueError(
                f"Unknown split: {self.split}. "
                f"Expected one of "
                f"{sorted(valid_splits)}."
            )

        if not self.features_dir.is_dir():

            raise FileNotFoundError(
                f"Features directory not found: "
                f"{self.features_dir}"
            )

        split_csv = Path(
            split_csv
        )

        if not split_csv.is_file():

            raise FileNotFoundError(
                f"Split file not found: "
                f"{split_csv}"
            )

        split_df = pd.read_csv(
            split_csv
        )

        required_columns = {
            "Class",
            "Experiment",
            "Split",
        }

        missing_columns = (
            required_columns
            - set(split_df.columns)
        )

        if missing_columns:

            raise ValueError(
                "Split file is missing these columns: "
                f"{sorted(missing_columns)}"
            )

        # Normalize values from the CSV.
        split_df["Class"] = (
            split_df["Class"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        split_df["Split"] = (
            split_df["Split"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        split_df["Experiment"] = (
            split_df["Experiment"]
            .apply(normalize_experiment_name)
        )

        # Keep only experiments assigned to the requested
        # split.
        split_df = split_df[
            split_df["Split"] == self.split
        ].reset_index(
            drop=True
        )

        # Each item stored here represents one actual sequence
        # that can be returned by __getitem__.
        self.samples = []

        missing_experiments = []
        missing_originals = []
        empty_variants = []

        for row in split_df.itertuples(
            index=False
        ):

            class_name = row.Class
            experiment_name = row.Experiment

            if class_name not in CLASS_TO_IDX:

                raise ValueError(
                    f"Unknown class in split file: "
                    f"{class_name}"
                )

            experiment_dir = (
                self.features_dir
                / class_name
                / experiment_name
            )

            if not experiment_dir.is_dir():

                missing_experiments.append(
                    str(experiment_dir)
                )

                continue

            # Training uses every available augmentation
            # variant.
            if self.split == "train":

                variant_dirs = sorted(
                    [
                        path
                        for path in experiment_dir.iterdir()
                        if path.is_dir()
                    ],
                    key=lambda path: (
                        path.name != "original",
                        path.name,
                    ),
                )

            # Validation and test use only original features.
            else:

                original_dir = (
                    experiment_dir
                    / "original"
                )

                if not original_dir.is_dir():

                    missing_originals.append(
                        str(original_dir)
                    )

                    continue

                variant_dirs = [
                    original_dir
                ]

            for variant_dir in variant_dirs:

                feature_files = sorted(
                    variant_dir.glob("*.pt"),
                    key=lambda path: path.name,
                )

                if not feature_files:

                    empty_variants.append(
                        str(variant_dir)
                    )

                    continue

                self.samples.append(
                    {
                        "class_name": class_name,
                        "experiment": experiment_name,
                        "variant": variant_dir.name,
                        "feature_files": feature_files,
                    }
                )

        if missing_experiments:

            preview = "\n".join(
                missing_experiments[:10]
            )

            raise FileNotFoundError(
                f"{len(missing_experiments)} experiment "
                f"feature directories were not found.\n\n"
                f"First missing directories:\n"
                f"{preview}"
            )

        if missing_originals:

            preview = "\n".join(
                missing_originals[:10]
            )

            raise FileNotFoundError(
                f"{len(missing_originals)} experiments "
                f"do not contain an original feature "
                f"directory.\n\n"
                f"First missing directories:\n"
                f"{preview}"
            )

        if empty_variants:

            print(
                f"WARNING: Ignored "
                f"{len(empty_variants)} empty variant "
                f"directories."
            )

        if not self.samples:

            raise ValueError(
                f"No feature sequences were found for "
                f"split '{self.split}'."
            )

        print(
            f"{self.split}: "
            f"{len(split_df)} experiments, "
            f"{len(self.samples)} sequences"
        )

        samples_per_class = {}

        for sample in self.samples:

            class_name = sample[
                "class_name"
            ]

            samples_per_class[
                class_name
            ] = (
                samples_per_class.get(
                    class_name,
                    0,
                )
                + 1
            )

        for class_name in CLASS_TO_IDX:

            count = samples_per_class.get(
                class_name,
                0,
            )

            print(
                f"    {class_name:<12}: "
                f"{count} sequences"
            )

    def __len__(self):
        """
        Return the number of sequence variants in this
        dataset split.
        """

        return len(
            self.samples
        )

    def __getitem__(self, idx):
        """
        Load one complete sequence variant.

        Returns:
            features:
                Tensor with shape:
                (sequence_length, 512)

            label:
                Integer class label
        """

        sample = self.samples[
            idx
        ]

        feature_files = sample[
            "feature_files"
        ]

        features = []

        for feature_file in feature_files:

            feature = torch.load(
                feature_file,
                map_location="cpu",
                weights_only=False,
            )

            # Handle an accidentally retained leading batch
            # dimension, if features have shape (1, 512).
            if (
                feature.ndim == 2
                and feature.shape[0] == 1
            ):

                feature = feature.squeeze(
                    0
                )

            if feature.ndim != 1:

                raise ValueError(
                    f"Unexpected feature shape "
                    f"{tuple(feature.shape)} in "
                    f"{feature_file}. Expected a "
                    f"one-dimensional feature vector."
                )

            features.append(
                feature
            )

        features = torch.stack(
            features
        )

        class_name = sample[
            "class_name"
        ]

        label = CLASS_TO_IDX[
            class_name
        ]

        return features, label


def collate_fn(batch):
    """
    Combine feature sequences with different lengths into
    one padded batch.

    Shorter sequences are padded with zeros. The original
    sequence lengths are returned so that the LSTM can ignore
    padding.
    """

    sequences = [
        item[0]
        for item in batch
    ]

    labels = torch.tensor(
        [
            item[1]
            for item in batch
        ],
        dtype=torch.long,
    )

    lengths = torch.tensor(
        [
            len(sequence)
            for sequence in sequences
        ],
        dtype=torch.long,
    )

    padded = pad_sequence(
        sequences,
        batch_first=True,
    )

    return padded, labels, lengths