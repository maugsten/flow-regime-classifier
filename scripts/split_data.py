from pathlib import Path
import random

import pandas as pd
import yaml


def main():

    # Load project configuration.
    with open(
        "configs/baseline.yaml",
        "r",
    ) as f:
        cfg = yaml.safe_load(f)

    # Read the data-splitting parameters from the configuration.
    random_seed = cfg["split"]["random_seed"]

    train_experiments = (
        cfg["split"]["train_experiments_per_class"]
    )
    val_experiments = (
        cfg["split"]["val_experiments_per_class"]
    )
    test_experiments = (
        cfg["split"]["test_experiments_per_class"]
    )

    # Calculate the expected number of experiments per class.
    expected_experiments = (
        train_experiments
        + val_experiments
        + test_experiments
    )

    # Read the feature and output paths from the configuration.
    features_dir = Path(
        cfg["outputs"]["features_dir"]
    )

    splits_file = Path(
        cfg["outputs"]["splits_file"]
    )

    # Create the parent output directory if it does not exist.
    splits_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Initialize the random generator for reproducible splits.
    random.seed(random_seed)

    # Store one row for each experiment and its assigned split.
    rows = []

    # Process classes in the order specified in baseline.yaml.
    class_names = cfg["data"]["classes"]

    for class_name in class_names:

        class_dir = (
            features_dir
            / class_name
        )

        # Ensure that the expected class directory exists.
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        # Find all experiment directories belonging to this class.
        experiments = sorted(
            d.name
            for d in class_dir.iterdir()
            if d.is_dir()
        )

        print(
            f"{class_name}: "
            f"{len(experiments)} experiments"
        )

        # Verify that the class contains the number of experiments
        # required by the configured train, validation, and test split.
        if len(experiments) != expected_experiments:

            raise ValueError(
                f"{class_name} has "
                f"{len(experiments)} experiments, "
                f"expected {expected_experiments}."
            )

        # Randomize experiment order before assigning the splits.
        random.shuffle(experiments)

        # Assign the configured number of experiments to training.
        train_exps = experiments[
            :train_experiments
        ]

        # Assign the next experiments to validation.
        val_exps = experiments[
            train_experiments:
            train_experiments + val_experiments
        ]

        # Assign the remaining experiments to testing.
        test_exps = experiments[
            train_experiments + val_experiments:
        ]

        # Record the training experiments.
        for experiment in train_exps:
            rows.append(
                {
                    "Class": class_name,
                    "Experiment": experiment,
                    "Split": "train",
                }
            )

        # Record the validation experiments.
        for experiment in val_exps:
            rows.append(
                {
                    "Class": class_name,
                    "Experiment": experiment,
                    "Split": "val",
                }
            )

        # Record the test experiments.
        for experiment in test_exps:
            rows.append(
                {
                    "Class": class_name,
                    "Experiment": experiment,
                    "Split": "test",
                }
            )

    # Convert the split assignments into a table.
    df = pd.DataFrame(rows)

    # Save the split table without a separate row-number column.
    df.to_csv(
        splits_file,
        index=False,
    )

    # Print the number of experiments in each class and split.
    print()
    print(
        df.groupby(
            ["Split", "Class"]
        ).size()
    )

    print(
        f"\nSaved to {splits_file}"
    )


if __name__ == "__main__":
    main()