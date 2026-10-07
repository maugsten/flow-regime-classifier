from pathlib import Path
import random

import pandas as pd


RANDOM_SEED = 42

TRAIN_EXPERIMENTS = 8
VAL_EXPERIMENTS = 2
TEST_EXPERIMENTS = 2


def main():

    random.seed(RANDOM_SEED)

    features_dir = Path("outputs/features")

    rows = []

    class_dirs = sorted(
        d for d in features_dir.iterdir()
        if d.is_dir()
    )

    for class_dir in class_dirs:

        experiments = sorted(
            d.name
            for d in class_dir.iterdir()
            if d.is_dir()
        )

        print(
            f"{class_dir.name}: "
            f"{len(experiments)} experiments"
        )

        if len(experiments) != 12:

            raise ValueError(
                f"{class_dir.name} has "
                f"{len(experiments)} experiments, "
                f"expected 12."
            )

        random.shuffle(experiments)

        train_exps = experiments[:TRAIN_EXPERIMENTS]
        val_exps = experiments[
            TRAIN_EXPERIMENTS:
            TRAIN_EXPERIMENTS + VAL_EXPERIMENTS
        ]
        test_exps = experiments[
            TRAIN_EXPERIMENTS + VAL_EXPERIMENTS:
        ]

        for exp in train_exps:
            rows.append(
                {
                    "Class": class_dir.name,
                    "Experiment": exp,
                    "Split": "train",
                }
            )

        for exp in val_exps:
            rows.append(
                {
                    "Class": class_dir.name,
                    "Experiment": exp,
                    "Split": "val",
                }
            )

        for exp in test_exps:
            rows.append(
                {
                    "Class": class_dir.name,
                    "Experiment": exp,
                    "Split": "test",
                }
            )

    df = pd.DataFrame(rows)

    df.to_csv(
        "outputs/splits.csv",
        index=False,
    )

    print()
    print(df.groupby(
        ["Split", "Class"]
    ).size())

    print(
        "\nSaved to outputs/splits.csv"
    )


if __name__ == "__main__":
    main()