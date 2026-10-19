# # from pathlib import Path
# # import random

# # import pandas as pd
# # import yaml


# # def main():

# #     # Load project configuration.
# #     with open(
# #         "configs/baseline.yaml",
# #         "r",
# #     ) as f:
# #         cfg = yaml.safe_load(f)

# #     # Read the data-splitting parameters from the configuration.
# #     random_seed = cfg["split"]["random_seed"]

# #     train_experiments = (
# #         cfg["split"]["train_experiments_per_class"]
# #     )
# #     val_experiments = (
# #         cfg["split"]["val_experiments_per_class"]
# #     )
# #     test_experiments = (
# #         cfg["split"]["test_experiments_per_class"]
# #     )

# #     # Calculate the expected number of experiments per class.
# #     expected_experiments = (
# #         train_experiments
# #         + val_experiments
# #         + test_experiments
# #     )

# #     # Read the feature and output paths from the configuration.
# #     features_dir = Path(
# #         cfg["outputs"]["features_dir"]
# #     )

# #     splits_file = Path(
# #         cfg["outputs"]["splits_file"]
# #     )

# #     # Create the parent output directory if it does not exist.
# #     splits_file.parent.mkdir(
# #         parents=True,
# #         exist_ok=True,
# #     )

# #     # Initialize the random generator for reproducible splits.
# #     random.seed(random_seed)

# #     # Store one row for each experiment and its assigned split.
# #     rows = []

# #     # Process classes in the order specified in baseline.yaml.
# #     class_names = cfg["data"]["classes"]

# #     for class_name in class_names:

# #         class_dir = (
# #             features_dir
# #             / class_name
# #         )

# #         # Ensure that the expected class directory exists.
# #         if not class_dir.is_dir():
# #             raise FileNotFoundError(
# #                 f"Class directory not found: {class_dir}"
# #             )

# #         # Find all experiment directories belonging to this class.
# #         experiments = sorted(
# #             d.name
# #             for d in class_dir.iterdir()
# #             if d.is_dir()
# #         )

# #         print(
# #             f"{class_name}: "
# #             f"{len(experiments)} experiments"
# #         )

# #         # Verify that the class contains the number of experiments
# #         # required by the configured train, validation, and test split.
# #         if len(experiments) != expected_experiments:

# #             raise ValueError(
# #                 f"{class_name} has "
# #                 f"{len(experiments)} experiments, "
# #                 f"expected {expected_experiments}."
# #             )

# #         # Randomize experiment order before assigning the splits.
# #         random.shuffle(experiments)

# #         # Assign the configured number of experiments to training.
# #         train_exps = experiments[
# #             :train_experiments
# #         ]

# #         # Assign the next experiments to validation.
# #         val_exps = experiments[
# #             train_experiments:
# #             train_experiments + val_experiments
# #         ]

# #         # Assign the remaining experiments to testing.Plea
# #         test_exps = experiments[
# #             train_experiments + val_experiments:
# #         ]

# #         # Record the training experiments.
# #         for experiment in train_exps:
# #             rows.append(
# #                 {
# #                     "Class": class_name,
# #                     "Experiment": experiment,
# #                     "Split": "train",
# #                 }
# #             )

# #         # Record the validation experiments.
# #         for experiment in val_exps:
# #             rows.append(
# #                 {
# #                     "Class": class_name,
# #                     "Experiment": experiment,
# #                     "Split": "val",
# #                 }
# #             )

# #         # Record the test experiments.
# #         for experiment in test_exps:
# #             rows.append(
# #                 {
# #                     "Class": class_name,
# #                     "Experiment": experiment,
# #                     "Split": "test",
# #                 }
# #             )

# #     # Convert the split assignments into a table.
# #     df = pd.DataFrame(rows)

# #     # Save the split table without a separate row-number column.
# #     df.to_csv(
# #         splits_file,
# #         index=False,
# #     )

# #     # Print the number of experiments in each class and split.
# #     print()
# #     print(
# #         df.groupby(
# #             ["Split", "Class"]
# #         ).size()
# #     )

# #     print(
# #         f"\nSaved to {splits_file}"
# #     )


# # if __name__ == "__main__":
# #     main()

# from pathlib import Path
# import random

# import pandas as pd
# import yaml


# def main():

#     # Load project configuration.
#     with open(
#         "configs/baseline.yaml",
#         "r",
#     ) as f:
#         cfg = yaml.safe_load(f)

#     # Read split settings.
#     random_seed = cfg["split"]["random_seed"]

#     train_fraction = (
#         cfg["split"]["train_fraction"]
#     )

#     val_fraction = (
#         cfg["split"]["val_fraction"]
#     )

#     test_fraction = (
#         cfg["split"]["test_fraction"]
#     )

#     # Verify fractions sum to 1.
#     if abs(
#         train_fraction
#         + val_fraction
#         + test_fraction
#         - 1.0
#     ) > 1e-6:

#         raise ValueError(
#             "train_fraction + "
#             "val_fraction + "
#             "test_fraction "
#             "must equal 1.0"
#         )

#     # Read paths from configuration.
#     features_dir = Path(
#         cfg["outputs"]["features_dir"]
#     )

#     splits_file = Path(
#         cfg["outputs"]["splits_file"]
#     )

#     # Create output directory.
#     splits_file.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     # Initialize random generator.
#     random.seed(random_seed)

#     # Store split assignments.
#     rows = []

#     # Process classes in YAML order.
#     class_names = cfg["data"]["classes"]

#     for class_name in class_names:

#         class_dir = (
#             features_dir
#             / class_name
#         )

#         if not class_dir.is_dir():

#             raise FileNotFoundError(
#                 f"Class directory not found: "
#                 f"{class_dir}"
#             )

#         experiments = sorted(
#             d.name
#             for d in class_dir.iterdir()
#             if d.is_dir()
#         )

#         n_total = len(experiments)

#         print(
#             f"{class_name}: "
#             f"{n_total} experiments"
#         )

#         # Randomize experiment order.
#         random.shuffle(experiments)

#         # Determine split sizes.
#         n_train = round(
#             train_fraction * n_total
#         )

#         n_val = round(
#             val_fraction * n_total
#         )

#         n_test = (
#             n_total
#             - n_train
#             - n_val
#         )

#         # Split experiments.
#         train_exps = experiments[
#             :n_train
#         ]

#         val_exps = experiments[
#             n_train:
#             n_train + n_val
#         ]

#         test_exps = experiments[
#             n_train + n_val:
#         ]

#         print(
#             f"  train={len(train_exps)}, "
#             f"val={len(val_exps)}, "
#             f"test={len(test_exps)}"
#         )

#         # Record training experiments.
#         for experiment in train_exps:

#             rows.append(
#                 {
#                     "Class": class_name,
#                     "Experiment": experiment,
#                     "Split": "train",
#                 }
#             )

#         # Record validation experiments.
#         for experiment in val_exps:

#             rows.append(
#                 {
#                     "Class": class_name,
#                     "Experiment": experiment,
#                     "Split": "val",
#                 }
#             )

#         # Record test experiments.
#         for experiment in test_exps:

#             rows.append(
#                 {
#                     "Class": class_name,
#                     "Experiment": experiment,
#                     "Split": "test",
#                 }
#             )

#     # Convert to dataframe.
#     df = pd.DataFrame(rows)

#     # Save split table.
#     df.to_csv(
#         splits_file,
#         index=False,
#     )

#     print()
#     print(
#         "Experiments per split "
#         "and class:"
#     )

#     print(
#         df.groupby(
#             ["Split", "Class"]
#         ).size()
#     )

#     print()
#     print(
#         "Total experiments "
#         "per split:"
#     )

#     print(
#         df["Split"]
#         .value_counts()
#     )

#     print(
#         f"\nSaved to "
#         f"{splits_file}"
#     )


# if __name__ == "__main__":
#     main()

from pathlib import Path
import random

import pandas as pd
import yaml


# ==========================================================
# CONFIG
# ==========================================================

CONFIG_FILE = Path("configs/baseline.yaml")

# Splitten skal opprettes fra det faktiske datasettet,
# ikke fra en gammel eller delvis featuremappe.
#
# Forventet struktur:
#
# data/
# ├── annular/
# │   ├── Exp_1001/
# │   ├── Exp_1002/
# │   └── ...
# ├── churn/
# ├── dispersed/
# ├── slug/
# ├── stratified/
# └── wavy/
DATA_DIR = Path("data")


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def find_experiments(class_dir):
    """
    Find experiment directories directly inside one class
    directory.

    Only directories with names beginning with "Exp_" are
    included.
    """

    experiments = sorted(
        path.name
        for path in class_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("Exp_")
    )

    return experiments


def verify_experiment_contains_images(
    class_dir,
    experiment_name,
):
    """
    Verify that an experiment directory contains at least one
    supported image file.
    """

    experiment_dir = (
        class_dir
        / experiment_name
    )

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    image_files = [
        path
        for path in experiment_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    ]

    return len(image_files) > 0


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------

    if not CONFIG_FILE.is_file():

        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{CONFIG_FILE}"
        )

    with open(
        CONFIG_FILE,
        "r",
    ) as file:

        cfg = yaml.safe_load(
            file
        )

    # ------------------------------------------------------
    # Read split settings
    # ------------------------------------------------------

    random_seed = (
        cfg["split"]["random_seed"]
    )

    train_fraction = (
        cfg["split"]["train_fraction"]
    )

    val_fraction = (
        cfg["split"]["val_fraction"]
    )

    test_fraction = (
        cfg["split"]["test_fraction"]
    )

    fraction_sum = (
        train_fraction
        + val_fraction
        + test_fraction
    )

    if abs(fraction_sum - 1.0) > 1e-6:

        raise ValueError(
            "The split fractions must sum to 1.0.\n"
            f"Current sum: {fraction_sum}"
        )

    for name, fraction in [
        ("train_fraction", train_fraction),
        ("val_fraction", val_fraction),
        ("test_fraction", test_fraction),
    ]:

        if fraction < 0 or fraction > 1:

            raise ValueError(
                f"{name} must be between 0 and 1. "
                f"Current value: {fraction}"
            )

    # ------------------------------------------------------
    # Read class names and output path
    # ------------------------------------------------------

    class_names = (
        cfg["data"]["classes"]
    )

    splits_file = Path(
        cfg["outputs"]["splits_file"]
    )

    splits_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Validate dataset directory
    # ------------------------------------------------------

    if not DATA_DIR.is_dir():

        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{DATA_DIR}"
        )

    print(
        f"Creating split from: "
        f"{DATA_DIR.resolve()}"
    )

    print(
        f"Random seed: {random_seed}"
    )

    print(
        "Fractions: "
        f"train={train_fraction:.2f}, "
        f"val={val_fraction:.2f}, "
        f"test={test_fraction:.2f}"
    )

    # Use a local random generator so that the split is
    # reproducible without modifying Python's global random
    # state.
    rng = random.Random(
        random_seed
    )

    rows = []

    # ------------------------------------------------------
    # Process each class separately
    # ------------------------------------------------------

    for class_name in class_names:

        class_dir = (
            DATA_DIR
            / class_name
        )

        if not class_dir.is_dir():

            raise FileNotFoundError(
                f"Class directory not found: "
                f"{class_dir}"
            )

        experiments = find_experiments(
            class_dir
        )

        if not experiments:

            raise ValueError(
                f"No experiment directories found in: "
                f"{class_dir}"
            )

        # Ensure that every included experiment contains at
        # least one image. Empty experiments should not enter
        # splits.csv.
        valid_experiments = []
        empty_experiments = []

        for experiment in experiments:

            has_images = (
                verify_experiment_contains_images(
                    class_dir=class_dir,
                    experiment_name=experiment,
                )
            )

            if has_images:

                valid_experiments.append(
                    experiment
                )

            else:

                empty_experiments.append(
                    experiment
                )

        if empty_experiments:

            print(
                f"\nWARNING: Ignoring "
                f"{len(empty_experiments)} empty "
                f"experiments in {class_name}:"
            )

            for experiment in empty_experiments:

                print(
                    f"  {experiment}"
                )

        experiments = (
            valid_experiments
        )

        n_total = len(
            experiments
        )

        if n_total == 0:

            raise ValueError(
                f"No valid experiments with images found "
                f"for class: {class_name}"
            )

        # Randomize experiment order reproducibly.
        rng.shuffle(
            experiments
        )

        # Calculate per-class split sizes.
        #
        # The test set receives the remainder so that every
        # experiment is assigned exactly once.
        n_train = round(
            train_fraction
            * n_total
        )

        n_val = round(
            val_fraction
            * n_total
        )

        n_test = (
            n_total
            - n_train
            - n_val
        )

        if n_test < 0:

            raise ValueError(
                f"Invalid split sizes for {class_name}: "
                f"train={n_train}, "
                f"val={n_val}, "
                f"test={n_test}"
            )

        # Assign experiments.
        train_experiments = experiments[
            :n_train
        ]

        val_experiments = experiments[
            n_train:
            n_train + n_val
        ]

        test_experiments = experiments[
            n_train + n_val:
        ]

        print(
            f"\n{class_name}: "
            f"{n_total} experiments"
        )

        print(
            f"  train: {len(train_experiments)}"
        )

        print(
            f"  val:   {len(val_experiments)}"
        )

        print(
            f"  test:  {len(test_experiments)}"
        )

        # Save assignments.
        for split_name, split_experiments in [
            ("train", train_experiments),
            ("val", val_experiments),
            ("test", test_experiments),
        ]:

            for experiment in split_experiments:

                rows.append(
                    {
                        "Class": class_name,
                        "Experiment": experiment,
                        "Split": split_name,
                    }
                )

    # ------------------------------------------------------
    # Build and validate split dataframe
    # ------------------------------------------------------

    split_df = pd.DataFrame(
        rows
    )

    duplicate_mask = (
        split_df["Experiment"]
        .duplicated(
            keep=False
        )
    )

    if duplicate_mask.any():

        duplicates = split_df.loc[
            duplicate_mask,
            [
                "Class",
                "Experiment",
                "Split",
            ],
        ]

        raise ValueError(
            "Some experiment names appear more than once "
            "in the dataset:\n\n"
            f"{duplicates.to_string(index=False)}"
        )

    # Sort the output for readability. The random split
    # assignment remains unchanged.
    split_order = pd.CategoricalDtype(
        categories=[
            "train",
            "val",
            "test",
        ],
        ordered=True,
    )

    split_df["Split"] = (
        split_df["Split"]
        .astype(split_order)
    )

    split_df = split_df.sort_values(
        by=[
            "Split",
            "Class",
            "Experiment",
        ]
    ).reset_index(
        drop=True
    )

    # Convert back to normal strings before saving.
    split_df["Split"] = (
        split_df["Split"]
        .astype(str)
    )

    # ------------------------------------------------------
    # Save split file
    # ------------------------------------------------------

    split_df.to_csv(
        splits_file,
        index=False,
    )

    # ------------------------------------------------------
    # Final validation and summary
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)

    print(
        f"Total experiments: "
        f"{len(split_df)}"
    )

    print(
        "\nExperiments per class and split:"
    )

    summary = pd.crosstab(
        split_df["Class"],
        split_df["Split"],
    ).reindex(
        index=class_names,
        columns=[
            "train",
            "val",
            "test",
        ],
        fill_value=0,
    )

    summary["total"] = (
        summary.sum(
            axis=1
        )
    )

    print(
        summary
    )

    print(
        "\nTotal experiments per split:"
    )

    split_counts = (
        split_df["Split"]
        .value_counts()
        .reindex(
            [
                "train",
                "val",
                "test",
            ],
            fill_value=0,
        )
    )

    print(
        split_counts
    )

    print(
        f"\nSaved split to: "
        f"{splits_file.resolve()}"
    )


if __name__ == "__main__":
    main()