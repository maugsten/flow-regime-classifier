# Flow Regime Classifier

A machine learning pipeline for flow regime classification from high-speed video data.

The project uses a two-stage approach:

1. Extract visual features from video frames using a pretrained ResNet18.
2. Classify complete experiments using an LSTM operating on sequences of extracted features.

Current flow regimes:

- Annular
- Churn
- Dispersed
- Slug
- Stratified
- Wavy

---

# Project Overview

Each experiment consists of a sequence of images extracted from a high-speed video.

Example:

```text
Exp_8281/
├── frame_0001.jpg
├── frame_0002.jpg
├── ...
└── frame_0300.jpg
```

Every frame is processed by a pretrained ResNet18 feature extractor.

```text
Image
  ↓
ResNet18
  ↓
512-dimensional feature vector
```

The resulting sequence of feature vectors is passed to an LSTM classifier:

```text
300 frames
      ↓
300 × 512 feature matrix
      ↓
LSTM
      ↓
Flow regime prediction
```

---

# Repository Structure

```text
flow-regime-classifier/

├── configs/
│   ├── baseline.yaml
│   ├── local.yaml
│   └── local.example.yaml
│
├── data/
│   ├── annular/
│   ├── churn/
│   ├── dispersed/
│   ├── slug/
│   ├── stratified/
│   └── wavy/
│
├── outputs/
│   ├── features/
│   ├── models/
│   ├── splits.csv
│   └── confusion_matrix.png
│
├── scripts/
│   ├── extract_features.py
│   ├── split_data.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── src/
│   └── flowregime/
│       ├── feature_extractor.py
│       ├── dataset.py
│       ├── model.py
│       ├── training.py
│       └── evaluation.py
│
└── README.md
```

---

# Dataset Structure

The dataset is organized by flow regime and experiment.

```text
data/

├── slug/
│   ├── Exp_8060/
│   ├── Exp_8067/
│   └── ...
│
├── annular/
├── churn/
├── dispersed/
├── stratified/
└── wavy/
```

Each experiment contains approximately 300 video frames.

```text
Exp_8281/
├── frame_0001.jpg
├── frame_0002.jpg
├── ...
└── frame_0300.jpg
```

---

# Pipeline

## Step 1: Feature Extraction

Extract ResNet18 features from all images.

```bash
python scripts/extract_features.py
```

Output:

```text
outputs/features/
├── slug/
│   ├── Exp_8281/
│   │   ├── frame_0001.pt
│   │   ├── frame_0002.pt
│   │   └── ...
```

Each feature file contains a 512-dimensional representation of a frame.

---

## Step 2: Dataset Split

Generate train/validation/test splits.

```bash
python scripts/split_data.py
```

Current split:

```text
Train:       8 experiments per class
Validation: 2 experiments per class
Test:       2 experiments per class
```

Output:

```text
outputs/splits.csv
```

Example:

```csv
Class,Experiment,Split
slug,Exp_8281,train
slug,Exp_8262,val
slug,Exp_8060,test
```

Experiments are split as complete units to avoid data leakage.

---

## Step 3: Train Model

Train the LSTM classifier.

```bash
python scripts/train_model.py
```

Training pipeline:

```text
Feature sequence
      ↓
LSTM
      ↓
Fully connected layer
      ↓
6 flow regime classes
```

The model with the highest validation accuracy is automatically saved.

Output:

```text
outputs/models/best_model.pt
```

---

## Step 4: Evaluate Model

Evaluate the best model on the test set.

```bash
python scripts/evaluate_model.py
```

Metrics:

- Accuracy
- Balanced Accuracy
- Classification Report
- Confusion Matrix

Outputs:

```text
outputs/confusion_matrix.png
```

and terminal statistics.

---

# Current Model

Feature extractor:

```text
ResNet18
(pretrained on ImageNet)
```

Classifier:

```text
Input size:   512
Hidden size:  128
Layers:       2
Classes:      6
```

Architecture:

```text
Frame
      ↓
ResNet18
      ↓
512 features
      ↓
Sequence
      ↓
LSTM
      ↓
Classifier
      ↓
Flow Regime
```

---

# Design Philosophy

The project separates:

```text
scripts/
```

for executable workflows:

- Extract features
- Train model
- Evaluate model

and

```text
src/flowregime/
```

for reusable library code:

- Models
- Datasets
- Feature extraction
- Training utilities
- Evaluation utilities

This keeps workflows simple while maintaining reusable components.

---

# Example Results

Initial proof-of-concept using:

```text
72 experiments
6 classes
300 frames per experiment
```

produced:

```text
Validation Accuracy ≈ 75%
Test Accuracy ≈ 50%
```

which is substantially above random guessing:

```text
Random baseline = 16.7%
```

and demonstrates that the extracted image features contain information relevant for flow regime classification.

---

# Future Work

Potential extensions include:

- Larger dataset
- Data augmentation
- Hyperparameter optimization
- Cross-validation
- GRU-based sequence models
- Transformer-based sequence models
- Multi-camera fusion
- Physics-informed features
- Temporal attention mechanisms

As the dataset grows, future work will focus on improving generalization performance and understanding which visual features are most important for flow regime classification.