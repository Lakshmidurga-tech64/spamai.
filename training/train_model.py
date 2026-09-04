"""
train_model.py
---------------
Trains the voice classifier on your own audio dataset.

HOW TO USE:
1. Put real human voice recordings (.wav, .mp3, or .m4a) into:
     dataset/real/
2. Put synthetic/cloned voice recordings into:
     dataset/synthetic/
3. From the project root, run:
     python training/train_model.py
4. This creates backend/models/voice_model.pkl
5. Open backend/model.py and set DEMO_MODE = False to start using
   your real trained model.

This script uses the EXACT SAME feature extraction function as the
backend (audio_processor.extract_features), so training and
prediction always stay consistent.
"""

import os
import sys
import glob

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Make sure we can import audio_processor.py from backend/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.append(BACKEND_DIR)

from audio_processor import extract_features  # noqa: E402

REAL_DIR = os.path.join(PROJECT_ROOT, "dataset", "real")
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "dataset", "synthetic")
MODEL_OUTPUT_PATH = os.path.join(BACKEND_DIR, "models", "voice_model.pkl")

SUPPORTED_EXTENSIONS = ("*.wav", "*.mp3", "*.m4a")
RANDOM_STATE = 42  # keeps results reproducible


def find_audio_files(folder_path: str):
    """Returns a list of all supported audio file paths in a folder."""
    files = []
    for pattern in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder_path, pattern)))
    return sorted(files)


def build_dataset():
    """
    Walks through dataset/real and dataset/synthetic, extracts
    features from every file, and builds the X (features) and
    y (labels) arrays used for training.

    Label convention: real = 0, synthetic = 1
    """
    real_files = find_audio_files(REAL_DIR)
    synthetic_files = find_audio_files(SYNTHETIC_DIR)

    print(f"Found {len(real_files)} real audio file(s) in {REAL_DIR}")
    print(f"Found {len(synthetic_files)} synthetic audio file(s) in {SYNTHETIC_DIR}")

    if len(real_files) == 0 or len(synthetic_files) == 0:
        print("\nERROR: You need at least some files in BOTH dataset/real/ "
              "and dataset/synthetic/ before training.")
        print("Add audio files to those folders and run this script again.")
        sys.exit(1)

    features_list = []
    labels_list = []

    for file_path in real_files:
        try:
            features = extract_features(file_path)
            features_list.append(features)
            labels_list.append(0)  # real
        except Exception as error:
            print(f"  Skipping {file_path} (could not process): {error}")

    for file_path in synthetic_files:
        try:
            features = extract_features(file_path)
            features_list.append(features)
            labels_list.append(1)  # synthetic
        except Exception as error:
            print(f"  Skipping {file_path} (could not process): {error}")

    X = np.array(features_list)
    y = np.array(labels_list)
    return X, y


def train_and_evaluate(X, y):
    """
    Splits the data, trains a RandomForest classifier inside a
    scikit-learn Pipeline (scaling + model together), and prints
    evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if len(set(y)) > 1 else None,
    )

    # Pipeline bundles the scaler and classifier together so the
    # exact same preprocessing is automatically applied at prediction
    # time - no risk of forgetting a step.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        )),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n--- Evaluation on held-out test set ---")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")

    return pipeline


def main():
    print("Building dataset from audio files...")
    X, y = build_dataset()
    print(f"\nTotal usable samples: {len(y)}")

    print("\nTraining model...")
    pipeline = train_and_evaluate(X, y)

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")
    print("Next step: open backend/model.py and set DEMO_MODE = False")


if __name__ == "__main__":
    main()
