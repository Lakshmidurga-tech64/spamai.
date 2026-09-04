"""
model.py
--------
Wraps the machine learning model.

Two modes:

1. REAL ML MODE
   Used automatically when a trained model file exists at
   backend/models/voice_model.pkl (created by training/train_model.py).

2. DEMO MODE
   Used automatically when no trained model file exists yet, OR when
   DEMO_MODE is manually forced to True below. In this mode, results
   are clearly labeled as simulated - never presented as real AI
   predictions.

This file also exposes DEMO_MODE so main.py can include it in the
API response, and so the frontend can show the demo-mode notice.
"""

import os
import random
import joblib
import numpy as np

from audio_processor import extract_features

# ---------------------------------------------------------------
# Set this to False once you have trained a real model with
# training/train_model.py. If True, demo mode is forced even if a
# trained model file exists.
# ---------------------------------------------------------------
DEMO_MODE = True

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "voice_model.pkl")

# This will hold the loaded scikit-learn pipeline once loaded.
_model = None


def _model_file_exists() -> bool:
    return os.path.isfile(MODEL_PATH)


def is_demo_mode() -> bool:
    """
    The app is in demo mode if DEMO_MODE is forced True, OR if there
    is simply no trained model file on disk yet.
    """
    return DEMO_MODE or not _model_file_exists()


def load_model():
    """
    Loads the trained scikit-learn pipeline from disk (once), and
    caches it in the _model global so we don't reload it on every
    request.
    """
    global _model
    if _model is None and _model_file_exists():
        _model = joblib.load(MODEL_PATH)
    return _model


def _demo_prediction() -> float:
    """
    Generates a random-but-plausible "synthetic probability" for
    demo mode, purely so the interface can be tested end-to-end
    before a real model exists. This is clearly NOT a real AI
    prediction, and the API response always says so.
    """
    return round(random.uniform(0.05, 0.95), 4)


def predict_synthetic_probability(file_path: str) -> float:
    """
    Given a path to an audio file, returns a probability (0.0-1.0)
    that the voice is synthetic/cloned.

    In demo mode, this is a random placeholder value.
    In real ML mode, this runs the actual trained pipeline.
    """
    if is_demo_mode():
        # We still try to extract features so file-validation issues
        # (corrupt audio, etc.) get caught even in demo mode.
        extract_features(file_path)
        return _demo_prediction()

    model = load_model()
    if model is None:
        # Safety net: if something went wrong loading the model,
        # fall back to demo behavior rather than crashing.
        extract_features(file_path)
        return _demo_prediction()

    features = extract_features(file_path)
    features = np.array(features).reshape(1, -1)  # model expects 2D input

    # predict_proba returns [[P(real), P(synthetic)]] for our pipeline
    probabilities = model.predict_proba(features)[0]
    synthetic_probability = float(probabilities[1])
    return synthetic_probability
