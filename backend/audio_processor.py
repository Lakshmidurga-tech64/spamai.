"""
audio_processor.py
-------------------
Handles loading an audio file and turning it into a fixed-length
list of numbers ("features") that the machine learning model can
understand.

IMPORTANT: The exact same function (extract_features) is used
during BOTH training (training/train_model.py) and prediction
(backend/model.py). If these ever go out of sync, the model's
predictions will be meaningless, because it will be seeing
different kinds of numbers than it was trained on.
"""

import numpy as np
import librosa

# ---- Settings that must stay the same everywhere they are used ----
TARGET_SAMPLE_RATE = 16000   # We resample every file to 16kHz
N_MFCC = 13                  # Number of MFCC coefficients to extract


def load_audio(file_path: str):
    """
    Loads an audio file from disk and returns it as a mono
    (single-channel) numpy array, resampled to TARGET_SAMPLE_RATE.

    librosa.load() already does resampling + mono conversion for us,
    so this function stays simple.
    """
    audio, sample_rate = librosa.load(
        file_path,
        sr=TARGET_SAMPLE_RATE,   # force this sample rate
        mono=True                # force single channel
    )
    return audio, sample_rate


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Scales the audio so its loudest point is at 1.0.
    This helps the model treat a quiet recording and a loud
    recording of the same voice more similarly.
    """
    max_value = np.max(np.abs(audio))
    if max_value > 0:
        audio = audio / max_value
    return audio


def extract_features(file_path: str) -> np.ndarray:
    """
    Turns an audio file into a single fixed-length numpy array of
    numbers (a "feature vector"). This is what the ML model actually
    looks at - it never sees the raw audio.

    Features used (all beginner-friendly, well-known audio features):
      - MFCCs (Mel-Frequency Cepstral Coefficients): describe the
        general "shape" of the voice's frequency content.
      - Spectral centroid: where the "center of mass" of the sound
        frequencies is (brightness of the sound).
      - Spectral bandwidth: how spread out the frequencies are.
      - Zero crossing rate: how often the waveform crosses zero,
        related to noisiness/percussiveness.
      - Chroma: pitch-class energy, related to tonal content.

    Every feature is averaged over time (np.mean) and we also keep
    the standard deviation, so the final vector has a FIXED length
    no matter how long the input audio was.
    """
    audio, sample_rate = load_audio(file_path)
    audio = normalize_audio(audio)

    # Guard against completely silent or empty files
    if audio.size == 0:
        raise ValueError("Audio file appears to be empty or unreadable.")

    # --- MFCCs ---
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # --- Spectral centroid ---
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    centroid_mean = np.mean(spectral_centroid)
    centroid_std = np.std(spectral_centroid)

    # --- Spectral bandwidth ---
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    bandwidth_mean = np.mean(spectral_bandwidth)
    bandwidth_std = np.std(spectral_bandwidth)

    # --- Zero crossing rate ---
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # --- Chroma ---
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Combine everything into ONE fixed-length vector
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        [centroid_mean, centroid_std],
        [bandwidth_mean, bandwidth_std],
        [zcr_mean, zcr_std],
        chroma_mean, chroma_std,
    ])

    return features
