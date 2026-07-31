"""
Feature Extraction Module for Voice Emotion Recognition

Extracts audio features (MFCC, Chroma, Mel Spectrogram) from audio files
for use in speech emotion classification.

Features extracted:
    - 40 MFCCs (Mel-Frequency Cepstral Coefficients)
    - 12 Chroma features
    - 128 Mel spectrogram bands
    Total: 180 features per audio sample
"""

import numpy as np
import librosa


def extract_features(file_path, sr=None, n_mfcc=40):
    """
    Extract audio features from a given audio file.

    Parameters
    ----------
    file_path : str
        Path to the audio file (.wav format recommended).
    sr : int or None
        Target sample rate. If None, uses the file's native sample rate.
    n_mfcc : int
        Number of MFCC coefficients to extract (default: 40).

    Returns
    -------
    numpy.ndarray
        1D array of shape (180,) containing concatenated features:
        [MFCC(40) | Chroma(12) | Mel(128)]
    """
    # Load audio file
    audio, sample_rate = librosa.load(file_path, sr=sr)

    # Trim silence from the beginning and end
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Ensure minimum audio length (pad if too short)
    if len(audio) < sample_rate:
        audio = np.pad(audio, (0, sample_rate - len(audio)), mode='constant')

    # Extract MFCC features (40 coefficients)
    mfcc = np.mean(
        librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc
        ).T,
        axis=0
    )

    # Extract Chroma features (12 pitch classes)
    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=audio,
            sr=sample_rate
        ).T,
        axis=0
    )

    # Extract Mel Spectrogram features (128 bands)
    mel = np.mean(
        librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate
        ).T,
        axis=0
    )

    # Concatenate all features into a single vector
    return np.hstack([mfcc, chroma, mel])
