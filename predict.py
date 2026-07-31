"""
CLI Emotion Prediction Tool

Predict the emotion in a given audio file using the trained model.

Usage:
    python predict.py <audio_file.wav>
    python predict.py test.wav
    python predict.py --record        # Record from microphone and predict
"""

import sys
import os
import numpy as np
import joblib
from feature_extraction import extract_features

# Directory containing saved models
MODEL_DIR = "models"


def load_model():
    """Load the trained model, scaler, and label encoder."""
    model = joblib.load(os.path.join(MODEL_DIR, "emotion_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return model, scaler, encoder


def predict_emotion(file_path):
    """
    Predict the emotion from an audio file.

    Parameters
    ----------
    file_path : str
        Path to the audio file.

    Returns
    -------
    tuple
        (predicted_emotion, confidence_percentage, all_probabilities)
    """
    model, scaler, encoder = load_model()

    # Extract features
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    features = scaler.transform(features)

    # Predict
    probs = model.predict_proba(features)
    pred_idx = np.argmax(probs)
    emotion = encoder.inverse_transform([pred_idx])[0]
    confidence = probs[0][pred_idx] * 100

    return emotion, confidence, dict(zip(encoder.classes_, probs[0] * 100))


def record_and_predict(duration=4, sample_rate=22050):
    """Record audio from microphone and predict emotion."""
    import sounddevice as sd
    import soundfile as sf

    temp_file = "recorded.wav"

    print(f"🎤 Recording for {duration} seconds... Speak now!")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1
    )
    sd.wait()
    sf.write(temp_file, recording, sample_rate)
    print("✅ Recording complete!\n")

    return predict_emotion(temp_file)


def main():
    """Main entry point for CLI prediction."""
    emoji_map = {
        "happy": "😊", "sad": "😢", "angry": "😠", "neutral": "😐",
        "fearful": "😨", "disgust": "🤢", "surprised": "😮"
    }

    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file.wav>")
        print("       python predict.py --record")
        sys.exit(1)

    if sys.argv[1] == "--record":
        emotion, confidence, all_probs = record_and_predict()
    else:
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            sys.exit(1)

        print(f"🔍 Analyzing: {file_path}\n")
        emotion, confidence, all_probs = predict_emotion(file_path)

    # Display results
    emoji = emoji_map.get(emotion, "🎭")
    print("=" * 40)
    print(f"  {emoji} Predicted Emotion: {emotion.upper()}")
    print(f"  📊 Confidence: {confidence:.1f}%")
    print("=" * 40)

    print("\n  All probabilities:")
    for emo, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
        marker = " ◀" if emo == emotion else ""
        print(f"    {emo:<10} {bar} {prob:5.1f}%{marker}")


if __name__ == "__main__":
    main()
