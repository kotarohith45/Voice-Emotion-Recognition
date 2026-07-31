"""
Training Pipeline for Voice Emotion Recognition

Trains SVM, Random Forest, and Logistic Regression models on RAVDESS and TESS
datasets. Extracts MFCC, Chroma, and Mel Spectrogram features, then saves the
best-performing model along with the scaler and label encoder.

Usage:
    1. Update RAVDESS_PATH and TESS_PATH to your local dataset directories.
    2. Run: python train_model.py

Datasets:
    - RAVDESS: https://zenodo.org/record/1188976
    - TESS: https://tspace.library.utoronto.ca/handle/1807/24487
"""

import os
import glob
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from feature_extraction import extract_features

# ============================================================================
# CONFIGURATION — Update these paths to match your dataset locations
# ============================================================================

RAVDESS_PATH = r"C:\Users\kotar\Downloads\RAVDESS"  # Update this path
TESS_PATH = r"C:\Users\kotar\Downloads\TESS"        # Update this path

# Emotions to classify
EMOTIONS = ["angry", "happy", "neutral", "sad"]

# RAVDESS emotion code mapping (from filename convention)
# Format: 03-01-XX-... where XX is the emotion code
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",   # calm → neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ============================================================================
# DATASET LOADING
# ============================================================================


def load_ravdess(path):
    """
    Load audio files and emotion labels from the RAVDESS dataset.

    RAVDESS filenames follow this pattern:
        03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
    """
    features, labels = [], []

    if not os.path.exists(path):
        print(f"⚠️  RAVDESS path not found: {path}")
        return features, labels

    audio_files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
    print(f"📂 Found {len(audio_files)} RAVDESS audio files")

    for i, file in enumerate(audio_files):
        try:
            filename = os.path.basename(file)
            parts = filename.split("-")

            if len(parts) < 3:
                continue

            emotion_code = parts[2]
            emotion = RAVDESS_EMOTION_MAP.get(emotion_code, None)

            # Skip emotions not in our target list
            if emotion not in EMOTIONS:
                continue

            feat = extract_features(file)
            features.append(feat)
            labels.append(emotion)

            if (i + 1) % 50 == 0:
                print(f"  ✅ Processed {i + 1}/{len(audio_files)} files")

        except Exception as e:
            print(f"  ❌ Error processing {file}: {e}")

    print(f"  📊 Loaded {len(features)} RAVDESS samples")
    return features, labels


def load_tess(path):
    """
    Load audio files and emotion labels from the TESS dataset.

    TESS filenames follow this pattern:
        {speaker}_{word}_{emotion}.wav
    """
    features, labels = [], []

    if not os.path.exists(path):
        print(f"⚠️  TESS path not found: {path}")
        return features, labels

    # TESS emotion folder name mapping
    tess_emotion_map = {
        "angry": "angry",
        "happy": "happy",
        "sad": "sad",
        "neutral": "neutral",
        "fear": "fearful",
        "disgust": "disgust",
        "ps": "surprised",     # pleasant surprise
    }

    audio_files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
    print(f"📂 Found {len(audio_files)} TESS audio files")

    for i, file in enumerate(audio_files):
        try:
            filename = os.path.basename(file).lower()
            name_no_ext = os.path.splitext(filename)[0]

            # TESS: emotion is the last part of filename
            parts = name_no_ext.split("_")
            emotion_key = parts[-1].strip()
            emotion = tess_emotion_map.get(emotion_key, emotion_key)

            # Skip emotions not in our target list
            if emotion not in EMOTIONS:
                continue

            feat = extract_features(file)
            features.append(feat)
            labels.append(emotion)

            if (i + 1) % 50 == 0:
                print(f"  ✅ Processed {i + 1}/{len(audio_files)} files")

        except Exception as e:
            print(f"  ❌ Error processing {file}: {e}")

    print(f"  📊 Loaded {len(features)} TESS samples")
    return features, labels


# ============================================================================
# TRAINING PIPELINE
# ============================================================================


def train_and_evaluate():
    """Train SVM, Random Forest, and Logistic Regression models."""

    print("=" * 60)
    print("  Voice Emotion Recognition — Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load datasets
    # ------------------------------------------------------------------
    print("\n📥 Step 1: Loading datasets...\n")

    ravdess_features, ravdess_labels = load_ravdess(RAVDESS_PATH)
    tess_features, tess_labels = load_tess(TESS_PATH)

    # Combine datasets
    all_features = ravdess_features + tess_features
    all_labels = ravdess_labels + tess_labels

    if len(all_features) == 0:
        print("\n❌ No data loaded! Please check your dataset paths.")
        print(f"   RAVDESS_PATH = {RAVDESS_PATH}")
        print(f"   TESS_PATH = {TESS_PATH}")
        return

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\n📊 Total samples: {len(X)}")
    for emotion in EMOTIONS:
        count = np.sum(y == emotion)
        print(f"   {emotion}: {count} samples")

    # ------------------------------------------------------------------
    # Step 2: Encode labels and scale features
    # ------------------------------------------------------------------
    print("\n⚙️  Step 2: Preprocessing...\n")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"   Classes: {list(label_encoder.classes_)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # Step 3: Train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")

    # ------------------------------------------------------------------
    # Step 4: Train models
    # ------------------------------------------------------------------
    print("\n🤖 Step 3: Training models...\n")

    models = {
        "SVM (SVC)": SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            multi_class='multinomial'
        )
    }

    results = {}
    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"   ✅ {name} — Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)\n")
        print(classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_
        ))
        print("-" * 50)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name

    # ------------------------------------------------------------------
    # Step 5: Results comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  📊 MODEL COMPARISON")
    print("=" * 60)
    print(f"\n  {'Model':<25} {'Accuracy':>10}")
    print(f"  {'-'*25} {'-'*10}")

    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " 🏆" if name == best_name else ""
        print(f"  {name:<25} {acc*100:>9.1f}%{marker}")

    print(f"\n  🏆 Best model: {best_name} ({best_accuracy*100:.1f}%)")

    # ------------------------------------------------------------------
    # Step 6: Save the best model
    # ------------------------------------------------------------------
    print("\n💾 Step 4: Saving best model...\n")

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, os.path.join("models", "emotion_model.pkl"))
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join("models", "label_encoder.pkl"))

    print("   ✅ Saved: models/emotion_model.pkl")
    print("   ✅ Saved: models/scaler.pkl")
    print("   ✅ Saved: models/label_encoder.pkl")

    # Save all models for comparison
    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        filepath = os.path.join("models", f"{safe_name}.pkl")
        joblib.dump(model, filepath)
        print(f"   ✅ Saved: {filepath}")

    print("\n" + "=" * 60)
    print("  ✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
