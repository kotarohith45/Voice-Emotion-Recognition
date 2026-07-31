"""
Automated Tests for Voice Emotion Recognition

Tests feature extraction, model loading, and end-to-end prediction pipeline.

Usage:
    python test_prediction.py
"""

import os
import sys
import numpy as np
import joblib

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

MODEL_DIR = "models"
TEST_AUDIO = "test.wav"
EXPECTED_FEATURE_LENGTH = 180
EXPECTED_EMOTIONS = {"angry", "happy", "neutral", "sad"}

passed = 0
failed = 0


def test(name, condition, detail=""):
    """Simple test assertion with pass/fail tracking."""
    global passed, failed
    if condition:
        print(f"  ✅ PASS: {name}")
        passed += 1
    else:
        print(f"  ❌ FAIL: {name}")
        if detail:
            print(f"          {detail}")
        failed += 1


# ============================================================================
# TEST SUITES
# ============================================================================


def test_feature_extraction():
    """Test the feature extraction module."""
    print("\n" + "=" * 50)
    print("  📐 Feature Extraction Tests")
    print("=" * 50)

    from feature_extraction import extract_features

    # Test 1: Feature extraction runs without error
    if not os.path.exists(TEST_AUDIO):
        print(f"  ⚠️  SKIP: Test audio file '{TEST_AUDIO}' not found")
        return

    try:
        features = extract_features(TEST_AUDIO)
        test("Feature extraction completes", True)
    except Exception as e:
        test("Feature extraction completes", False, str(e))
        return

    # Test 2: Correct output shape
    test(
        f"Feature vector length = {EXPECTED_FEATURE_LENGTH}",
        len(features) == EXPECTED_FEATURE_LENGTH,
        f"Got {len(features)}, expected {EXPECTED_FEATURE_LENGTH}"
    )

    # Test 3: Output type
    test(
        "Features are numpy array",
        isinstance(features, np.ndarray)
    )

    # Test 4: No NaN values
    test(
        "No NaN values in features",
        not np.any(np.isnan(features))
    )

    # Test 5: No Inf values
    test(
        "No Inf values in features",
        not np.any(np.isinf(features))
    )

    # Test 6: Features are 1D
    test(
        "Features are 1D array",
        features.ndim == 1
    )


def test_model_loading():
    """Test that all model files load correctly."""
    print("\n" + "=" * 50)
    print("  🤖 Model Loading Tests")
    print("=" * 50)

    # Test 1: Model file exists
    model_path = os.path.join(MODEL_DIR, "emotion_model.pkl")
    test("emotion_model.pkl exists", os.path.exists(model_path))

    # Test 2: Scaler file exists
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    test("scaler.pkl exists", os.path.exists(scaler_path))

    # Test 3: Label encoder file exists
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    test("label_encoder.pkl exists", os.path.exists(encoder_path))

    # Test 4: Model loads
    try:
        model = joblib.load(model_path)
        test("Model loads successfully", True)
    except Exception as e:
        test("Model loads successfully", False, str(e))
        return

    # Test 5: Scaler loads
    try:
        scaler = joblib.load(scaler_path)
        test("Scaler loads successfully", True)
    except Exception as e:
        test("Scaler loads successfully", False, str(e))

    # Test 6: Encoder loads
    try:
        encoder = joblib.load(encoder_path)
        test("Label encoder loads successfully", True)
    except Exception as e:
        test("Label encoder loads successfully", False, str(e))
        return

    # Test 7: Encoder has expected classes
    classes = set(encoder.classes_)
    test(
        f"Encoder classes = {EXPECTED_EMOTIONS}",
        classes == EXPECTED_EMOTIONS,
        f"Got {classes}"
    )

    # Test 8: Scaler expects correct number of features
    test(
        f"Scaler expects {EXPECTED_FEATURE_LENGTH} features",
        scaler.n_features_in_ == EXPECTED_FEATURE_LENGTH,
        f"Got {scaler.n_features_in_}"
    )

    # Test 9: Model has predict_proba
    test(
        "Model supports predict_proba",
        hasattr(model, 'predict_proba')
    )


def test_end_to_end():
    """Test the full prediction pipeline end-to-end."""
    print("\n" + "=" * 50)
    print("  🔄 End-to-End Prediction Tests")
    print("=" * 50)

    if not os.path.exists(TEST_AUDIO):
        print(f"  ⚠️  SKIP: Test audio file '{TEST_AUDIO}' not found")
        return

    from feature_extraction import extract_features

    try:
        # Step 1: Extract features
        features = extract_features(TEST_AUDIO)
        features = features.reshape(1, -1)
        test("Feature extraction for test file", True)

        # Step 2: Scale features
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        features_scaled = scaler.transform(features)
        test("Feature scaling", True)

        # Step 3: Predict
        model = joblib.load(os.path.join(MODEL_DIR, "emotion_model.pkl"))
        encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

        probs = model.predict_proba(features_scaled)
        test("Model prediction", True)

        # Step 4: Decode prediction
        pred_idx = np.argmax(probs)
        emotion = encoder.inverse_transform([pred_idx])[0]
        confidence = probs[0][pred_idx] * 100

        test(
            f"Predicted emotion: '{emotion}' is valid",
            emotion in EXPECTED_EMOTIONS,
            f"Got '{emotion}'"
        )

        test(
            f"Confidence ({confidence:.1f}%) is in valid range",
            0 <= confidence <= 100
        )

        # Step 5: Probabilities sum to ~100%
        prob_sum = np.sum(probs[0]) * 100
        test(
            f"Probabilities sum ≈ 100% (got {prob_sum:.1f}%)",
            abs(prob_sum - 100) < 1
        )

        print(f"\n  📊 Test prediction: {emotion.upper()} ({confidence:.1f}%)")

    except Exception as e:
        test("End-to-end pipeline", False, str(e))


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("\n" + "=" * 50)
    print("  🧪 Voice Emotion Recognition — Test Suite")
    print("=" * 50)

    test_feature_extraction()
    test_model_loading()
    test_end_to_end()

    # Summary
    total = passed + failed
    print("\n" + "=" * 50)
    print(f"  📊 Results: {passed}/{total} passed, {failed}/{total} failed")

    if failed == 0:
        print("  ✅ All tests passed!")
    else:
        print("  ⚠️  Some tests failed. See details above.")

    print("=" * 50 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
