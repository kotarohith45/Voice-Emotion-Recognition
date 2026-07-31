# 🎙️ Voice Emotion Recognition

> A Machine Learning-based audio emotion classifier that detects human emotions from speech using MFCC, Chroma, and Mel Spectrogram features.

---

## 📌 About The Project

This project builds a **real-time speech emotion recognition system** that can classify spoken audio into four emotional categories. It uses classical machine learning models trained on two standard speech emotion datasets.

### What It Does

- Records audio from your **microphone** (4 seconds)
- Extracts **180 audio features** from the speech signal
- Classifies the emotion using a **trained SVM model**
- Displays the result with **confidence scores** and **visualizations**

### Emotions Detected

| Emotion | Emoji | Description |
|---------|-------|-------------|
| Happy   | 😊    | Joyful, excited speech |
| Sad     | 😢    | Low energy, sorrowful tone |
| Angry   | 😠    | Loud, aggressive speech |
| Neutral | 😐    | Calm, flat tone |

---

## 🧠 How It Works

```
Audio Input → Feature Extraction → Scaling → ML Model → Emotion Output
```

### Feature Extraction Pipeline

The system extracts **180 numerical features** from each audio sample:

| Feature Type | Count | What It Captures |
|-------------|-------|------------------|
| **MFCC** (Mel-Frequency Cepstral Coefficients) | 40 | Vocal tone and timbre |
| **Chroma** | 12 | Pitch class distribution |
| **Mel Spectrogram** | 128 | Frequency energy distribution |

All features are extracted using the **Librosa** library after trimming silence from the audio.

### Machine Learning Models

Three models are trained and compared:

| Model | Algorithm | Key Parameters |
|-------|-----------|----------------|
| **SVM** (Support Vector Classifier) | RBF kernel | C=10, gamma=scale |
| **Random Forest** | 200 decision trees | max_depth=None |
| **Logistic Regression** | Multinomial + L2 | max_iter=1000 |

The best-performing model is automatically saved and used for predictions.

---

## 📂 Datasets Used

| Dataset | Full Name | Samples | Source |
|---------|-----------|---------|--------|
| **RAVDESS** | Ryerson Audio-Visual Database of Emotional Speech and Song | 1,440 audio files | [Download](https://zenodo.org/record/1188976) |
| **TESS** | Toronto Emotional Speech Set | 2,800 audio files | [Download](https://tspace.library.utoronto.ca/handle/1807/24487) |

Both datasets contain `.wav` files with labeled emotions, spoken by multiple actors.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Programming language |
| **Librosa** | Audio feature extraction |
| **Scikit-learn** | Machine learning (SVM, RF, LR) |
| **SoundDevice** | Microphone recording |
| **SoundFile** | Audio file I/O |
| **Matplotlib** | Waveform and chart visualization |
| **Tkinter** | GUI framework |
| **Joblib** | Model serialization |
| **NumPy** | Numerical operations |

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- A working microphone (for real-time recording)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/kotarohith45/Voice-Emotion-Recognition.git
cd Voice-Emotion-Recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify everything works
python test_prediction.py
```

You should see: `21/21 passed — All tests passed!`

---

## 🚀 Usage

### Option 1: GUI Application (Recommended)

```bash
python app.py
```

This opens a desktop window where you can:
- Click **"Start Recording"** → Speak for 4 seconds → See the predicted emotion
- Click **"Upload File"** → Select a `.wav` file → See the prediction
- View the **waveform** and **probability distribution** chart

### Option 2: Command Line — Predict from a File

```bash
python predict.py test.wav
```

Output:
```
========================================
  😠 Predicted Emotion: ANGRY
  📊 Confidence: 77.8%
========================================

  All probabilities:
    angry      ███████████████░░░░░  77.8%
    happy      ██░░░░░░░░░░░░░░░░░░  14.7%
    sad        █░░░░░░░░░░░░░░░░░░░   5.0%
    neutral    ░░░░░░░░░░░░░░░░░░░░   2.5%
```

### Option 3: Command Line — Record and Predict

```bash
python predict.py --record
```

Records 4 seconds from your microphone and predicts the emotion.

---

## 🔧 Retraining the Model (Optional)

If you want to retrain on your own copy of RAVDESS and TESS:

```bash
# 1. Edit dataset paths in train_model.py (lines 33-34)
RAVDESS_PATH = r"C:\path\to\your\RAVDESS"
TESS_PATH = r"C:\path\to\your\TESS"

# 2. Run training
python train_model.py
```

The script will:
- Load and parse both datasets
- Extract features from all audio files
- Train SVM, Random Forest, and Logistic Regression
- Print accuracy comparison and classification reports
- Save the best model to `models/`

---

## 📁 Project Structure

```
Voice-Emotion-Recognition/
│
├── app.py                  # GUI application (Tkinter + Matplotlib)
├── predict.py              # Command-line prediction tool
├── train_model.py          # Training pipeline (SVM, RF, LR)
├── feature_extraction.py   # Shared feature extraction module
├── test_prediction.py      # Automated test suite (21 tests)
│
├── models/
│   ├── emotion_model.pkl   # Trained SVM classifier
│   ├── scaler.pkl          # StandardScaler for feature normalization
│   └── label_encoder.pkl   # Label encoder (emotion ↔ number mapping)
│
├── test.wav                # Sample audio file for testing
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## ✅ Testing

Run the automated test suite:

```bash
python test_prediction.py
```

This runs **21 tests** covering:
- Feature extraction output shape and validity
- Model file loading and compatibility
- End-to-end prediction pipeline

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **RAVDESS** — Livingstone & Russo (2018)
- **TESS** — Dupuis & Pichora-Fuller (2010)
- **Librosa** — McFee et al. (audio analysis library)
