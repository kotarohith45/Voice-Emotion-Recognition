# 🎙️ Voice Emotion Recognition

A machine learning-based audio emotion classifier that detects emotions from speech in real-time. Built with Python, this system extracts audio features using Librosa and classifies emotions using trained ML models.

## ✨ Features

- **Real-time emotion detection** from microphone input
- **Multiple ML models**: SVM, Random Forest, and Logistic Regression
- **Audio feature extraction**: MFCC, Chroma, and Mel Spectrogram (180 features)
- **Interactive GUI** with waveform visualization and confidence bars
- **CLI prediction tool** for batch processing and scripting
- **File upload support** for analyzing pre-recorded audio files

## 📊 Emotions Detected

| Emotion  | Emoji |
|----------|-------|
| Happy    | 😊    |
| Sad      | 😢    |
| Angry    | 😠    |
| Neutral  | 😐    |

## 🗂️ Datasets

This project is trained on two publicly available datasets:

- **[RAVDESS](https://zenodo.org/record/1188976)** — Ryerson Audio-Visual Database of Emotional Speech and Song
- **[TESS](https://tspace.library.utoronto.ca/handle/1807/24487)** — Toronto Emotional Speech Set

## 🛠️ Tech Stack

- **Python 3.10+**
- **Librosa** — Audio feature extraction (MFCC, Chroma, Mel Spectrogram)
- **Scikit-learn** — Machine learning models (SVM, Random Forest, Logistic Regression)
- **SoundDevice** — Real-time audio recording
- **Matplotlib** — Waveform and probability visualization
- **Tkinter** — Desktop GUI framework

## 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/Voice_Emotion_Recognition.git
   cd Voice_Emotion_Recognition
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**

   ```bash
   python test_prediction.py
   ```

## 🚀 Usage

### GUI Application

Launch the interactive desktop application:

```bash
python app.py
```

- Click **"Start Recording"** to record 4 seconds of audio
- Click **"Upload File"** to analyze a pre-recorded `.wav` file
- View the predicted emotion, confidence score, waveform, and probability distribution

### Command-Line Prediction

Predict emotion from an audio file:

```bash
python predict.py path/to/audio.wav
```

Record and predict from microphone:

```bash
python predict.py --record
```

### Train Models (Optional)

To retrain models on your own RAVDESS/TESS data:

1. Update the dataset paths in `train_model.py`:

   ```python
   RAVDESS_PATH = r"C:\path\to\RAVDESS"
   TESS_PATH = r"C:\path\to\TESS"
   ```

2. Run the training pipeline:

   ```bash
   python train_model.py
   ```

   This trains SVM, Random Forest, and Logistic Regression models, compares their accuracy, and saves the best one.

## 📁 Project Structure

```
Voice_Emotion_Recognition/
├── app.py                  # GUI application (tkinter)
├── train_model.py          # Training pipeline (SVM, RF, LR)
├── predict.py              # CLI prediction tool
├── feature_extraction.py   # Shared feature extraction module
├── test_prediction.py      # Automated test suite
├── requirements.txt        # Python dependencies
├── models/
│   ├── emotion_model.pkl   # Trained classifier
│   ├── scaler.pkl          # Feature scaler
│   └── label_encoder.pkl   # Label encoder
├── test.wav                # Sample test audio
└── README.md
```

## 🔬 Feature Extraction

The system extracts **180 audio features** from each audio sample:

| Feature             | Count | Description                                |
|---------------------|-------|--------------------------------------------|
| MFCC                | 40    | Mel-Frequency Cepstral Coefficients        |
| Chroma              | 12    | Pitch class distribution                   |
| Mel Spectrogram     | 128   | Frequency bands on the mel scale           |

Features are computed using **Librosa** with silence trimming applied before extraction.

## 🤖 Model Comparison

| Model               | Description                              |
|----------------------|------------------------------------------|
| SVM (SVC)            | Support Vector Classifier with RBF kernel|
| Random Forest        | Ensemble of 200 decision trees           |
| Logistic Regression  | Multinomial with L2 regularization       |

The training script automatically selects and saves the best-performing model.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- RAVDESS dataset by Livingstone & Russo (2018)
- TESS dataset by Dupuis & Pichora-Fuller (2010)
- Librosa library for audio analysis
