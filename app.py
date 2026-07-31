"""
Voice Emotion Recognition — GUI Application

A tkinter-based desktop application that records audio from the microphone
and predicts the speaker's emotion using a trained ML model.

Features:
    - Real-time audio recording with visual feedback
    - Emotion prediction with confidence percentage
    - Waveform visualization of the recorded audio
    - Probability distribution bar chart
    - Non-blocking UI with threaded recording

Usage:
    python app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import joblib
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from feature_extraction import extract_features

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = "models"
DURATION = 4            # Recording duration in seconds
SAMPLE_RATE = 22050     # Audio sample rate

EMOJI_MAP = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😮",
}

# Color scheme
COLORS = {
    "bg": "#1a1a2e",
    "surface": "#16213e",
    "primary": "#0f3460",
    "accent": "#e94560",
    "text": "#ffffff",
    "text_dim": "#a0a0b0",
    "success": "#00d4aa",
    "warning": "#ffc107",
    "bar_bg": "#2a2a4a",
}

EMOTION_COLORS = {
    "happy": "#FFD700",
    "sad": "#4A90D9",
    "angry": "#FF4444",
    "neutral": "#88CC88",
    "fearful": "#9B59B6",
    "disgust": "#27AE60",
    "surprised": "#F39C12",
}


# ============================================================================
# APPLICATION
# ============================================================================


class EmotionRecognizerApp:
    """Main GUI application for Voice Emotion Recognition."""

    def __init__(self, root):
        self.root = root
        self.root.title("Voice Emotion Recognition")
        self.root.geometry("750x700")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(True, True)
        self.root.minsize(650, 600)

        self.is_recording = False
        self.recording_data = None

        # Load model
        self._load_model()

        # Build the UI
        self._build_ui()

    def _load_model(self):
        """Load the trained model, scaler, and label encoder."""
        try:
            self.model = joblib.load(os.path.join(MODEL_DIR, "emotion_model.pkl"))
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
            self.encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False
            print("⚠️  Model files not found in 'models/' directory.")
            print("   Run train_model.py first, or place .pkl files in models/")

    def _build_ui(self):
        """Construct all UI elements."""
        # Title
        title_frame = tk.Frame(self.root, bg=COLORS["bg"])
        title_frame.pack(fill="x", pady=(20, 5))

        tk.Label(
            title_frame,
            text="🎙️ Voice Emotion Recognition",
            font=("Segoe UI", 22, "bold"),
            fg=COLORS["text"],
            bg=COLORS["bg"]
        ).pack()

        tk.Label(
            title_frame,
            text="Speak into your microphone to detect emotion",
            font=("Segoe UI", 10),
            fg=COLORS["text_dim"],
            bg=COLORS["bg"]
        ).pack(pady=(2, 0))

        # ---- Buttons Frame ----
        btn_frame = tk.Frame(self.root, bg=COLORS["bg"])
        btn_frame.pack(pady=15)

        self.record_btn = tk.Button(
            btn_frame,
            text="🎤  Start Recording",
            command=self._on_record,
            font=("Segoe UI", 13, "bold"),
            fg="#ffffff",
            bg=COLORS["accent"],
            activebackground="#c73450",
            activeforeground="#ffffff",
            relief="flat",
            padx=25,
            pady=10,
            cursor="hand2"
        )
        self.record_btn.pack(side="left", padx=8)

        self.upload_btn = tk.Button(
            btn_frame,
            text="📂  Upload File",
            command=self._on_upload,
            font=("Segoe UI", 11),
            fg="#ffffff",
            bg=COLORS["primary"],
            activebackground="#0a2540",
            activeforeground="#ffffff",
            relief="flat",
            padx=18,
            pady=8,
            cursor="hand2"
        )
        self.upload_btn.pack(side="left", padx=8)

        # ---- Status Label ----
        self.status_var = tk.StringVar(value="Ready — Press 'Start Recording' to begin")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            fg=COLORS["text_dim"],
            bg=COLORS["bg"]
        )
        self.status_label.pack(pady=(0, 10))

        # ---- Result Display ----
        result_frame = tk.Frame(self.root, bg=COLORS["surface"], padx=20, pady=15)
        result_frame.pack(fill="x", padx=30)

        self.emoji_label = tk.Label(
            result_frame,
            text="🎭",
            font=("Segoe UI Emoji", 48),
            bg=COLORS["surface"],
            fg=COLORS["text"]
        )
        self.emoji_label.pack()

        self.emotion_label = tk.Label(
            result_frame,
            text="Waiting for input...",
            font=("Segoe UI", 20, "bold"),
            bg=COLORS["surface"],
            fg=COLORS["text"]
        )
        self.emotion_label.pack()

        self.confidence_label = tk.Label(
            result_frame,
            text="",
            font=("Segoe UI", 12),
            bg=COLORS["surface"],
            fg=COLORS["success"]
        )
        self.confidence_label.pack(pady=(3, 0))

        # ---- Confidence Progress Bar ----
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor=COLORS["bar_bg"],
            background=COLORS["accent"],
            thickness=14,
        )

        self.progress = ttk.Progressbar(
            result_frame,
            style="Custom.Horizontal.TProgressbar",
            orient="horizontal",
            length=350,
            mode="determinate"
        )
        self.progress.pack(pady=(8, 5))

        # ---- Visualization Frame (Waveform + Probability Bars) ----
        viz_frame = tk.Frame(self.root, bg=COLORS["bg"])
        viz_frame.pack(fill="both", expand=True, padx=30, pady=10)

        self.fig = Figure(figsize=(7, 3), dpi=90, facecolor=COLORS["bg"])
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.18, wspace=0.35)

        self.ax_wave = self.fig.add_subplot(121)
        self.ax_bars = self.fig.add_subplot(122)

        self._style_axes(self.ax_wave, "Waveform")
        self._style_axes(self.ax_bars, "Probabilities")

        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _style_axes(self, ax, title):
        """Apply consistent dark styling to matplotlib axes."""
        ax.set_facecolor(COLORS["surface"])
        ax.set_title(title, color=COLORS["text_dim"], fontsize=10, pad=8)
        ax.tick_params(colors=COLORS["text_dim"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(COLORS["bar_bg"])

    # ---- Event Handlers ----

    def _on_record(self):
        """Handle the Record button click."""
        if not self.model_loaded:
            self.status_var.set("❌ Model not loaded! Run train_model.py first.")
            return

        if self.is_recording:
            return

        self.is_recording = True
        self.record_btn.config(text="🔴  Recording...", bg="#aa3344", state="disabled")
        self.status_var.set(f"🎤 Recording for {DURATION} seconds... Speak now!")

        # Run recording in a separate thread to avoid freezing the GUI
        thread = threading.Thread(target=self._record_and_predict, daemon=True)
        thread.start()

    def _on_upload(self):
        """Handle the Upload button click."""
        if not self.model_loaded:
            self.status_var.set("❌ Model not loaded! Run train_model.py first.")
            return

        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

        if file_path:
            self.status_var.set(f"🔍 Analyzing: {os.path.basename(file_path)}...")
            thread = threading.Thread(
                target=self._predict_from_file,
                args=(file_path,),
                daemon=True
            )
            thread.start()

    # ---- Core Logic ----

    def _record_and_predict(self):
        """Record audio from microphone and run prediction."""
        try:
            # Record audio
            recording = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1
            )
            sd.wait()

            # Save the recording
            sf.write("recorded.wav", recording, SAMPLE_RATE)

            self.recording_data = recording.flatten()

            # Predict
            self._predict_from_file("recorded.wav")

        except Exception as e:
            self.root.after(0, self._show_error, str(e))
        finally:
            self.root.after(0, self._reset_record_button)

    def _predict_from_file(self, file_path):
        """Run prediction on an audio file and update the UI."""
        try:
            # Extract features
            features = extract_features(file_path)
            features = features.reshape(1, -1)
            features = self.scaler.transform(features)

            # Predict probabilities
            probs = self.model.predict_proba(features)
            pred_idx = np.argmax(probs)
            emotion = self.encoder.inverse_transform([pred_idx])[0]
            confidence = probs[0][pred_idx] * 100

            all_probs = dict(zip(self.encoder.classes_, probs[0] * 100))

            # Load audio for waveform display
            import librosa
            audio_data, _ = librosa.load(file_path, sr=SAMPLE_RATE)

            # Update UI (must be on main thread)
            self.root.after(0, self._update_result, emotion, confidence, all_probs, audio_data)

        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _update_result(self, emotion, confidence, all_probs, audio_data):
        """Update all UI elements with the prediction result."""
        emoji = EMOJI_MAP.get(emotion, "🎭")
        color = EMOTION_COLORS.get(emotion, COLORS["text"])

        # Update labels
        self.emoji_label.config(text=emoji)
        self.emotion_label.config(text=emotion.upper(), fg=color)
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")

        # Update progress bar
        self.progress["value"] = confidence

        # Update status
        self.status_var.set(f"✅ Prediction complete — Detected: {emotion.upper()}")

        # Update waveform plot
        self.ax_wave.clear()
        self._style_axes(self.ax_wave, "Waveform")
        time_axis = np.linspace(0, len(audio_data) / SAMPLE_RATE, num=len(audio_data))
        self.ax_wave.plot(time_axis, audio_data, color=COLORS["accent"], linewidth=0.5, alpha=0.8)
        self.ax_wave.set_xlabel("Time (s)", color=COLORS["text_dim"], fontsize=8)
        self.ax_wave.fill_between(time_axis, audio_data, alpha=0.15, color=COLORS["accent"])

        # Update probability bar chart
        self.ax_bars.clear()
        self._style_axes(self.ax_bars, "Probabilities")

        emotions_sorted = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        names = [e[0] for e in emotions_sorted]
        values = [e[1] for e in emotions_sorted]
        bar_colors = [EMOTION_COLORS.get(n, "#888888") for n in names]

        bars = self.ax_bars.barh(names, values, color=bar_colors, height=0.6, alpha=0.85)
        self.ax_bars.set_xlim(0, 100)
        self.ax_bars.set_xlabel("%", color=COLORS["text_dim"], fontsize=8)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            self.ax_bars.text(
                bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va='center', fontsize=8, color=COLORS["text_dim"]
            )

        self.ax_bars.invert_yaxis()
        self.canvas.draw()

    def _show_error(self, message):
        """Display an error message in the status bar."""
        self.status_var.set(f"❌ Error: {message}")
        self._reset_record_button()

    def _reset_record_button(self):
        """Reset the record button to its default state."""
        self.is_recording = False
        self.record_btn.config(
            text="🎤  Start Recording",
            bg=COLORS["accent"],
            state="normal"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Launch the Voice Emotion Recognition application."""
    root = tk.Tk()
    app = EmotionRecognizerApp(root)

    # Center the window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()