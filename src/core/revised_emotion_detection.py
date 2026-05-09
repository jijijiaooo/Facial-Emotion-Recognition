#!/usr/bin/env python3
"""
Revised Emotion Detection - Using 2026 Revised CNN Model
6 emotion classes: angry, disgust, fear, happy, neutral, sad
No 'surprise' emotion - simplified but accurate model
"""

import cv2
import numpy as np
import os
import importlib
import json
import csv
from pathlib import Path
import time
from datetime import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
tf_keras = getattr(tf, 'keras', None)

try:
    standalone_keras = importlib.import_module('keras')
except Exception:
    standalone_keras = None

# Prefer standalone Keras 3 when available (model may be serialized with keras.src.* paths).
if standalone_keras is not None and hasattr(standalone_keras, 'src'):
    keras = standalone_keras
elif tf_keras is not None:
    keras = tf_keras
elif standalone_keras is not None:
    keras = standalone_keras
else:
    raise ImportError("Keras is required. Install tensorflow or keras.")

# --- FocalLoss definition for custom_objects (EXACT from training) ---
try:
    register_keras_serializable = keras.saving.register_keras_serializable
except Exception:
    register_keras_serializable = keras.utils.register_keras_serializable

@register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance and hard examples
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    Formula: FL = -α(1-pt)^γ * log(pt)
    """
    def __init__(self, gamma=2.0, label_smoothing=0.0, reduction="sum_over_batch_size", name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, self.gamma)
        focal_loss = focal_weight * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'label_smoothing': self.label_smoothing
        })
        return config

# Configure GPU/MPS (Metal Performance Shaders for Apple Silicon)
print("=" * 60)
print("GPU Configuration:")
print("=" * 60)

# Check available devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Found {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("ℹ️ No GPU found, checking for Apple Silicon MPS...")
    # For Apple Silicon Macs, check MPS (Metal Performance Shaders)
    try:
        # Set memory growth for MPS if available
        if hasattr(tf.config, 'list_physical_devices'):
            mps_devices = tf.config.list_physical_devices('MPS')
            if mps_devices:
                print(f"✅ Found Apple Metal GPU (MPS): {len(mps_devices)} device(s)")
                print("   Using Metal Performance Shaders for acceleration")
            else:
                print("ℹ️ No MPS device found, will use CPU")
    except Exception as e:
        print(f"ℹ️ MPS check failed: {e}, using CPU")

# Print TensorFlow build info
if tf.test.is_built_with_cuda():
    print("✅ TensorFlow built with CUDA support")
elif hasattr(tf.test, 'is_built_with_rocm') and tf.test.is_built_with_rocm():
    print("✅ TensorFlow built with ROCm support")
else:
    print("ℹ️ TensorFlow built without CUDA (may support MPS or CPU only)")

print("=" * 60)


class RevisedEmotionDetector:
    """
    Revised Emotion Detector using 2026 CNN model
    6 emotion classes (no surprise): angry, disgust, fear, happy, neutral, sad
    """
    
    # 6 emotion classes (revised dataset)
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

    # Display aliases for user-visible output labels
    EMOTION_DISPLAY_ALIASES = {
        'fear': 'shocked'
    }
    
    # Color scheme for each emotion (BGR format)
    EMOTION_COLORS = {
        'angry': (0, 0, 255),      # Red
        'disgust': (0, 128, 0),    # Dark Green
        'fear': (128, 0, 128),     # Purple
        'shocked': (128, 0, 128),  # Purple (same as fear)
        'happy': (0, 255, 255),    # Yellow
        'neutral': (200, 200, 200), # Light Gray
        'sad': (255, 0, 0)         # Blue
    }
    
    def __init__(self, model_path=None, show_debug=False, single_face=True):
        """
        Initialize the Revised Emotion Detector
        
        Args:
            model_path: Path to the revised CNN model (.keras file)
            show_debug: Whether to show debug information
            single_face: If True, process only the largest/closest face (default: True)
        """
        self.show_debug = show_debug
        self.single_face = single_face
        
        # Load model
        if model_path is None:
            # Default to the revised model 20260220_030450 (updated)
            model_path = 'models/emotion_revised_cnn_20260220_030450.keras'
        
        self.model_path = model_path
        self.model = self._load_model()
        
        # Initialize Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError(f"❌ Failed to load Haar Cascade from {cascade_path}")
        
        print("✅ Haar Cascade face detector loaded successfully")
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # Statistics tracking
        self.total_predictions = 0
        self.emotion_counts = {
            self.get_display_label(label): 0 for label in self.EMOTION_LABELS
        }

        # Processing-time tracking
        self.session_start_time = time.time()
        self.session_perf_start_time = time.perf_counter()
        self.last_frame_perf_time = None
        self.total_prediction_time_sec = 0.0
        self.prediction_trial_count = 0
        self.prediction_times_sec = []
        self.frame_elapsed_time_sec = []
        self.frame_instant_fps = []
        self.frame_detected_faces = []
        
        print("\n" + "="*70)
        print("REVISED EMOTION DETECTION - 2026 CNN Model")
        print("="*70)
        print(f"Model: {os.path.basename(self.model_path)}")
        print(f"Classes: {', '.join(self.EMOTION_LABELS)}")
        print(f"Mode: {'Single face (closest)' if self.single_face else 'Multiple faces'}")
        print(f"Input shape: {self.model.input_shape}")
        print(f"Output shape: {self.model.output_shape}")
        print("="*70 + "\n")

    def get_display_label(self, emotion_label):
        """Map internal model labels to user-visible labels."""
        return self.EMOTION_DISPLAY_ALIASES.get(emotion_label, emotion_label)
    
    def _load_model(self):
        """Load the revised CNN model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"❌ Model file not found: {self.model_path}\n"
                f"   Please ensure the revised model is available."
            )

        print(f"\n📦 Loading revised emotion model...")
        print(f"   Path: {self.model_path}")
        print(f"   TensorFlow: {tf.__version__} | Keras: {keras.__version__}")

        try:
            model = keras.models.load_model(
                self.model_path,
                custom_objects={"FocalLoss": FocalLoss}
            )
            print(f"✅ Model loaded successfully")
            return model
        except Exception as e:
            error_text = str(e)
            if "keras.src.models.functional" in error_text:
                raise RuntimeError(
                    "❌ Failed to load model due to Keras version mismatch. "
                    "This model was serialized with newer standalone Keras (3.x), "
                    f"but current runtime is TensorFlow {tf.__version__} / Keras {keras.__version__}. "
                    "Use a Keras 3 compatible runtime (for example TensorFlow 2.16+ with keras>=3)."
                ) from None
            raise RuntimeError(f"❌ Failed to load model: {e}") from None
    
    def detect_faces(self, image, single_face=True):
        """
        Detect faces in an image using Haar Cascade
        
        Args:
            image: Input image (BGR format)
            single_face: If True, return only the largest/closest face
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If single_face mode, return only the largest face (closest to camera)
        if single_face and len(faces) > 0:
            # Calculate area for each face and get the largest
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            return [largest_face]
        
        return faces
    
    def preprocess_face(self, face_img):
        """
        Preprocess face image for the revised CNN model
        
        Args:
            face_img: Face image (BGR or grayscale)
            
        Returns:
            Preprocessed face array ready for model input
        """
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_img
        
        # Resize to model input size (112x112)
        resized_face = cv2.resize(gray_face, (112, 112))
        
        # Normalize to [0, 1]
        normalized_face = resized_face / 255.0
        
        # Expand dimensions to match model input: (1, 112, 112, 1)
        processed_face = np.expand_dims(normalized_face, axis=[0, -1])
        
        return processed_face
    
    def predict_emotion(self, face_img):
        """
        Predict emotion from a face image
        
        Args:
            face_img: Face image (BGR or grayscale)
            
        Returns:
            Tuple of (emotion_label, confidence)
        """
        # Preprocess face
        processed_face = self.preprocess_face(face_img)
        
        # Get prediction
        predict_start = time.perf_counter()
        predictions = self.model.predict(processed_face, verbose=0)
        predict_elapsed = time.perf_counter() - predict_start
        
        # Get emotion with highest confidence
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        emotion_label = self.EMOTION_LABELS[emotion_idx]
        display_label = self.get_display_label(emotion_label)
        
        # Update statistics
        self.total_predictions += 1
        self.prediction_trial_count += 1
        self.total_prediction_time_sec += float(predict_elapsed)
        self.prediction_times_sec.append(float(predict_elapsed))
        self.emotion_counts[display_label] += 1
        
        if self.show_debug:
            print(f"\n🔍 Prediction Details:")
            print(f"   Detected: {display_label} ({confidence*100:.1f}%)")
            print("   All probabilities:")
            for i, label in enumerate(self.EMOTION_LABELS):
                display_name = self.get_display_label(label)
                print(f"      {display_name:10s}: {predictions[0][i]*100:5.1f}%")
        
        return display_label, confidence

    def get_processing_speed_metrics(self):
        """Compute runtime processing-speed metrics for the current session."""
        trial_count = int(self.prediction_trial_count)
        times = np.array(self.prediction_times_sec, dtype=np.float64)
        mean_processing_time_sec = (
            self.total_prediction_time_sec / trial_count if trial_count > 0 else 0.0
        )
        mean_fps = (
            trial_count / self.total_prediction_time_sec if self.total_prediction_time_sec > 0 else 0.0
        )

        if trial_count > 0:
            fps_per_trial = 1.0 / times
            median_processing_time_sec = float(np.median(times))
            std_processing_time_sec = float(np.std(times, ddof=0))
            min_processing_time_sec = float(np.min(times))
            max_processing_time_sec = float(np.max(times))
            p95_processing_time_sec = float(np.percentile(times, 95))
            p99_processing_time_sec = float(np.percentile(times, 99))
            mean_instant_fps = float(np.mean(fps_per_trial))
            median_fps = float(np.median(fps_per_trial))
            std_fps = float(np.std(fps_per_trial, ddof=0))
            p05_fps = float(np.percentile(fps_per_trial, 5))
            p95_fps = float(np.percentile(fps_per_trial, 95))
        else:
            median_processing_time_sec = 0.0
            std_processing_time_sec = 0.0
            min_processing_time_sec = 0.0
            max_processing_time_sec = 0.0
            p95_processing_time_sec = 0.0
            p99_processing_time_sec = 0.0
            mean_instant_fps = 0.0
            median_fps = 0.0
            std_fps = 0.0
            p05_fps = 0.0
            p95_fps = 0.0

        computation = {
            'mean_processing_time_formula': 'total_prediction_time_sec / trial_count',
            'mean_fps_formula': 'trial_count / total_prediction_time_sec',
            'mean_processing_time_substitution': (
                f"{self.total_prediction_time_sec:.6f} / {trial_count} = {mean_processing_time_sec:.6f}"
                if trial_count > 0 else '0 / 0 = 0'
            ),
            'mean_fps_substitution': (
                f"{trial_count} / {self.total_prediction_time_sec:.6f} = {mean_fps:.6f}"
                if self.total_prediction_time_sec > 0 else '0 / 0 = 0'
            ),
        }

        return {
            'mode': 'runtime',
            'session_duration_sec': float(time.time() - self.session_start_time),
            'samples': int(self.total_predictions),
            'trial_count': trial_count,
            'total_inference_passes': trial_count,
            'total_prediction_time_sec': float(self.total_prediction_time_sec),
            'mean_processing_time_sec': float(mean_processing_time_sec),
            'median_processing_time_sec': median_processing_time_sec,
            'std_processing_time_sec': std_processing_time_sec,
            'min_processing_time_sec': min_processing_time_sec,
            'max_processing_time_sec': max_processing_time_sec,
            'p95_processing_time_sec': p95_processing_time_sec,
            'p99_processing_time_sec': p99_processing_time_sec,
            'mean_fps': float(mean_fps),
            'mean_instantaneous_fps': mean_instant_fps,
            'median_fps': median_fps,
            'std_fps': std_fps,
            'p05_fps': p05_fps,
            'p95_fps': p95_fps,
            'computation': computation,
        }

    def _plot_fps_analysis(self, speed_metrics, output_dir):
        """Save an FPS-focused analysis chart with stability and distribution views."""
        if not self.prediction_times_sec:
            return None

        per_trial_fps = 1.0 / np.array(self.prediction_times_sec, dtype=np.float64)
        trial_idx = np.arange(1, len(per_trial_fps) + 1)
        window = min(20, len(per_trial_fps))

        if window > 1:
            rolling_fps = np.convolve(per_trial_fps, np.ones(window) / window, mode='valid')
            rolling_x = np.arange(window, len(per_trial_fps) + 1)
        else:
            rolling_fps = per_trial_fps
            rolling_x = trial_idx

        mean_fps = speed_metrics['mean_fps']
        median_fps = speed_metrics['median_fps']
        p05_fps = speed_metrics['p05_fps']
        p95_fps = speed_metrics['p95_fps']

        old_rc = plt.rcParams.copy()
        try:
            plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 20})
            fig1, ax1 = plt.subplots(figsize=(15, 8.5))
            ax1.axhspan(10, 30, color='#9be7a8', alpha=0.20, label='Real-time range (10-30 FPS)')
            ax1.plot(trial_idx, per_trial_fps, color='#5dade2', linewidth=0.9, alpha=0.55, label='Per-trial FPS')
            ax1.plot(rolling_x, rolling_fps, color='#1f618d', linewidth=2.0, label=f'Rolling FPS (window={window})')
            ax1.axhline(mean_fps, color='#c0392b', linestyle='--', linewidth=1.6, label=f'Mean FPS: {mean_fps:.2f}')
            ax1.axhline(median_fps, color='#27ae60', linestyle='--', linewidth=1.4, label=f'Median FPS: {median_fps:.2f}')
            ax1.set_title('FPS Stability Over Trials')
            ax1.set_xlabel('Trial', labelpad=8)
            ax1.set_ylabel('Frames per second')
            ax1.grid(alpha=0.3)
            ax1.legend(
                loc='upper left',
                bbox_to_anchor=(0, -0.32, 1, 0.12),
                ncol=3,
                mode='expand',
                framealpha=0.9,
                fontsize=18,
                labelspacing=0.4,
                columnspacing=2.5,
                handletextpad=1.8,
                borderpad=0.3,
            )
            fig1.tight_layout(rect=[0, 0.15, 1, 1])
            out_path1 = os.path.join(output_dir, 'fps_stability.png')
            fig1.savefig(out_path1, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            fig2, ax2 = plt.subplots(figsize=(15, 8.5))
            ax2.hist(per_trial_fps, bins=min(32, max(10, len(per_trial_fps) // 8)), color='#48c9b0', alpha=0.82, edgecolor='white')
            ax2.axvline(mean_fps, color='#c0392b', linestyle='--', linewidth=1.6, label=f'Mean: {mean_fps:.2f}')
            ax2.axvline(median_fps, color='#27ae60', linestyle='--', linewidth=1.4, label=f'Median: {median_fps:.2f}')
            ax2.axvline(p05_fps, color='#7d3c98', linestyle='--', linewidth=1.2, label=f'P05: {p05_fps:.2f}')
            ax2.axvline(p95_fps, color='#8e44ad', linestyle='--', linewidth=1.2, label=f'P95: {p95_fps:.2f}')
            ax2.set_title('FPS Distribution')
            ax2.set_xlabel('Frames per second')
            ax2.set_ylabel('Frequency')
            ax2.grid(alpha=0.3)
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, framealpha=0.9, fontsize=18, labelspacing=0.4, columnspacing=2.5, handletextpad=1.8, borderpad=0.3)
            fig2.tight_layout(rect=[0, 0.14, 1, 1])
            out_path2 = os.path.join(output_dir, 'fps_distribution.png')
            fig2.savefig(out_path2, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            return {'stability': out_path1, 'distribution': out_path2}
        finally:
            plt.rcParams.update(old_rc)

    def _plot_cumulative_frames_over_time(self, output_dir):
        """Save cumulative frames processed vs elapsed time plot."""
        if not self.prediction_times_sec:
            return None

        times = np.array(self.prediction_times_sec, dtype=np.float64)
        elapsed_sec = np.cumsum(times)
        cumulative_frames = np.arange(1, len(times) + 1)

        # Linear reference slope from overall average FPS.
        avg_fps = len(times) / float(np.sum(times)) if np.sum(times) > 0 else 0.0
        ref_frames = avg_fps * elapsed_sec

        plt.figure(figsize=(10, 6))
        plt.plot(elapsed_sec, cumulative_frames, color='#1f77b4', linewidth=2.2, label='Observed cumulative frames')
        plt.plot(elapsed_sec, ref_frames, color='#d62728', linestyle='--', linewidth=1.6, label=f'Average-rate reference ({avg_fps:.2f} FPS)')

        plt.title('Cumulative Frames vs Elapsed Time')
        plt.xlabel('Elapsed time (seconds)')
        plt.ylabel('Cumulative frames processed')
        plt.grid(alpha=0.3)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, framealpha=0.9)

        summary_text = (
            f'Total frames: {len(times)}\n'
            f'Total time: {elapsed_sec[-1]:.2f} s\n'
            f'Average FPS: {avg_fps:.2f}'
        )
        plt.text(
            0.02,
            0.98,
            summary_text,
            transform=plt.gca().transAxes,
            va='top',
            ha='left',
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#bdc3c7'),
        )

        out_path = os.path.join(output_dir, 'cumulative_frames_over_time.png')
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        return out_path

    def _plot_fps_vs_detected_faces(self, output_dir):
        """Save dual-axis FPS vs detected-faces over time chart."""
        if not self.frame_elapsed_time_sec:
            return None

        elapsed_sec = np.array(self.frame_elapsed_time_sec, dtype=np.float64)
        instant_fps = np.array(self.frame_instant_fps, dtype=np.float64)
        detected_faces = np.array(self.frame_detected_faces, dtype=np.int32)

        if len(instant_fps) > 1:
            window = min(15, len(instant_fps))
            smooth_fps = np.convolve(instant_fps, np.ones(window) / window, mode='same')
        else:
            smooth_fps = instant_fps

        fig, ax1 = plt.subplots(figsize=(10.5, 6))
        ax2 = ax1.twinx()

        ax1.plot(elapsed_sec, instant_fps, color='#4c8eda', alpha=0.35, linewidth=1.0, label='Instant FPS')
        ax1.plot(elapsed_sec, smooth_fps, color='#2f6fb6', linewidth=2.2, label='Smoothed FPS')
        ax1.set_xlabel('Time (Seconds)', fontweight='bold')
        ax1.set_ylabel('Frame Rate Per Second', color='#2f6fb6', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#2f6fb6')
        ax1.grid(alpha=0.25)

        ax2.plot(elapsed_sec, detected_faces, color='#d14b46', linewidth=2.0, label='Detected Faces', drawstyle='steps-post')
        ax2.set_ylabel('Number of Detected Faces', color='#d14b46', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#d14b46')

        mean_fps = float(np.mean(instant_fps)) if len(instant_fps) > 0 else 0.0
        mean_faces = float(np.mean(detected_faces)) if len(detected_faces) > 0 else 0.0
        ax1.text(0.02, 0.95, f'Mean FPS: {mean_fps:.2f}', transform=ax1.transAxes,
             color='#2f6fb6', fontsize=14, fontweight='bold')
        ax2.text(0.62, 0.95, f'Mean Faces: {mean_faces:.2f}', transform=ax2.transAxes,
             color='#d14b46', fontsize=14, fontweight='bold')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.9)

        plt.title('FPS and Detected Faces vs Time', fontweight='bold')
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        out_path = os.path.join(output_dir, 'fps_vs_detected_faces.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return out_path

    def _plot_processing_time_variability(self, output_dir):
        """Save processing-time variability plots from per-trial timings."""
        if not self.prediction_times_sec:
            return None

        times_ms = np.array(self.prediction_times_sec, dtype=np.float64) * 1000.0
        trial_idx = np.arange(1, len(times_ms) + 1)

        mean_ms = float(np.mean(times_ms))
        median_ms = float(np.median(times_ms))
        p95_ms = float(np.percentile(times_ms, 95))
        p99_ms = float(np.percentile(times_ms, 99))
        max_ms = float(np.max(times_ms))
        min_ms = float(np.min(times_ms))

        # Focus axis on typical range so one rare spike does not flatten the whole trend.
        focus_upper_ms = max(p99_ms * 1.15, mean_ms * 1.4)
        outlier_mask = times_ms > focus_upper_ms
        outlier_count = int(np.sum(outlier_mask))
        display_times_ms = np.minimum(times_ms, focus_upper_ms)

        old_rc = plt.rcParams.copy()
        try:
            plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 20})
            fig3, ax3 = plt.subplots(figsize=(15, 8.5))
            warmup_trials = min(20, len(times_ms))
            if warmup_trials > 0:
                ax3.axvspan(1, warmup_trials, color='#f1c40f', alpha=0.12, label=f'Warm-up (first {warmup_trials} trials)')
            ax3.plot(trial_idx, display_times_ms, color='#2980b9', linewidth=1.1, alpha=0.9, label='Per-trial time')
            ax3.axhline(mean_ms, color='#c0392b', linestyle='--', linewidth=1.6, label=f'Mean: {mean_ms:.2f} ms')
            ax3.axhline(median_ms, color='#27ae60', linestyle='--', linewidth=1.4, label=f'Median: {median_ms:.2f} ms')
            ax3.axhline(p95_ms, color='#8e44ad', linestyle='--', linewidth=1.4, label=f'P95: {p95_ms:.2f} ms')
            if outlier_count > 0:
                ax3.scatter(trial_idx[outlier_mask], np.full(outlier_count, focus_upper_ms), color='#e74c3c', marker='x', s=35, label=f'Outliers above axis: {outlier_count}')
            ax3.set_title('Per-Trial Processing Time')
            ax3.set_xlabel('Trial')
            ax3.set_ylabel('Processing time (ms)')
            ax3.set_ylim(0, focus_upper_ms * 1.05)
            ax3.grid(alpha=0.3)
            ax3.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.18),
                ncol=4,
                framealpha=0.9,
                fontsize=13,
                labelspacing=0.4,
                columnspacing=1.2,
                handletextpad=0.8,
                borderpad=0.5,
            )
            fig3.tight_layout(rect=[0, 0.12, 1, 1])
            out_path3 = os.path.join(output_dir, 'processing_time_per_trial.png')
            fig3.savefig(out_path3, dpi=300, bbox_inches='tight')
            plt.close(fig3)
            fig4, ax4 = plt.subplots(figsize=(15, 8.5))
            focused_times_ms = times_ms[times_ms <= focus_upper_ms]
            p01_ms = float(np.percentile(focused_times_ms, 1)) if len(focused_times_ms) > 0 else min_ms
            p99_focus_ms = float(np.percentile(focused_times_ms, 99)) if len(focused_times_ms) > 0 else max_ms
            half_range = max(median_ms - p01_ms, p99_focus_ms - median_ms, 0.25)
            x_left = median_ms - half_range * 1.1
            x_right = median_ms + half_range * 1.1
            ax4.hist(focused_times_ms, bins=min(32, max(10, len(focused_times_ms) // 8)), color='#16a085', alpha=0.82, edgecolor='white')
            ax4.axvline(mean_ms, color='#c0392b', linestyle='--', linewidth=1.6, label=f'Mean: {mean_ms:.2f} ms')
            ax4.axvline(median_ms, color='#27ae60', linestyle='--', linewidth=1.4, label=f'Median: {median_ms:.2f} ms')
            ax4.axvline(p95_ms, color='#8e44ad', linestyle='--', linewidth=1.4, label=f'P95: {p95_ms:.2f} ms')
            ax4.set_title('Processing Time Distribution')
            ax4.set_xlabel('Processing time (ms)')
            ax4.set_ylabel('Frequency')
            ax4.set_xlim(x_left, x_right)
            ax4.grid(alpha=0.3)
            ax4.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.18),
                ncol=3,
                framealpha=0.9,
                fontsize=13,
                labelspacing=0.4,
                columnspacing=1.2,
                handletextpad=0.8,
                borderpad=0.5,
            )
            fig4.tight_layout(rect=[0, 0.12, 1, 1])
            out_path4 = os.path.join(output_dir, 'processing_time_distribution.png')
            fig4.savefig(out_path4, dpi=300, bbox_inches='tight')
            plt.close(fig4)
            return {'per_trial': out_path3, 'distribution': out_path4}
        finally:
            plt.rcParams.update(old_rc)

    def _plot_processing_speed(self, speed_metrics, output_dir):
        """Save a processing-speed graph for runtime session metrics."""
        fps = speed_metrics['mean_fps']
        sec_per_pred = speed_metrics['mean_processing_time_sec']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        real_time_min = 10
        real_time_max = 30
        fps_color = '#2ecc71' if fps >= real_time_min else '#e67e22'
        axes[0].bar(['Avg FPS'], [fps], color=fps_color, alpha=0.85)
        axes[0].axhspan(real_time_min, real_time_max, color='#9be7a8', alpha=0.25, label='Real-time range (10-30 FPS)')
        axes[0].set_ylabel('Frames per second')
        axes[0].set_title('Processing Throughput')
        axes[0].set_ylim(0, max(real_time_max + 5, fps * 1.2 if fps > 0 else real_time_max + 5))
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=1, framealpha=0.9)
        axes[0].text(0, fps, f'{fps:.2f} FPS', ha='center', va='bottom', fontweight='bold')

        axes[1].bar(['Mean Processing Time'], [sec_per_pred], color='#3498db', alpha=0.85)
        axes[1].set_ylabel('Seconds')
        axes[1].set_title('Processing Time (Mean)')
        axes[1].set_ylim(0, sec_per_pred * 1.3 if sec_per_pred > 0 else 0.1)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].text(0, sec_per_pred, f'{sec_per_pred:.4f} sec (mean)', ha='center', va='bottom', fontweight='bold')

        plt.suptitle(f'Processing Speed and Efficiency - {os.path.basename(self.model_path)}')
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        out_path = os.path.join(output_dir, 'processing_speed_efficiency.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return out_path

    def save_runtime_analysis(self, output_dir=None):
        """Save processing-speed analysis artifacts for a runtime session."""
        speed_metrics = self.get_processing_speed_metrics()

        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join('evaluation_results', f'revised_runtime_analysis_{timestamp}')

        os.makedirs(output_dir, exist_ok=True)

        payload = {
            'model_path': self.model_path,
            'evaluation_date': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'emotions': [self.get_display_label(label) for label in self.EMOTION_LABELS],
            'emotion_counts': self.emotion_counts,
            'processing_speed': speed_metrics,
            'per_trial_processing_time_sec': self.prediction_times_sec,
            'runtime_time_series': {
                'elapsed_time_sec': self.frame_elapsed_time_sec,
                'instant_fps': self.frame_instant_fps,
                'detected_faces': self.frame_detected_faces,
            },
        }

        json_path = os.path.join(output_dir, 'runtime_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2)

        csv_path = os.path.join(output_dir, 'per_trial_processing_time.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trial', 'processing_time_sec', 'processing_time_ms'])
            for i, t in enumerate(self.prediction_times_sec, start=1):
                writer.writerow([i, f'{t:.9f}', f'{t * 1000.0:.3f}'])

        frame_csv_path = os.path.join(output_dir, 'fps_faces_time_series.csv')
        with open(frame_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['elapsed_time_sec', 'instant_fps', 'detected_faces'])
            for t, fps, faces in zip(self.frame_elapsed_time_sec, self.frame_instant_fps, self.frame_detected_faces):
                writer.writerow([f'{t:.6f}', f'{fps:.6f}', int(faces)])

        md_path = os.path.join(output_dir, 'PROCESSING_SPEED_EFFICIENCY_ANALYSIS.md')
        with open(md_path, 'w') as f:
            f.write('## 3.2 Processing Speed & Efficiency\n\n')
            f.write('### Data\n\n')
            f.write('- Frames per second (FPS)\n')
            f.write('- Processing time per prediction\n\n')
            f.write('### Analysis\n\n')
            f.write('- Mean FPS\n')
            f.write('- Mean processing time\n\n')
            f.write('### Computed Results\n\n')
            f.write('| Metric | Value |\n')
            f.write('|--------|-------|\n')
            f.write(f"| Avg FPS | {speed_metrics['mean_fps']:.2f} FPS |\n")
            f.write(f"| Mean Processing Time | {speed_metrics['mean_processing_time_sec']:.4f} sec |\n")
            f.write(f"| Median Processing Time | {speed_metrics['median_processing_time_sec']:.4f} sec |\n")
            f.write(f"| Std Processing Time | {speed_metrics['std_processing_time_sec']:.4f} sec |\n")
            f.write(f"| P95 Processing Time | {speed_metrics['p95_processing_time_sec']:.4f} sec |\n")
            f.write(f"| P99 Processing Time | {speed_metrics['p99_processing_time_sec']:.4f} sec |\n")
            f.write(f"| Total Prediction Time | {speed_metrics['total_prediction_time_sec']:.2f} sec |\n")
            f.write(f"| Samples | {speed_metrics['samples']} |\n")
            f.write(f"| Trials Used for Mean | {speed_metrics['trial_count']} |\n")
            f.write(f"| Total Inference Passes | {speed_metrics['total_inference_passes']} |\n")
            f.write(f"| Mean Instantaneous FPS | {speed_metrics['mean_instantaneous_fps']:.2f} |\n")
            f.write(f"| Median FPS | {speed_metrics['median_fps']:.2f} |\n")
            f.write(f"| FPS Std Dev | {speed_metrics['std_fps']:.2f} |\n")
            f.write(f"| FPS P05 | {speed_metrics['p05_fps']:.2f} |\n")
            f.write(f"| FPS P95 | {speed_metrics['p95_fps']:.2f} |\n")
            f.write('\n### Computation\n\n')
            f.write('- Mean Processing Time = Total Prediction Time / Trials Used for Mean\n')
            f.write(
                f"- Mean Processing Time = {speed_metrics['total_prediction_time_sec']:.6f} / "
                f"{speed_metrics['trial_count']} = {speed_metrics['mean_processing_time_sec']:.6f} sec\n"
            )
            f.write('- Mean FPS = Trials Used for Mean / Total Prediction Time\n')
            f.write(
                f"- Mean FPS = {speed_metrics['trial_count']} / "
                f"{speed_metrics['total_prediction_time_sec']:.6f} = {speed_metrics['mean_fps']:.6f}\n"
            )
            f.write('\n### Variability Notes\n\n')
            f.write('- Median and P95/P99 are recommended for research reporting because they are robust to outliers.\n')
            f.write('- Use per_trial_processing_time.csv for reproducibility and further statistical tests.\n')
            f.write('\n### Conclusion\n\n')
            f.write('- Faster = more efficient\n')
            f.write('- Compare with real-time standard (~10-30 FPS)\n')

        graph_path = self._plot_processing_speed(speed_metrics, output_dir)
        variability_plot_path = self._plot_processing_time_variability(output_dir)
        fps_plot_path = self._plot_fps_analysis(speed_metrics, output_dir)
        cumulative_plot_path = self._plot_cumulative_frames_over_time(output_dir)
        fps_faces_plot_path = self._plot_fps_vs_detected_faces(output_dir)

        print(f"✅ Runtime analysis saved: {output_dir}")
        print(f"   - {os.path.basename(json_path)}")
        print(f"   - {os.path.basename(csv_path)}")
        print(f"   - {os.path.basename(frame_csv_path)}")
        print(f"   - {os.path.basename(md_path)}")
        print(f"   - {os.path.basename(graph_path)}")
        if variability_plot_path:
            if isinstance(variability_plot_path, dict):
                for key, path in variability_plot_path.items():
                    print(f"   - {os.path.basename(path)}")
            else:
                print(f"   - {os.path.basename(variability_plot_path)}")
        if fps_plot_path:
            if isinstance(fps_plot_path, dict):
                for key, path in fps_plot_path.items():
                    print(f"   - {os.path.basename(path)}")
            else:
                print(f"   - {os.path.basename(fps_plot_path)}")
        if cumulative_plot_path:
            print(f"   - {os.path.basename(cumulative_plot_path)}")
        if fps_faces_plot_path:
            print(f"   - {os.path.basename(fps_faces_plot_path)}")

        return {
            'output_dir': output_dir,
            'json_path': json_path,
            'csv_path': csv_path,
            'frame_csv_path': frame_csv_path,
            'markdown_path': md_path,
            'graph_path': graph_path,
            'variability_plot_path': variability_plot_path,
            'fps_plot_path': fps_plot_path,
            'cumulative_plot_path': cumulative_plot_path,
            'fps_faces_plot_path': fps_faces_plot_path,
        }
    
    def draw_emotion_info(self, image, x, y, w, h, emotion, confidence):
        """
        Draw emotion information on the image
        
        Args:
            image: Image to draw on
            x, y, w, h: Face bounding box coordinates
            emotion: Predicted emotion label
            confidence: Confidence score
        """
        # Get color for this emotion
        color = self.EMOTION_COLORS.get(emotion, (255, 255, 255))
        
        # Draw face rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Prepare text
        label = f"{emotion}: {confidence*100:.1f}%"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            image,
            (x, y - text_height - 10),
            (x + text_width + 10, y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Draw additional info if debug mode
        if self.show_debug:
            info_y = y + h + 20
            cv2.putText(
                image,
                f"Size: {w}x{h}",
                (x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_frame_count += 1
        elapsed_time = time.time() - self.fps_start_time
        
        if elapsed_time >= 1.0:
            self.current_fps = self.fps_frame_count / elapsed_time
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
    
    def print_statistics(self):
        """Print emotion detection statistics"""
        speed_metrics = self.get_processing_speed_metrics()

        print("\n" + "="*70)
        print("EMOTION DETECTION STATISTICS")
        print("="*70)
        print(f"Total predictions: {self.total_predictions}")
        print(f"Total prediction time: {speed_metrics['total_prediction_time_sec']:.2f} sec")
        print(f"Trials used for mean: {speed_metrics['trial_count']}")
        print(f"Mean processing time: {speed_metrics['mean_processing_time_sec']:.4f} sec")
        print(f"Mean FPS: {speed_metrics['mean_fps']:.2f}")
        print("\nEmotion distribution:")
        for emotion, count in sorted(self.emotion_counts.items(), 
                                     key=lambda x: x[1], reverse=True):
            percentage = (count / self.total_predictions * 100) if self.total_predictions > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"  {emotion:10s}: {count:5d} ({percentage:5.1f}%) {bar}")
        print("="*70)
    
    def process_image(self, image_path, output_path=None):
        """
        Process a single image and detect emotions
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save output image
            
        Returns:
            List of (emotion, confidence) tuples for each face
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not read image: {image_path}")
            return []
        
        # Detect faces (single face mode by default)
        faces = self.detect_faces(image, single_face=self.single_face)
        mode_text = " (largest/closest)" if self.single_face and len(faces) > 0 else ""
        print(f"📷 Found {len(faces)} face(s){mode_text} in {os.path.basename(image_path)}")
        
        results = []
        
        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_img = image[y:y+h, x:x+w]
            
            # Predict emotion
            emotion, confidence = self.predict_emotion(face_img)
            results.append((emotion, confidence))
            
            # Draw results
            self.draw_emotion_info(image, x, y, w, h, emotion, confidence)
            
            print(f"   Face {i+1}: {emotion} ({confidence*100:.1f}%)")
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"✅ Output saved to: {output_path}")
        else:
            cv2.imshow('Revised Emotion Detection', image)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
    
    def run(self):
        """Run real-time emotion detection on webcam"""
        print("\n" + "="*70)
        print("REVISED EMOTION DETECTION - Real-time Webcam")
        print("="*70)
        print("Model: Revised CNN 2026 (6 classes)")
        print("Classes: angry, disgust, shocked, happy, neutral, sad")
        print("="*70)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 'd' to toggle debug info")
        print("  - Press 's' to save screenshot")
        print("  - Press 'p' to print statistics")
        print("="*70 + "\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces (single face mode by default)
                faces = self.detect_faces(frame, single_face=self.single_face)

                # Record frame-level time series for FPS vs detected-faces chart.
                now_perf = time.perf_counter()
                if self.last_frame_perf_time is None:
                    instant_fps = 0.0
                else:
                    frame_delta = now_perf - self.last_frame_perf_time
                    instant_fps = (1.0 / frame_delta) if frame_delta > 0 else 0.0
                self.last_frame_perf_time = now_perf

                elapsed_from_start = now_perf - self.session_perf_start_time
                self.frame_elapsed_time_sec.append(float(elapsed_from_start))
                self.frame_instant_fps.append(float(instant_fps))
                self.frame_detected_faces.append(int(len(faces)))
                
                # Process each face
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence = self.predict_emotion(face_img)
                    
                    # Draw results
                    self.draw_emotion_info(frame, x, y, w, h, emotion, confidence)
                
                # Update and display FPS
                self.update_fps()
                fps_text = f"FPS: {self.current_fps:.1f}"
                cv2.putText(frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display info
                info_text = f"Faces: {len(faces)} | Predictions: {self.total_predictions}"
                cv2.putText(frame, info_text, (10, frame.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                control_text = "Press 'd' debug | 'p' stats | 's' save | 'q' quit"
                cv2.putText(frame, control_text, (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Revised Emotion Detection - 2026 CNN', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")
                elif key == ord('s'):
                    screenshot_path = f'screenshot_revised_{screenshot_count}.png'
                    cv2.imwrite(screenshot_path, frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                    screenshot_count += 1
                elif key == ord('p'):
                    self.print_statistics()
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            if self.total_predictions > 0:
                self.print_statistics()
            
            print("\n✅ Revised emotion detection stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Revised Emotion Detection using 2026 CNN Model')
    parser.add_argument('--model', type=str, 
                       default='models/emotion_revised_cnn_20260220_030450.keras',
                       help='Path to the revised CNN model')
    parser.add_argument('--image', type=str, help='Process a single image instead of webcam')
    parser.add_argument('--output', type=str, help='Output path for processed image')
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    parser.add_argument('--all-faces', action='store_true', 
                       help='Process all detected faces (default: only largest/closest face)')
    parser.add_argument('--analysis-output', type=str, default=None,
                       help='Optional output directory for runtime analysis artifacts')
    
    args = parser.parse_args()
    
    try:
        detector = RevisedEmotionDetector(
            model_path=args.model,
            show_debug=args.debug,
            single_face=not args.all_faces  # Default: single face, unless --all-faces flag
        )
        
        if args.image:
            # Process single image
            detector.process_image(args.image, args.output)
            if detector.total_predictions > 0:
                detector.save_runtime_analysis(args.analysis_output)
        else:
            # Run real-time detection
            detector.run()
            if detector.total_predictions > 0:
                detector.save_runtime_analysis(args.analysis_output)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
