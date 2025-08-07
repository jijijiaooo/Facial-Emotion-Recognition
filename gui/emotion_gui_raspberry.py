#!/usr/bin/env python3
"""
Raspberry Pi Emotion Recognition GUI
Tkinter-based GUI optimized for Raspberry Pi OS
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import threading
import time
from PIL import Image, ImageTk
from collections import deque, Counter
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'core'))

class EmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition - Raspberry Pi")
        self.root.geometry("800x600")
        
        # Configure for Raspberry Pi
        self.root.configure(bg='#2c3e50')
        
        # Emotion recognition variables
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.colors = {
            'Angry': '#e74c3c', 'Disgust': '#27ae60', 'Fear': '#9b59b6',
            'Happy': '#f1c40f', 'Neutral': '#95a5a6', 'Sad': '#3498db',
            'Surprise': '#e67e22'
        }
        
        # Camera and detection variables
        self.cap = None
        self.model = None
        self.is_running = False
        self.current_frame = None
        self.detection_enabled = True
        
        # Statistics
        self.emotion_counts = Counter()
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0
        
        # Face detection
        self.face_cascade = None
        self.init_face_detection()
        
        # Load model
        self.load_model()
        
        # Create GUI
        self.create_widgets()
        
        # Start camera
        self.start_camera()
    
    def init_face_detection(self):
        """Initialize face detection"""
        try:
            # Try to find haar cascade file
            cascade_paths = [
                'haarcascade_frontalface_default.xml',
                'config/haarcascade_frontalface_default.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            ]
            
            for path in cascade_paths:
                if os.path.exists(path):
                    self.face_cascade = cv2.CascadeClassifier(path)
                    break
            
            if self.face_cascade is None:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            self.log_message("✅ Face detection initialized")
            
        except Exception as e:
            self.log_message(f"⚠️ Face detection error: {e}")
    
    def load_model(self):
        """Load emotion recognition model"""
        try:
            # Try different model paths
            model_paths = [
                'model_file_30epochs.h5',
                'models/model_file_30epochs.h5',
                'src/core/model_file_30epochs.h5'
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    try:
                        from tensorflow import keras
                        self.model = keras.models.load_model(path)
                        self.log_message(f"✅ Model loaded: {path}")
                        return
                    except ImportError:
                        self.log_message("⚠️ TensorFlow not available, using basic detection")
                        break
                    except Exception as e:
                        self.log_message(f"⚠️ Model load error: {e}")
                        continue
            
            self.log_message("⚠️ No model loaded - using basic emotion detection")
            
        except Exception as e:
            self.log_message(f"❌ Model loading failed: {e}")
    
    def create_widgets(self):
        """Create GUI widgets optimized for Raspberry Pi"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="🎭 Emotion Recognition System",
            font=("Arial", 16, "bold"),
            bg='#2c3e50',
            fg='white'
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Camera controls
        self.start_btn = tk.Button(
            control_frame,
            text="▶️ Start Camera",
            command=self.toggle_camera,
            bg='#27ae60',
            fg='white',
            font=("Arial", 10, "bold"),
            width=15
        )
        self.start_btn.grid(row=0, column=0, pady=5, sticky=tk.W+tk.E)
        
        self.detection_btn = tk.Button(
            control_frame,
            text="🎯 Toggle Detection",
            command=self.toggle_detection,
            bg='#3498db',
            fg='white',
            font=("Arial", 10),
            width=15
        )
        self.detection_btn.grid(row=1, column=0, pady=5, sticky=tk.W+tk.E)
        
        # Settings
        ttk.Separator(control_frame, orient='horizontal').grid(row=2, column=0, sticky=tk.W+tk.E, pady=10)
        
        tk.Label(control_frame, text="Settings:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky=tk.W)
        
        # Confidence threshold
        tk.Label(control_frame, text="Confidence:").grid(row=4, column=0, sticky=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.6)
        confidence_scale = tk.Scale(
            control_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.confidence_var
        )
        confidence_scale.grid(row=5, column=0, sticky=tk.W+tk.E)
        
        # Statistics
        ttk.Separator(control_frame, orient='horizontal').grid(row=6, column=0, sticky=tk.W+tk.E, pady=10)
        
        stats_label = tk.Label(control_frame, text="Statistics:", font=("Arial", 10, "bold"))
        stats_label.grid(row=7, column=0, sticky=tk.W)
        
        self.stats_text = tk.Text(
            control_frame,
            height=8,
            width=25,
            font=("Courier", 8),
            bg='#34495e',
            fg='white'
        )
        self.stats_text.grid(row=8, column=0, sticky=tk.W+tk.E+tk.N+tk.S, pady=5)
        
        # Center panel - Video feed
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = tk.Label(
            video_frame,
            text="📷 Camera feed will appear here",
            bg='black',
            fg='white',
            font=("Arial", 12),
            width=50,
            height=20
        )
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right panel - Emotion display
        emotion_frame = ttk.LabelFrame(main_frame, text="Current Emotion", padding="10")
        emotion_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        self.emotion_display = tk.Label(
            emotion_frame,
            text="😐\nNeutral",
            font=("Arial", 24, "bold"),
            bg='#95a5a6',
            fg='white',
            width=12,
            height=4
        )
        self.emotion_display.grid(row=0, column=0, pady=10)
        
        self.confidence_display = tk.Label(
            emotion_frame,
            text="Confidence: 0%",
            font=("Arial", 12),
            bg='#2c3e50',
            fg='white'
        )
        self.confidence_display.grid(row=1, column=0)
        
        # Emotion history
        history_label = tk.Label(emotion_frame, text="Recent Emotions:", font=("Arial", 10, "bold"))
        history_label.grid(row=2, column=0, pady=(20, 5))
        
        self.history_frame = tk.Frame(emotion_frame, bg='#2c3e50')
        self.history_frame.grid(row=3, column=0, sticky=tk.W+tk.E)
        
        # Bottom status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg='#34495e',
            fg='white'
        )
        status_bar.grid(row=2, column=0, columnspan=3, sticky=tk.W+tk.E, pady=(10, 0))
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="System Log", padding="5")
        log_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W+tk.E, pady=(10, 0))
        
        self.log_text = tk.Text(
            log_frame,
            height=4,
            font=("Courier", 8),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        self.log_text.grid(row=0, column=0, sticky=tk.W+tk.E)
        
        # Scrollbar for log
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky=tk.N+tk.S)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        
        log_frame.columnconfigure(0, weight=1)
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # Keep only last 50 lines
        lines = self.log_text.get("1.0", tk.END).split('\n')
        if len(lines) > 50:
            self.log_text.delete("1.0", f"{len(lines)-50}.0")
    
    def start_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            
            # Optimize for Raspberry Pi
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for Pi
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                self.is_running = True
                self.start_btn.config(text="⏸️ Stop Camera", bg='#e74c3c')
                self.log_message("✅ Camera started")
                self.update_frame()
            else:
                self.log_message("❌ Failed to open camera")
                
        except Exception as e:
            self.log_message(f"❌ Camera error: {e}")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def stop_camera(self):
        """Stop camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(text="▶️ Start Camera", bg='#27ae60')
        self.video_label.config(image='', text="📷 Camera stopped")
        self.log_message("⏸️ Camera stopped")
    
    def toggle_detection(self):
        """Toggle emotion detection"""
        self.detection_enabled = not self.detection_enabled
        status = "enabled" if self.detection_enabled else "disabled"
        self.log_message(f"🎯 Detection {status}")
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        if self.model is None:
            # Basic rule-based emotion detection
            return self.basic_emotion_detection(face_img)
        
        try:
            # Preprocess for model
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))
            
            # Predict
            result = self.model.predict(reshaped, verbose=0)
            emotion_idx = np.argmax(result, axis=1)[0]
            confidence = result[0][emotion_idx]
            
            return self.emotions[emotion_idx], confidence
            
        except Exception as e:
            self.log_message(f"⚠️ Prediction error: {e}")
            return "Neutral", 0.5
    
    def basic_emotion_detection(self, face_img):
        """Basic emotion detection when no model is available"""
        # Simple brightness-based detection
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness > 140:
            return "Happy", 0.7
        elif brightness < 80:
            return "Sad", 0.6
        else:
            return "Neutral", 0.5
    
    def update_emotion_display(self, emotion, confidence):
        """Update emotion display"""
        # Emotion emojis
        emojis = {
            'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
            'Happy': '😊', 'Neutral': '😐', 'Sad': '😢',
            'Surprise': '😲'
        }
        
        emoji = emojis.get(emotion, '😐')
        color = self.colors.get(emotion, '#95a5a6')
        
        self.emotion_display.config(
            text=f"{emoji}\n{emotion}",
            bg=color
        )
        
        self.confidence_display.config(
            text=f"Confidence: {confidence:.1%}"
        )
        
        # Update statistics
        self.emotion_counts[emotion] += 1
        self.update_statistics()
    
    def update_statistics(self):
        """Update statistics display"""
        total = sum(self.emotion_counts.values())
        
        stats_text = f"FPS: {self.current_fps}\n"
        stats_text += f"Total: {total}\n\n"
        
        for emotion in self.emotions:
            count = self.emotion_counts[emotion]
            percentage = (count / total * 100) if total > 0 else 0
            stats_text += f"{emotion[:4]}: {count:3d} ({percentage:4.1f}%)\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_start >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start = time.time()
    
    def update_frame(self):
        """Update video frame"""
        if not self.is_running or not self.cap:
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.log_message("⚠️ Failed to read frame")
                self.root.after(100, self.update_frame)
                return
            
            self.update_fps()
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces and emotions
            if self.detection_enabled:
                faces = self.detect_faces(frame)
                
                for (x, y, w, h) in faces:
                    # Extract face
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence = self.predict_emotion(face_img)
                    
                    # Only update if confidence is above threshold
                    if confidence >= self.confidence_var.get():
                        self.update_emotion_display(emotion, confidence)
                        
                        # Draw rectangle and label
                        color = tuple(int(self.colors[emotion].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                        color = (color[2], color[1], color[0])  # RGB to BGR
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        label = f"{emotion} ({confidence:.1%})"
                        cv2.putText(frame, label, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Convert frame for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Resize for display (optimize for Pi)
            display_size = (480, 360)
            frame_pil = frame_pil.resize(display_size, Image.Resampling.LANCZOS)
            
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update video label
            self.video_label.config(image=frame_tk, text="")
            self.video_label.image = frame_tk  # Keep reference
            
            # Update status
            self.status_var.set(f"Running - FPS: {self.current_fps} - Detection: {'ON' if self.detection_enabled else 'OFF'}")
            
        except Exception as e:
            self.log_message(f"⚠️ Frame update error: {e}")
        
        # Schedule next update
        self.root.after(50, self.update_frame)  # ~20 FPS for Pi
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.root.destroy()

def main():
    # Check if running on Raspberry Pi
    try:
        with open('/proc/cpuinfo', 'r') as f:
            if 'Raspberry Pi' in f.read():
                print("🍓 Running on Raspberry Pi - optimized settings enabled")
    except:
        pass
    
    # Create and run GUI
    root = tk.Tk()
    app = EmotionGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    print("🎭 Starting Emotion Recognition GUI...")
    print("💡 Optimized for Raspberry Pi OS")
    
    root.mainloop()

if __name__ == "__main__":
    main()