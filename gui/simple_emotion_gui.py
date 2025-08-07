#!/usr/bin/env python3
"""
Simple Emotion Recognition GUI
Compatible version for macOS and other systems
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import threading
import time
from PIL import Image, ImageTk
from collections import Counter
import os
import sys

class SimpleEmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition")
        self.root.geometry("900x700")
        
        # Emotion recognition variables
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.colors = {
            'Angry': 'red', 'Disgust': 'green', 'Fear': 'purple',
            'Happy': 'yellow', 'Neutral': 'gray', 'Sad': 'blue',
            'Surprise': 'orange'
        }
        
        # Camera and detection variables
        self.cap = None
        self.model = None
        self.is_running = False
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
        
        # Start update loop
        self.update_loop()
    
    def init_face_detection(self):
        """Initialize face detection"""
        try:
            # Try to find haar cascade file
            cascade_paths = [
                'haarcascade_frontalface_default.xml',
                'config/haarcascade_frontalface_default.xml',
                os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            ]
            
            for path in cascade_paths:
                if os.path.exists(path):
                    self.face_cascade = cv2.CascadeClassifier(path)
                    break
            
            if self.face_cascade is None or self.face_cascade.empty():
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            print("✅ Face detection initialized")
            
        except Exception as e:
            print(f"⚠️ Face detection error: {e}")
            self.face_cascade = None
    
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
                        print(f"✅ Model loaded: {path}")
                        return
                    except ImportError:
                        print("⚠️ TensorFlow not available, using basic detection")
                        break
                    except Exception as e:
                        print(f"⚠️ Model load error: {e}")
                        continue
            
            print("⚠️ No model loaded - using basic emotion detection")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
    
    def create_widgets(self):
        """Create simple GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="🎭 Emotion Recognition System",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Start/Stop button
        self.start_btn = tk.Button(
            control_frame,
            text="▶️ Start Camera",
            command=self.toggle_camera,
            font=("Arial", 12),
            bg='lightgreen'
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Detection toggle
        self.detection_btn = tk.Button(
            control_frame,
            text="🎯 Detection: ON",
            command=self.toggle_detection,
            font=("Arial", 12),
            bg='lightblue'
        )
        self.detection_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="Status: Ready",
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # Video frame
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = tk.Label(
            video_frame,
            text="📷 Camera feed will appear here\nClick 'Start Camera' to begin",
            font=("Arial", 14),
            bg='black',
            fg='white',
            width=60,
            height=20
        )
        self.video_label.pack(expand=True)
        
        # Emotion display frame
        emotion_frame = ttk.Frame(main_frame)
        emotion_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Current emotion
        emotion_display_frame = ttk.LabelFrame(emotion_frame, text="Current Emotion", padding="10")
        emotion_display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.emotion_label = tk.Label(
            emotion_display_frame,
            text="😐 Neutral",
            font=("Arial", 20, "bold"),
            bg='lightgray',
            width=15,
            height=3
        )
        self.emotion_label.pack()
        
        self.confidence_label = tk.Label(
            emotion_display_frame,
            text="Confidence: 0%",
            font=("Arial", 12)
        )
        self.confidence_label.pack(pady=(5, 0))
        
        # Statistics
        stats_frame = ttk.LabelFrame(emotion_frame, text="Statistics", padding="10")
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.stats_text = tk.Text(
            stats_frame,
            height=8,
            width=25,
            font=("Courier", 10)
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="System Log", padding="5")
        log_frame.pack(fill=tk.X)
        
        self.log_text = tk.Text(
            log_frame,
            height=4,
            font=("Courier", 9)
        )
        self.log_text.pack(fill=tk.X)
        
        # Add scrollbar to log
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # Keep only last 20 lines
        lines = self.log_text.get("1.0", tk.END).split('\n')
        if len(lines) > 20:
            self.log_text.delete("1.0", f"{len(lines)-20}.0")
        
        # Also print to console
        print(f"[{timestamp}] {message}")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            
            if self.cap.isOpened():
                self.is_running = True
                self.start_btn.config(text="⏸️ Stop Camera", bg='lightcoral')
                self.log_message("✅ Camera started")
            else:
                self.log_message("❌ Failed to open camera")
                messagebox.showerror("Camera Error", "Could not open camera")
                
        except Exception as e:
            self.log_message(f"❌ Camera error: {e}")
            messagebox.showerror("Camera Error", f"Camera error: {e}")
    
    def stop_camera(self):
        """Stop camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(text="▶️ Start Camera", bg='lightgreen')
        self.video_label.config(image='', text="📷 Camera stopped\nClick 'Start Camera' to begin")
        self.log_message("⏸️ Camera stopped")
    
    def toggle_detection(self):
        """Toggle emotion detection"""
        self.detection_enabled = not self.detection_enabled
        status = "ON" if self.detection_enabled else "OFF"
        self.detection_btn.config(text=f"🎯 Detection: {status}")
        self.log_message(f"🎯 Detection {status}")
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        if self.face_cascade is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        if self.model is None:
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
            print(f"Prediction error: {e}")
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
        color = self.colors.get(emotion, 'gray')
        
        self.emotion_label.config(
            text=f"{emoji} {emotion}",
            bg=color
        )
        
        self.confidence_label.config(
            text=f"Confidence: {confidence:.1%}"
        )
        
        # Update statistics
        self.emotion_counts[emotion] += 1
        self.update_statistics()
    
    def update_statistics(self):
        """Update statistics display"""
        total = sum(self.emotion_counts.values())
        
        stats_text = f"FPS: {self.current_fps}\n"
        stats_text += f"Total Detections: {total}\n\n"
        
        for emotion in self.emotions:
            count = self.emotion_counts[emotion]
            percentage = (count / total * 100) if total > 0 else 0
            stats_text += f"{emotion}: {count} ({percentage:.1f}%)\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_start >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start = time.time()
    
    def update_loop(self):
        """Main update loop"""
        if self.is_running and self.cap:
            try:
                ret, frame = self.cap.read()
                if ret:
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
                            
                            # Only update if confidence is reasonable
                            if confidence >= 0.5:
                                self.update_emotion_display(emotion, confidence)
                                
                                # Draw rectangle and label
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                
                                label = f"{emotion} ({confidence:.1%})"
                                cv2.putText(frame, label, (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Convert frame for Tkinter
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Resize for display
                    display_size = (640, 480)
                    frame_pil = frame_pil.resize(display_size, Image.Resampling.LANCZOS)
                    
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    
                    # Update video label
                    self.video_label.config(image=frame_tk, text="")
                    self.video_label.image = frame_tk  # Keep reference
                    
                    # Update status
                    self.status_label.config(
                        text=f"Status: Running - FPS: {self.current_fps} - Detection: {'ON' if self.detection_enabled else 'OFF'}"
                    )
                
            except Exception as e:
                self.log_message(f"⚠️ Frame update error: {e}")
        
        # Schedule next update
        self.root.after(50, self.update_loop)  # ~20 FPS
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.root.destroy()

def main():
    print("🎭 Starting Simple Emotion Recognition GUI...")
    
    # Create and run GUI
    root = tk.Tk()
    app = SimpleEmotionGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        app.on_closing()

if __name__ == "__main__":
    main()