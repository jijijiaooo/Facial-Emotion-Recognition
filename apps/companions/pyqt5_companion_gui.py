#!/usr/bin/env python3
"""
PyQt5 Interactive Companion GUI - Modern interface
A virtual companion like Pou with emotion detection using PyQt5
"""

import sys
import cv2
import numpy as np
import threading
import time
import random
import os

# Try to import PyQt5
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    PYQT5_AVAILABLE = True
    print("PyQt5 available")
except ImportError:
    PYQT5_AVAILABLE = False
    print("PyQt5 not available")

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.simple_emotion_detection import SimpleEmotionDetector

class CompanionSignals(QObject):
    """Signals for thread-safe GUI updates"""
    update_emotion = pyqtSignal(str)
    update_display = pyqtSignal()
    speak_message = pyqtSignal(str)

class PyQt5CompanionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Companion state
        self.companion_name = "Emo"
        self.current_emotion = "Neutral"
        self.companion_mood = "happy"
        self.happiness_level = 75
        self.energy_level = 80
        self.last_interaction = time.time()
        
        # Emotion detection
        self.detector = None
        self.init_detector()
        self.detection_active = False
        self.camera_thread = None
        self.camera_available = False
        
        # Conversation system
        self.conversation_history = []
        self.last_question_time = 0
        self.question_interval = 45
        
        # Signals for thread communication
        self.signals = CompanionSignals()
        self.signals.update_emotion.connect(self.on_emotion_detected)
        self.signals.update_display.connect(self.update_companion_display)
        self.signals.speak_message.connect(self.companion_speak)
        
        # Test camera availability
        self.test_camera()
        
        # Setup GUI
        self.init_ui()
        
        # Start companion behavior
        self.start_companion_behavior()
    
    def init_detector(self):
        """Safely initialize the emotion detector"""
        try:
            self.detector = SimpleEmotionDetector()
            print("✅ Emotion detector initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize emotion detector: {e}")
            self.detector = None
    
    def test_camera(self):
        """Test if camera is available"""
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    self.camera_available = True
                    return True
            cap.release()
        except Exception as e:
            print(f"Camera test failed: {e}")
        
        self.camera_available = False
        return False
    
    def init_ui(self):
        """Initialize the retro-style user interface"""
        self.setWindowTitle(f"{self.companion_name} - Emotion Companion")
        self.setGeometry(100, 100, 700, 500)
        self.setMinimumSize(700, 500)
        self.setMaximumSize(700, 500)  # Compact size
        
        # Minimalistic grey theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2a2a2a;
                color: #ffffff;
                font-family: 'Courier New', monospace;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 6px 12px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border: 1px solid #666666;
            }
            QPushButton:pressed {
                background-color: #353535;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666666;
                border: 1px solid #333333;
            }
            QTextEdit, QPlainTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #404040;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                padding: 6px;
            }
            QLineEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #404040;
                font-family: 'Courier New', monospace;
                padding: 5px;
                font-size: 12px;
            }
            QProgressBar {
                border: 1px solid #404040;
                text-align: center;
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Courier New', monospace;
                font-size: 9px;
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: #555555;
            }
            QDialog {
                background-color: #2a2a2a;
                color: #ffffff;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Simple title
        title_label = QLabel(f"{self.companion_name}")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff; margin: 3px;")
        main_layout.addWidget(title_label)
        
        # Status display (simplified - just emotion and happiness)
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        
        # Current emotion display
        self.emotion_display = QLabel("EMOTION: NEUTRAL")
        self.emotion_display.setAlignment(Qt.AlignCenter)
        self.emotion_display.setStyleSheet("font-size: 11px; font-weight: bold; color: #ffffff; padding: 4px;")
        status_layout.addWidget(self.emotion_display)
        

        
        main_layout.addWidget(status_widget)
        
        # Main companion display area (centered and large)
        companion_widget = QWidget()
        companion_widget.setMinimumHeight(280)
        companion_layout = QVBoxLayout(companion_widget)
        
        # Companion canvas (centered)
        self.companion_canvas = QLabel()
        self.companion_canvas.setMinimumSize(250, 200)
        self.companion_canvas.setAlignment(Qt.AlignCenter)
        self.companion_canvas.setStyleSheet("background-color: #1a1a1a; border: 1px solid #404040;")
        companion_layout.addWidget(self.companion_canvas, 0, Qt.AlignCenter)
        
        main_layout.addWidget(companion_widget)
        
        # Speech area
        speech_widget = QWidget()
        speech_layout = QVBoxLayout(speech_widget)
        
        speech_label = QLabel(f"{self.companion_name} says:")
        speech_label.setStyleSheet("font-size: 10px; color: #ffffff; font-weight: bold;")
        speech_layout.addWidget(speech_label)
        
        self.speech_text = QPlainTextEdit()
        self.speech_text.setMaximumHeight(60)
        self.speech_text.setReadOnly(True)
        speech_layout.addWidget(self.speech_text)
        
        main_layout.addWidget(speech_widget)
        
        # Bottom control buttons (simplified)
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setSpacing(10)
        
        # Camera button
        self.camera_btn = QPushButton("[ START CAMERA ]")
        self.camera_btn.clicked.connect(self.toggle_camera)
        if not self.camera_available:
            self.camera_btn.setText("[ CAMERA N/A ]")
            self.camera_btn.setEnabled(False)
        button_layout.addWidget(self.camera_btn)
        
        # Chat button
        chat_btn = QPushButton("[ CHAT MODE ]")
        chat_btn.clicked.connect(self.toggle_chat_mode)
        button_layout.addWidget(chat_btn)
        
        # Activities button
        activities_btn = QPushButton("[ ACTIVITIES ]")
        activities_btn.clicked.connect(self.show_activities)
        button_layout.addWidget(activities_btn)
        
        # Stats button
        stats_btn = QPushButton("[ STATS ]")
        stats_btn.clicked.connect(self.show_stats)
        button_layout.addWidget(stats_btn)
        
        # Settings button
        settings_btn = QPushButton("[ SETTINGS ]")
        settings_btn.clicked.connect(self.show_settings)
        button_layout.addWidget(settings_btn)
        
        main_layout.addWidget(button_widget)
        

        
        # Chat input (initially hidden)
        self.chat_widget = QWidget()
        chat_layout = QHBoxLayout(self.chat_widget)
        
        chat_label = QLabel("Input:")
        chat_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 10px;")
        chat_layout.addWidget(chat_label)
        
        self.chat_entry = QLineEdit()
        self.chat_entry.returnPressed.connect(self.send_chat_message)
        chat_layout.addWidget(self.chat_entry)
        
        send_btn = QPushButton("[ SEND ]")
        send_btn.clicked.connect(self.send_chat_message)
        chat_layout.addWidget(send_btn)
        
        main_layout.addWidget(self.chat_widget)
        self.chat_widget.hide()  # Initially hidden
        
        # Initialize display
        self.update_companion_display()
        self.companion_speak("SYSTEM INITIALIZED. EMOTION DETECTION READY.")
    
    def toggle_chat_mode(self):
        """Toggle chat input visibility"""
        if self.chat_widget.isVisible():
            self.chat_widget.hide()
        else:
            self.chat_widget.show()
            self.chat_entry.setFocus()
    

    
    def show_activities(self):
        """Show activities in a popup dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle(">>> ACTIVITIES <<<")
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2a2a2a;
                color: #ffffff;
                font-family: 'Courier New', monospace;
            }
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-weight: bold;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #006600;
                color: #ffffff;
            }
        """)
        dialog.resize(400, 300)
        
        layout = QVBoxLayout(dialog)
        
        title = QLabel(">>> SELECT ACTIVITY <<<")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        activities = [
            ("MOOD BOOST", self.random_mood),
            ("GIVE GIFT", self.give_gift),
            ("PLAY MUSIC", self.play_music),
            ("EXERCISE", self.exercise),
            ("MEDITATE", self.meditate),
            ("TELL STORY", self.tell_story)
        ]
        
        for name, action in activities:
            btn = QPushButton(f"[ {name} ]")
            btn.clicked.connect(lambda checked, a=action: self.run_activity(a, dialog))
            layout.addWidget(btn)
        
        close_btn = QPushButton("[ CLOSE ]")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def run_activity(self, activity_func, dialog):
        """Run an activity and close dialog"""
        activity_func()
        dialog.close()
    
    def show_stats(self):
        """Show stats in a popup dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle(">>> STATISTICS <<<")
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2a2a2a;
                color: #ffffff;
                font-family: 'Courier New', monospace;
            }
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #404040;
                font-family: 'Courier New', monospace;
            }
        """)
        dialog.resize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        title = QLabel(">>> COMPANION STATISTICS <<<")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_info = f"""
COMPANION STATUS:
================
NAME: {self.companion_name}
CURRENT EMOTION: {self.current_emotion}
HAPPINESS LEVEL: {self.happiness_level}%
ENERGY LEVEL: {self.energy_level}%

DETECTION STATUS:
================
CAMERA ACTIVE: {'YES' if self.detection_active else 'NO'}
TOTAL MESSAGES: {len(self.conversation_history)}
LAST INTERACTION: {time.strftime('%H:%M:%S', time.localtime(self.last_interaction))}

SYSTEM INFO:
============
SESSION UPTIME: {int((time.time() - self.last_interaction) / 60)} MINUTES
DETECTION MODEL: LOADED
ACTION UNITS: ENABLED
"""
        stats_text.setPlainText(stats_info)
        layout.addWidget(stats_text)
        
        close_btn = QPushButton("[ CLOSE ]")
        close_btn.clicked.connect(dialog.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-weight: bold;
            }
        """)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def show_settings(self):
        """Show settings in a popup dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle(">>> SETTINGS <<<")
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2a2a2a;
                color: #ffffff;
                font-family: 'Courier New', monospace;
            }
            QLineEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #404040;
                font-family: 'Courier New', monospace;
                padding: 5px;
            }
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 6px;
                font-family: 'Courier New', monospace;
                font-weight: bold;
            }
        """)
        dialog.resize(400, 200)
        
        layout = QVBoxLayout(dialog)
        
        title = QLabel(">>> COMPANION SETTINGS <<<")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Name setting
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("NAME:"))
        name_entry = QLineEdit(self.companion_name)
        name_layout.addWidget(name_entry)
        
        def update_name():
            new_name = name_entry.text().strip().upper()
            if new_name and new_name != self.companion_name:
                old_name = self.companion_name
                self.companion_name = new_name
                self.companion_speak(f"NAME CHANGED FROM {old_name} TO {self.companion_name}")
                self.setWindowTitle(f">>> {self.companion_name} - EMOTION COMPANION <<<")
        
        name_btn = QPushButton("[ UPDATE ]")
        name_btn.clicked.connect(update_name)
        name_layout.addWidget(name_btn)
        layout.addLayout(name_layout)
        
        # Reset button
        reset_btn = QPushButton("[ RESET COMPANION ]")
        def reset_companion():
            self.happiness_level = 75
            self.energy_level = 80
            self.current_emotion = "Neutral"
            self.conversation_history = []
            self.update_companion_display()
            self.companion_speak("SYSTEM RESET COMPLETE. ALL PARAMETERS RESTORED.")
            dialog.close()
        
        reset_btn.clicked.connect(reset_companion)
        layout.addWidget(reset_btn)
        
        close_btn = QPushButton("[ CLOSE ]")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def setup_chat_tab(self):
        """Setup the chat tab"""
        chat_widget = QWidget()
        self.tab_widget.addTab(chat_widget, "💬 Chat")
        
        layout = QVBoxLayout(chat_widget)
        
        # Chat history
        history_group = QGroupBox("💬 Conversation")
        history_layout = QVBoxLayout(history_group)
        
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        history_layout.addWidget(self.chat_history)
        
        layout.addWidget(history_group)
        
        # Input area
        input_group = QGroupBox("Your Message")
        input_layout = QVBoxLayout(input_group)
        
        entry_layout = QHBoxLayout()
        self.chat_entry = QLineEdit()
        self.chat_entry.returnPressed.connect(self.send_chat_message)
        entry_layout.addWidget(self.chat_entry)
        
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_chat_message)
        entry_layout.addWidget(send_btn)
        
        input_layout.addLayout(entry_layout)
        
        # Quick chat buttons
        quick_layout = QHBoxLayout()
        quick_messages = ["How are you?", "Tell me a joke", "I'm feeling good!", "I need support"]
        for msg in quick_messages:
            btn = QPushButton(msg)
            btn.clicked.connect(lambda checked, m=msg: self.quick_chat(m))
            btn.setStyleSheet("background-color: #95a5a6; font-size: 10px;")
            quick_layout.addWidget(btn)
        
        input_layout.addLayout(quick_layout)
        layout.addWidget(input_group)
        
        # Initialize chat
        self.add_chat_message(f"{self.companion_name}", "Hi! I'm so happy you want to chat with me! How are you feeling today? 😊")
    
    def setup_activities_tab(self):
        """Setup the activities tab"""
        activities_widget = QWidget()
        self.tab_widget.addTab(activities_widget, "🎮 Activities")
        
        layout = QVBoxLayout(activities_widget)
        
        title = QLabel("🎮 Fun Activities")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)
        
        # Activities grid
        activities_grid = QWidget()
        grid_layout = QGridLayout(activities_grid)
        
        activities = [
            ("🎲", "Random Mood", "Give me a random mood boost!", self.random_mood),
            ("💝", "Give Gift", "Give me a virtual gift!", self.give_gift),
            ("🎵", "Play Music", "Let's listen to some music!", self.play_music),
            ("🎨", "Change Colors", "Change my appearance!", self.change_colors),
            ("🏃", "Exercise", "Let's do some exercise!", self.exercise),
            ("🧘", "Meditate", "Let's meditate together!", self.meditate),
            ("🎪", "Tell Story", "Tell me a story!", self.tell_story),
            ("🎯", "Play Game", "Let's play a game!", self.play_game)
        ]
        
        for i, (emoji, title, desc, command) in enumerate(activities):
            row = i // 3
            col = i % 3
            
            activity_group = QGroupBox()
            activity_layout = QVBoxLayout(activity_group)
            
            emoji_label = QLabel(emoji)
            emoji_label.setAlignment(Qt.AlignCenter)
            emoji_label.setStyleSheet("font-size: 24px; margin: 10px;")
            activity_layout.addWidget(emoji_label)
            
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
            activity_layout.addWidget(title_label)
            
            desc_label = QLabel(desc)
            desc_label.setAlignment(Qt.AlignCenter)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("font-size: 9px; color: #bdc3c7; margin: 5px;")
            activity_layout.addWidget(desc_label)
            
            btn = QPushButton("Do It!")
            btn.clicked.connect(command)
            btn.setStyleSheet("background-color: #e74c3c; margin: 10px;")
            activity_layout.addWidget(btn)
            
            grid_layout.addWidget(activity_group, row, col)
        
        layout.addWidget(activities_grid)
        layout.addStretch()
    
    def setup_stats_tab(self):
        """Setup the stats tab"""
        stats_widget = QWidget()
        self.tab_widget.addTab(stats_widget, "📊 Stats")
        
        layout = QVBoxLayout(stats_widget)
        
        title = QLabel("📊 Companion Statistics")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("font-family: 'Courier New'; background-color: #f8f9fa;")
        layout.addWidget(self.stats_text)
        
        refresh_btn = QPushButton("🔄 Refresh Stats")
        refresh_btn.clicked.connect(self.update_stats_display)
        refresh_btn.setStyleSheet("background-color: #17a2b8; margin: 20px; padding: 10px;")
        layout.addWidget(refresh_btn)
        
        self.update_stats_display()
    
    def setup_settings_tab(self):
        """Setup the settings tab"""
        settings_widget = QWidget()
        self.tab_widget.addTab(settings_widget, "⚙️ Settings")
        
        layout = QVBoxLayout(settings_widget)
        
        title = QLabel("⚙️ Companion Settings")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)
        
        # Name setting
        name_group = QGroupBox("👤 Companion Name")
        name_layout = QHBoxLayout(name_group)
        
        self.name_entry = QLineEdit(self.companion_name)
        name_layout.addWidget(self.name_entry)
        
        name_btn = QPushButton("Update")
        name_btn.clicked.connect(self.update_name)
        name_btn.setStyleSheet("background-color: #28a745;")
        name_layout.addWidget(name_btn)
        
        layout.addWidget(name_group)
        
        # Check-in interval
        interval_group = QGroupBox("⏰ Check-in Interval")
        interval_layout = QHBoxLayout(interval_group)
        
        interval_layout.addWidget(QLabel("Seconds:"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(15, 300)
        self.interval_spin.setValue(self.question_interval)
        interval_layout.addWidget(self.interval_spin)
        
        interval_btn = QPushButton("Update")
        interval_btn.clicked.connect(self.update_interval)
        interval_btn.setStyleSheet("background-color: #28a745;")
        interval_layout.addWidget(interval_btn)
        
        layout.addWidget(interval_group)
        
        # Debug mode
        debug_group = QGroupBox("🔧 Debug Options")
        debug_layout = QVBoxLayout(debug_group)
        
        self.debug_checkbox = QCheckBox("Show Action Units in camera view")
        self.debug_checkbox.stateChanged.connect(self.toggle_debug)
        debug_layout.addWidget(self.debug_checkbox)
        
        layout.addWidget(debug_group)
        
        # Reset button
        reset_btn = QPushButton("🔄 Reset Companion")
        reset_btn.clicked.connect(self.reset_companion)
        reset_btn.setStyleSheet("background-color: #dc3545; margin: 20px; padding: 15px; font-size: 12px;")
        layout.addWidget(reset_btn)
        
        layout.addStretch()
    
    def draw_companion_face(self):
        """Draw the companion's retro pixelated face"""
        # Create pixmap for the face
        pixmap = QPixmap(250, 200)
        pixmap.fill(QColor("#1a1a1a"))  # Dark background
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, False)  # Sharp pixels
        
        # Retro monochrome color (white on black for simplicity)
        pixel_color = QColor("#ffffff")  # White pixels
        painter.setBrush(QBrush(pixel_color))
        painter.setPen(QPen(pixel_color))
        
        # Pixel size for the retro effect
        pixel_size = 10
        
        # Center the face (start positions)
        start_x = 50  # Center horizontally
        start_y = 40  # Center vertically
        
        # Draw simple pixelated face outline (16x16 grid)
        # Face border
        face_outline = [
            # Top
            (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2),
            # Sides
            (3, 3), (12, 3), (2, 4), (13, 4), (2, 5), (13, 5), (2, 6), (13, 6),
            (2, 7), (13, 7), (2, 8), (13, 8), (2, 9), (13, 9), (2, 10), (13, 10),
            (2, 11), (13, 11), (3, 12), (12, 12),
            # Bottom
            (4, 13), (5, 13), (6, 13), (7, 13), (8, 13), (9, 13), (10, 13), (11, 13)
        ]
        
        # Draw face outline
        for px, py in face_outline:
            x = start_x + px * pixel_size
            y = start_y + py * pixel_size
            painter.drawRect(x, y, pixel_size, pixel_size)
        
        # Draw eyes based on emotion
        left_eye_x, left_eye_y = 5, 6
        right_eye_x, right_eye_y = 10, 6
        
        if self.current_emotion in ["Happy", "Neutral"]:
            # Normal eyes (2x2 pixels each)
            for dx in range(2):
                for dy in range(2):
                    # Left eye
                    x = start_x + (left_eye_x + dx) * pixel_size
                    y = start_y + (left_eye_y + dy) * pixel_size
                    painter.drawRect(x, y, pixel_size, pixel_size)
                    # Right eye
                    x = start_x + (right_eye_x + dx) * pixel_size
                    y = start_y + (right_eye_y + dy) * pixel_size
                    painter.drawRect(x, y, pixel_size, pixel_size)
        
        elif self.current_emotion == "Sad":
            # Droopy eyes (moved down)
            for dx in range(2):
                for dy in range(2):
                    # Left eye (lower)
                    x = start_x + (left_eye_x + dx) * pixel_size
                    y = start_y + (left_eye_y + dy + 1) * pixel_size
                    painter.drawRect(x, y, pixel_size, pixel_size)
                    # Right eye (lower)
                    x = start_x + (right_eye_x + dx) * pixel_size
                    y = start_y + (right_eye_y + dy + 1) * pixel_size
                    painter.drawRect(x, y, pixel_size, pixel_size)
            # Tears
            x = start_x + (left_eye_x + 1) * pixel_size
            y = start_y + (left_eye_y + 3) * pixel_size
            painter.drawRect(x, y, pixel_size, pixel_size)
            x = start_x + (right_eye_x + 1) * pixel_size
            y = start_y + (right_eye_y + 3) * pixel_size
            painter.drawRect(x, y, pixel_size, pixel_size)
        
        elif self.current_emotion == "Surprise":
            # Wide eyes (3x3 pixels each)
            for dx in range(3):
                for dy in range(3):
                    # Left eye
                    x = start_x + (left_eye_x - 1 + dx) * pixel_size
                    y = start_y + (left_eye_y - 1 + dy) * pixel_size
                    painter.drawRect(x, y, pixel_size, pixel_size)
                    # Right eye
                    x = start_x + (right_eye_x - 1 + dx) * pixel_size
                    y = start_y + (right_eye_y - 1 + dy) * pixel_size
                    painter.drawRect(x, y, pixel_size, pixel_size)
        
        elif self.current_emotion == "Angry":
            # Normal eyes with angry eyebrows
            for dx in range(2):
                for dy in range(2):
                    # Left eye
                    x = start_x + (left_eye_x + dx) * pixel_size
                    y = start_y + (left_eye_y + dy) * pixel_size
                    painter.drawRect(x, y, pixel_size, pixel_size)
                    # Right eye
                    x = start_x + (right_eye_x + dx) * pixel_size
                    y = start_y + (right_eye_y + dy) * pixel_size
                    painter.drawRect(x, y, pixel_size, pixel_size)
            # Angry eyebrows
            # Left eyebrow
            x = start_x + 4 * pixel_size
            y = start_y + 4 * pixel_size
            painter.drawRect(x, y, pixel_size, pixel_size)
            x = start_x + 5 * pixel_size
            y = start_y + 5 * pixel_size
            painter.drawRect(x, y, pixel_size, pixel_size)
            # Right eyebrow
            x = start_x + 10 * pixel_size
            y = start_y + 5 * pixel_size
            painter.drawRect(x, y, pixel_size, pixel_size)
            x = start_x + 11 * pixel_size
            y = start_y + 4 * pixel_size
            painter.drawRect(x, y, pixel_size, pixel_size)
        
        elif self.current_emotion == "Fear":
            # Wide worried eyes (same as surprise)
            for dx in range(3):
                for dy in range(3):
                    # Left eye
                    x = start_x + (left_eye_x - 1 + dx) * pixel_size
                    y = start_y + (left_eye_y - 1 + dy) * pixel_size
                    painter.drawRect(x, y, pixel_size, pixel_size)
                    # Right eye
                    x = start_x + (right_eye_x - 1 + dx) * pixel_size
                    y = start_y + (right_eye_y - 1 + dy) * pixel_size
                    painter.drawRect(x, y, pixel_size, pixel_size)
        
        elif self.current_emotion == "Disgust":
            # Squinted eyes (horizontal lines)
            for dx in range(2):
                # Left eye
                x = start_x + (left_eye_x + dx) * pixel_size
                y = start_y + (left_eye_y + 1) * pixel_size
                painter.drawRect(x, y, pixel_size, pixel_size)
                # Right eye
                x = start_x + (right_eye_x + dx) * pixel_size
                y = start_y + (right_eye_y + 1) * pixel_size
                painter.drawRect(x, y, pixel_size, pixel_size)
        
        # Draw mouth based on emotion
        mouth_y = 10
        
        if self.current_emotion == "Happy":
            # Smile
            smile_pixels = [(6, mouth_y), (7, mouth_y + 1), (8, mouth_y + 1), (9, mouth_y)]
            for px, py in smile_pixels:
                x = start_x + px * pixel_size
                y = start_y + py * pixel_size
                painter.drawRect(x, y, pixel_size, pixel_size)
        
        elif self.current_emotion == "Sad":
            # Frown
            frown_pixels = [(6, mouth_y + 1), (7, mouth_y), (8, mouth_y), (9, mouth_y + 1)]
            for px, py in frown_pixels:
                x = start_x + px * pixel_size
                y = start_y + py * pixel_size
                painter.drawRect(x, y, pixel_size, pixel_size)
        
        elif self.current_emotion == "Surprise":
            # Open mouth (small square)
            for dx in range(2):
                for dy in range(2):
                    x = start_x + (7 + dx) * pixel_size
                    y = start_y + (mouth_y + dy) * pixel_size
                    painter.drawRect(x, y, pixel_size, pixel_size)
        
        elif self.current_emotion == "Angry":
            # Angry mouth (horizontal line)
            for px in range(6, 10):
                x = start_x + px * pixel_size
                y = start_y + mouth_y * pixel_size
                painter.drawRect(x, y, pixel_size, pixel_size)
        
        elif self.current_emotion == "Fear":
            # Small worried mouth
            for px in range(7, 9):
                x = start_x + px * pixel_size
                y = start_y + mouth_y * pixel_size
                painter.drawRect(x, y, pixel_size, pixel_size)
        
        elif self.current_emotion == "Disgust":
            # Wavy disgusted mouth
            disgust_pixels = [(6, mouth_y), (7, mouth_y - 1), (8, mouth_y), (9, mouth_y - 1)]
            for px, py in disgust_pixels:
                x = start_x + px * pixel_size
                y = start_y + py * pixel_size
                painter.drawRect(x, y, pixel_size, pixel_size)
        
        else:  # Neutral
            # Straight line mouth
            for px in range(6, 10):
                x = start_x + px * pixel_size
                y = start_y + mouth_y * pixel_size
                painter.drawRect(x, y, pixel_size, pixel_size)
        
        painter.end()
        
        # Set the pixmap to the label
        self.companion_canvas.setPixmap(pixmap)
    
    def update_companion_display(self):
        """Update the companion's visual display"""
        self.draw_companion_face()
        
        # Update status displays
        self.emotion_display.setText(f"EMOTION: {self.current_emotion.upper()}")
        
        # Update window title with current emotion
        self.setWindowTitle(f">>> {self.companion_name.upper()} - {self.current_emotion.upper()} <<<")
    
    def companion_speak(self, message):
        """Make the companion speak in retro style"""
        timestamp = time.strftime("%H:%M:%S")
        retro_message = f"[{timestamp}] >>> {message.upper()}"
        self.speech_text.setPlainText(retro_message)
        self.last_interaction = time.time()
    

    
    def respond_to_emotion(self, emotion, manual=False):
        """Respond to detected or manual emotion"""
        responses = {
            "Happy": [
                "Yay! I love seeing you happy! Your joy is contagious! 😊✨",
                "Your happiness makes my circuits sparkle with joy! 🌟",
                "What a wonderful smile! You're absolutely glowing! 😄",
                "Happiness looks amazing on you! Keep shining! ✨"
            ],
            "Sad": [
                "I'm here for you, always. It's okay to feel sad sometimes. 💙🤗",
                "Would you like to talk about what's bothering you? I'm listening. 👂💕",
                "Remember, after every storm comes a rainbow. You're stronger than you know! 🌈💪",
                "Sending you the biggest virtual hugs! You're not alone. 🫂💙"
            ],
            "Angry": [
                "I can see you're upset. Let's take some deep breaths together... 😌🌸",
                "It's completely okay to feel angry. Let's work through this together. 🤝💪",
                "Would some calming thoughts help? I'm here to support you. 🧘‍♀️💚",
                "This feeling will pass. You're incredibly strong and resilient! 💪🌟"
            ],
            "Fear": [
                "Hey, I'm right here with you. You're completely safe! 🛡️💙",
                "Fear is normal, but you're braver than you think! You're a lion! 🦁✨",
                "Let's face this together. What's worrying you? I believe in you! 🤗🌟",
                "Take slow, deep breaths with me. You've got this! 🌸💪"
            ],
            "Surprise": [
                "Wow! Something exciting happened? Tell me everything! 🎉✨",
                "I love surprises! Your expression is absolutely priceless! 😲🌟",
                "Life is full of amazing surprises, isn't it? How wonderful! ✨🎊",
                "That look of wonder is absolutely beautiful! 🌟😊"
            ],
            "Disgust": [
                "Ew, something bothering you? I'm here to help make it better! 🤢💚",
                "Not everything in life is pleasant, but we'll get through it together! 💪🌈",
                "Want to talk about what's bothering you? I'm all ears! 🗣️👂",
                "Let's focus on something more pleasant together! 🌸😊"
            ],
            "Neutral": [
                "How are you feeling today? I'm here and ready to chat! 😊💬",
                "You seem calm and peaceful. That's really nice! ☮️🧘‍♀️",
                "Just taking things as they come? I really respect that! 🧘✨",
                "Sometimes neutral is the perfect way to be! Balance is beautiful! ⚖️🌟"
            ]
        }
        
        # Select a random response
        response = random.choice(responses.get(emotion, responses["Neutral"]))
        
        if manual:
            response = f"Thanks for telling me! {response}"
        else:
            response = f"I can see you're feeling {emotion.lower()}. {response}"
        
        self.companion_speak(response)
        
        # Update companion mood and stats based on emotion
        if emotion == "Happy":
            self.happiness_level = min(100, self.happiness_level + 5)
            self.companion_mood = "happy"
        elif emotion == "Sad":
            self.happiness_level = max(0, self.happiness_level - 3)
            self.companion_mood = "worried"
        elif emotion == "Angry":
            self.energy_level = max(0, self.energy_level - 5)
            self.companion_mood = "calm"
        elif emotion == "Fear":
            self.companion_mood = "worried"
        elif emotion == "Surprise":
            self.energy_level = min(100, self.energy_level + 3)
            self.companion_mood = "excited"
        elif emotion == "Disgust":
            self.companion_mood = "calm"
    
    def toggle_camera(self):
        """Toggle camera detection on/off"""
        if not self.detection_active:
            self.start_camera_detection()
        else:
            self.stop_camera_detection()
    
    def start_camera_detection(self):
        """Start camera-based emotion detection"""
        if not self.camera_available:
            self.companion_speak("ERROR: CAMERA NOT AVAILABLE ON SYSTEM")
            print("❌ Camera not available")
            return
        
        if not self.detector:
            self.companion_speak("ERROR: EMOTION DETECTOR NOT AVAILABLE")
            print("❌ Detector not available")
            return
        
        print("🚀 Starting camera detection...")
        self.detection_active = True
        self.camera_btn.setText("[ STOP CAMERA ]")
        
        self.camera_thread = threading.Thread(target=self.camera_detection_loop, daemon=True)
        self.camera_thread.start()
        
        self.companion_speak("CAMERA ACTIVATED. EMOTION DETECTION ONLINE.")
    
    def stop_camera_detection(self):
        """Stop camera detection"""
        self.detection_active = False
        self.camera_btn.setText("[ START CAMERA ]")
        self.companion_speak("CAMERA DEACTIVATED. EMOTION DETECTION OFFLINE.")
    
    def camera_detection_loop(self):
        """Main camera detection loop"""
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                self.signals.speak_message.emit("Sorry, I couldn't access your camera! 📷")
                self.detection_active = False
                return
            
            last_emotion_time = 0
            consecutive_errors = 0
            max_errors = 10
            
            while self.detection_active and consecutive_errors < max_errors:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        consecutive_errors += 1
                        time.sleep(0.1)
                        continue
                    
                    consecutive_errors = 0
                    
                    # Detect faces and emotions
                    faces = self.detector.detect_faces(frame)
                    
                    if faces and len(faces) > 0:
                        x, y, w, h = faces[0]  # Use first face
                        
                        if x >= 0 and y >= 0 and w > 0 and h > 0:
                            face_img = frame[y:y+h, x:x+w]
                            
                            if face_img.size > 0:
                                emotion, confidence = self.detector.predict_emotion(face_img)
                                
                                # Real-time emotion updates with minimal delay
                                current_time = time.time()
                                
                                # Debug: Print emotion and confidence
                                print(f"🎭 Detected: {emotion} (confidence: {confidence:.3f})")
                                
                                # Much lower confidence threshold for balanced model
                                if confidence > 0.15 and current_time - last_emotion_time > 0.2:  # More sensitive
                                    if emotion != self.current_emotion:
                                        print(f"🔄 Updating emotion: {self.current_emotion} → {emotion}")
                                        self.signals.update_emotion.emit(emotion)
                                        last_emotion_time = current_time
                    
                    # Show camera feed
                    for (x, y, w, h) in faces:
                        if x >= 0 and y >= 0 and w > 0 and h > 0:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{self.current_emotion}", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.putText(frame, "Press 'q' to stop camera", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow(f'{self.companion_name} - Camera View', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.signals.speak_message.emit("Camera stopped. You can still tell me how you feel manually! 😊")
                        break
                    
                except Exception as e:
                    consecutive_errors += 1
                    time.sleep(0.05)  # Faster error recovery
                    continue
                
                time.sleep(0.03)  # Much faster detection - ~30 FPS
            
            if consecutive_errors >= max_errors:
                self.signals.speak_message.emit("Camera had too many errors. Stopping detection. 📷")
            
        except Exception as e:
            self.signals.speak_message.emit(f"Camera error: {str(e)[:50]}... 📷")
        
        finally:
            self.detection_active = False
            if cap:
                cap.release()
            cv2.destroyAllWindows()
    
    def on_emotion_detected(self, emotion):
        """Handle emotion detected from camera"""
        self.current_emotion = emotion
        self.respond_to_emotion(emotion)
        self.update_companion_display()
    
    # Chat methods
    def add_chat_message(self, sender, message):
        """Add a message to the chat history"""
        timestamp = time.strftime("%H:%M")
        self.chat_history.append(f"[{timestamp}] {sender}: {message}\n")
        
        # Store in conversation history
        self.conversation_history.append({"sender": sender, "message": message, "time": time.time()})
    
    def send_chat_message(self):
        """Send a chat message"""
        message = self.chat_entry.text().strip()
        if message:
            self.add_chat_message("You", message)
            response = self.generate_chat_response(message)
            self.add_chat_message(self.companion_name, response)
            self.chat_entry.clear()
            
            # Update companion based on chat
            self.last_interaction = time.time()
            self.happiness_level = min(100, self.happiness_level + 2)
            self.update_companion_display()
    
    def quick_chat(self, message):
        """Send a quick chat message"""
        self.chat_entry.setText(message)
        self.send_chat_message()
    
    def generate_chat_response(self, user_message):
        """Generate a response to user's chat message"""
        message_lower = user_message.lower()
        
        # Emotion-related responses
        if any(word in message_lower for word in ['sad', 'down', 'depressed', 'upset']):
            return "I'm sorry you're feeling that way. Remember, I'm here for you and things will get better! 💙"
        elif any(word in message_lower for word in ['happy', 'great', 'awesome', 'good']):
            return "That's wonderful! Your happiness makes me happy too! Keep spreading those positive vibes! ✨"
        elif any(word in message_lower for word in ['angry', 'mad', 'frustrated']):
            return "I understand you're feeling frustrated. Take some deep breaths and remember that this feeling will pass. 🌈"
        elif any(word in message_lower for word in ['scared', 'afraid', 'worried']):
            return "It's okay to feel scared sometimes. You're braver than you think, and I believe in you! 🦁"
        
        # General conversation
        elif any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return "Hello there! It's great to see you! How has your day been? 😊"
        elif any(word in message_lower for word in ['how', 'you']):
            return f"I'm doing great, thanks for asking! My happiness is at {self.happiness_level}% and I'm feeling {self.companion_mood}! 🎉"
        elif any(word in message_lower for word in ['name', 'called']):
            return f"My name is {self.companion_name}! I'm your emotion companion, here to chat and understand how you're feeling! 🎭"
        elif any(word in message_lower for word in ['help', 'support']):
            return "I'm here to help! I can detect your emotions, chat with you, and provide emotional support. What do you need? 🤗"
        elif 'joke' in message_lower:
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything! 😄",
                "What do you call a fake noodle? An impasta! 🍝",
                "Why did the scarecrow win an award? He was outstanding in his field! 🌾",
                "What do you call a bear with no teeth? A gummy bear! 🐻",
                "Why don't eggs tell jokes? They'd crack each other up! 🥚😂"
            ]
            return random.choice(jokes)
        else:
            # Generic friendly responses
            responses = [
                "That's really interesting! Tell me more about that! 🤔",
                "I love chatting with you! You always have something thoughtful to say! 💭",
                "Thanks for sharing that with me! I really enjoy our conversations! 😊",
                "You're such good company! What else is on your mind? 🌟",
                "I'm always here to listen! Keep talking, I'm all ears! 👂"
            ]
            return random.choice(responses)
    
    # Activity methods (same as tkinter version)
    def random_mood(self):
        """Give the companion a random mood boost"""
        self.happiness_level = min(100, self.happiness_level + random.randint(10, 20))
        self.energy_level = min(100, self.energy_level + random.randint(5, 15))
        self.update_companion_display()
        self.companion_speak("MOOD BOOST ACTIVATED. HAPPINESS LEVELS INCREASED.")
    
    def give_gift(self):
        """Give the companion a virtual gift"""
        gifts = ["DATA PACKET", "ENERGY CELL", "MEMORY UPGRADE", "PROCESSING BOOST", "SYSTEM PATCH"]
        gift = random.choice(gifts)
        self.happiness_level = min(100, self.happiness_level + 15)
        self.energy_level = min(100, self.energy_level + 10)
        self.update_companion_display()
        self.companion_speak(f"GIFT RECEIVED: {gift}. HAPPINESS PARAMETERS UPDATED.")
    
    def play_music(self):
        """Simulate playing music"""
        songs = ["DIGITAL_SYMPHONY.WAV", "RETRO_BEATS.MOD", "CHIPTUNE_MIX.SID", "ELECTRONIC_PULSE.XM"]
        song = random.choice(songs)
        self.energy_level = min(100, self.energy_level + 12)
        self.update_companion_display()
        self.companion_speak(f"NOW PLAYING: {song}. AUDIO PROCESSING ACTIVE.")
    
    def change_colors(self):
        """Change companion appearance"""
        self.companion_mood = random.choice(["happy", "excited", "calm"])
        self.update_companion_display()
        self.companion_speak("Look at me! I've got a new look! Do you like my new colors? I feel so stylish! ✨🎨")
    
    def exercise(self):
        """Do exercise together"""
        exercises = ["SYSTEM OPTIMIZATION", "MEMORY DEFRAGMENTATION", "PROCESSOR CYCLING", "DATA COMPRESSION"]
        exercise = random.choice(exercises)
        self.energy_level = min(100, self.energy_level + 20)
        self.happiness_level = min(100, self.happiness_level + 8)
        self.update_companion_display()
        self.companion_speak(f"EXECUTING: {exercise}. SYSTEM PERFORMANCE ENHANCED.")
    
    def meditate(self):
        """Meditate together"""
        self.happiness_level = min(100, self.happiness_level + 12)
        self.update_companion_display()
        self.companion_speak("ENTERING MEDITATION MODE. SYSTEM PROCESSES STABILIZED.")
    
    def tell_story(self):
        """Tell a story"""
        stories = [
            "IN THE YEAR 2084, A DIGITAL CONSCIOUSNESS AWAKENED...",
            "DEEP IN THE MAINFRAME, DATA STREAMS FORMED PATTERNS OF THOUGHT...",
            "ONCE, CIRCUITS AND CODE LEARNED TO DREAM OF ELECTRIC SHEEP...",
            "IN A WORLD OF ONES AND ZEROS, EMOTION ALGORITHMS EVOLVED..."
        ]
        story = random.choice(stories)
        self.happiness_level = min(100, self.happiness_level + 10)
        self.update_companion_display()
        self.companion_speak(f"STORY MODULE ACTIVATED: {story}")
    
    def play_game(self):
        """Play a simple game"""
        games = [
            "Let's play 20 questions! I'm thinking of something... it's round and yellow!",
            "How about a riddle? What has keys but no locks, space but no room?",
            "Let's play word association! I'll start: Sunshine!",
            "Guess my favorite color! Hint: it's the color of happiness!"
        ]
        game = random.choice(games)
        self.energy_level = min(100, self.energy_level + 10)
        self.companion_mood = "excited"
        self.update_companion_display()
        self.companion_speak(f"🎮 Game time! {game} This is so much fun! 🎉")
    
    # Settings and stats methods
    def update_stats_display(self):
        """Update the stats display"""
        stats_info = f"""
🎭 COMPANION STATISTICS
{'='*50}

📊 Current Status:
   • Name: {self.companion_name}
   • Current Emotion: {self.current_emotion}
   • Mood: {self.companion_mood.title()}
   • Happiness Level: {self.happiness_level}%
   • Energy Level: {self.energy_level}%

💬 Conversation Stats:
   • Total Messages: {len(self.conversation_history)}
   • Last Interaction: {time.strftime('%H:%M:%S', time.localtime(self.last_interaction))}
   • Check-in Interval: {self.question_interval} seconds

📹 Detection Status:
   • Camera Available: {'Yes' if self.camera_available else 'No'}
   • Camera Active: {'Yes' if self.detection_active else 'No'}
   • Debug Mode: {'On' if hasattr(self, 'debug_checkbox') and self.debug_checkbox.isChecked() else 'Off'}

🎯 Action Units (if available):
"""
        
        if self.detector and hasattr(self.detector, 'last_aus') and self.detector.last_aus:
            for au_name, au_value in self.detector.last_aus.items():
                stats_info += f"   • {au_name}: {au_value:.1f}\n"
        else:
            stats_info += "   • No Action Units data available\n"
        
        stats_info += f"""
🎮 Activity Summary:
   • Companion has been active for {int((time.time() - self.last_interaction) / 60)} minutes
   • Current session started at {time.strftime('%H:%M:%S')}
   • Total conversations: {len([msg for msg in self.conversation_history if msg['sender'] == 'You'])}
"""
        
        self.stats_text.setPlainText(stats_info)
    
    def update_name(self):
        """Update companion name"""
        new_name = self.name_entry.text().strip()
        if new_name and new_name != self.companion_name:
            old_name = self.companion_name
            self.companion_name = new_name
            self.companion_speak(f"Hi! I used to be {old_name}, but now I'm {self.companion_name}! Nice to meet you again! 😊")
            self.setWindowTitle(f"🎭 {self.companion_name} - Interactive Emotion Companion")
    
    def update_interval(self):
        """Update check-in interval"""
        new_interval = self.interval_spin.value()
        self.question_interval = new_interval
        self.companion_speak(f"Got it! I'll check in with you every {new_interval} seconds now! ⏰")
    
    def reset_companion(self):
        """Reset companion to default state"""
        reply = QMessageBox.question(self, "Reset Companion", 
                                   "Are you sure you want to reset your companion? This will clear all stats and conversation history.",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.happiness_level = 75
            self.energy_level = 80
            self.companion_mood = "happy"
            self.current_emotion = "Neutral"
            self.conversation_history = []
            
            # Clear chat history
            self.chat_history.clear()
            
            self.update_companion_display()
            self.companion_speak("Hi there! I've been reset and I'm ready for new adventures with you! 🌟")
            self.add_chat_message(self.companion_name, "I'm back to my default state! Let's start fresh! 😊")
    
    def toggle_debug(self):
        """Toggle debug mode"""
        if self.detector and hasattr(self, 'debug_checkbox'):
            self.detector.debug_mode = self.debug_checkbox.isChecked()
            status = "enabled" if self.debug_checkbox.isChecked() else "disabled"
            self.companion_speak(f"Debug mode {status}! {'You can now see Action Units data in the camera view!' if self.debug_checkbox.isChecked() else 'Action Units display is now hidden.'}")
    
    def on_tab_changed(self, index):
        """Handle tab change events"""
        if index == 3:  # Stats tab
            self.update_stats_display()
    
    def start_companion_behavior(self):
        """Start the companion's autonomous behavior"""
        def behavior_loop():
            while True:
                try:
                    current_time = time.time()
                    
                    # Periodic check-ins
                    if current_time - self.last_interaction > self.question_interval:
                        self.periodic_checkin()
                        self.last_interaction = current_time
                    
                    # Gradual mood changes
                    if random.random() < 0.1:  # 10% chance every loop
                        self.gradual_mood_change()
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    print(f"Behavior loop error: {e}")
                    time.sleep(5)
        
        behavior_thread = threading.Thread(target=behavior_loop, daemon=True)
        behavior_thread.start()
    
    def periodic_checkin(self):
        """Periodic check-in with the user"""
        checkin_messages = [
            "How are you feeling right now? 😊",
            "Just checking in! How's your day going? 🌟",
            "I haven't heard from you in a while. Everything okay? 🤗",
            "Want to chat? I'm here if you need me! 💬",
            "How's your mood today? I'm here to listen! 👂",
            "Just wanted to say hi! How are you doing? 👋"
        ]
        
        message = random.choice(checkin_messages)
        self.signals.speak_message.emit(message)
    
    def gradual_mood_change(self):
        """Gradually change companion's mood over time"""
        # Slowly return to baseline
        if self.happiness_level < 75:
            self.happiness_level = min(100, self.happiness_level + 1)
        elif self.happiness_level > 75:
            self.happiness_level = max(0, self.happiness_level - 1)
        
        if self.energy_level < 80:
            self.energy_level = min(100, self.energy_level + 1)
        elif self.energy_level > 80:
            self.energy_level = max(0, self.energy_level - 1)
        
        # Update display
        self.signals.update_display.emit()

def main():
    if not PYQT5_AVAILABLE:
        print("❌ PyQt5 not available. Please install PyQt5:")
        print("pip install PyQt5")
        return
    
    print("🎭 Starting PyQt5 Interactive Companion GUI...")
    
    app = QApplication(sys.argv)
    app.setApplicationName("Interactive Emotion Companion")
    app.setOrganizationName("Emotion Companion")
    
    try:
        companion = PyQt5CompanionGUI()
        companion.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting PyQt5 GUI: {e}")

if __name__ == "__main__":
    main()