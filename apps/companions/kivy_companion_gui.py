#!/usr/bin/env python3
"""
Kivy Interactive Companion GUI - Modern interface
A virtual companion like Pou with emotion detection using Kivy
"""

import sys
import cv2
import numpy as np
import threading
import time
import random
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Try to import emotion detection
try:
    from src.core.simple_emotion_detection import SimpleEmotionDetector
    EMOTION_DETECTION_AVAILABLE = True
    print("✅ Emotion detection available")
except ImportError:
    EMOTION_DETECTION_AVAILABLE = False
    print("⚠️ Emotion detection not available")
    SimpleEmotionDetector = None

# Kivy imports
try:
    import kivy
    kivy.require('2.0.0')
    from kivy.app import App
    from kivy.uix.widget import Widget
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.textinput import TextInput
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.popup import Popup
    from kivy.uix.progressbar import ProgressBar
    from kivy.clock import Clock
    from kivy.graphics import Color, Rectangle, Line
    from kivy.graphics.instructions import Canvas
    from kivy.core.window import Window
    from kivy.utils import get_color_from_hex
    KIVY_AVAILABLE = True
    print("✅ Kivy available")
except ImportError:
    KIVY_AVAILABLE = False
    print("❌ Kivy not available")

class CompanionCanvas(Widget):
    """Custom widget for drawing the companion's pixelated face"""
    
    def __init__(self, **kwargs):
        super(CompanionCanvas, self).__init__(**kwargs)
        self.emotion = "Neutral"
        self.bind(pos=self.update_graphics, size=self.update_graphics)
        self.update_graphics()
    
    def set_emotion(self, emotion):
        """Update the companion's emotion"""
        self.emotion = emotion
        self.update_graphics()
    
    def update_graphics(self, *args):
        """Draw the companion's face based on current emotion"""
        self.canvas.clear()
        
        with self.canvas:
            # Calculate face dimensions first for background sizing
            grid_size = 16  # 16x16 grid
            face_area = min(self.width, self.height) * 0.95
            pixel_size = int(face_area // grid_size)
            if pixel_size < 1:
                pixel_size = 1
            
            face_width = grid_size * pixel_size
            face_height = grid_size * pixel_size
            
            # Dark background sized and positioned to match the face
            Color(0.1, 0.1, 0.1, 1)  # Dark grey
            bg_x = self.x + (self.width - face_width) // 2
            bg_y = self.y + 10  # Same as face position
            Rectangle(pos=(bg_x - 10, bg_y - 10), size=(face_width + 20, face_height + 20))
            
            # White pixels for the face
            Color(1, 1, 1, 1)  # White
            
            # Use the already calculated dimensions
            start_x = self.x + (self.width - grid_size * pixel_size) // 2
            # Position face at the very top of the canvas
            start_y = self.y + 10  # Just 10 pixels from top
            
            # Draw face outline
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
            
            # Draw face outline pixels
            for px, py in face_outline:
                x = start_x + px * pixel_size
                y = start_y + (15 - py) * pixel_size  # Flip Y coordinate
                Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
            
            # Draw eyes based on emotion
            self._draw_eyes(start_x, start_y, pixel_size)
            
            # Draw mouth based on emotion
            self._draw_mouth(start_x, start_y, pixel_size)
    
    def _draw_eyes(self, start_x, start_y, pixel_size):
        """Draw eyes based on current emotion"""
        left_eye_x, left_eye_y = 5, 6
        right_eye_x, right_eye_y = 10, 6
        
        if self.emotion in ["Happy", "Neutral"]:
            # Normal eyes (2x2 pixels each)
            for dx in range(2):
                for dy in range(2):
                    # Left eye
                    x = start_x + (left_eye_x + dx) * pixel_size
                    y = start_y + (15 - (left_eye_y + dy)) * pixel_size
                    Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
                    # Right eye
                    x = start_x + (right_eye_x + dx) * pixel_size
                    y = start_y + (15 - (right_eye_y + dy)) * pixel_size
                    Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
        
        elif self.emotion == "Sad":
            # Droopy eyes and tears
            for dx in range(2):
                for dy in range(2):
                    # Left eye (lower)
                    x = start_x + (left_eye_x + dx) * pixel_size
                    y = start_y + (15 - (left_eye_y + dy + 1)) * pixel_size
                    Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
                    # Right eye (lower)
                    x = start_x + (right_eye_x + dx) * pixel_size
                    y = start_y + (15 - (right_eye_y + dy + 1)) * pixel_size
                    Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
            # Tears
            x = start_x + (left_eye_x + 1) * pixel_size
            y = start_y + (15 - (left_eye_y + 3)) * pixel_size
            Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
            x = start_x + (right_eye_x + 1) * pixel_size
            y = start_y + (15 - (right_eye_y + 3)) * pixel_size
            Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
        
        elif self.emotion == "Surprise":
            # Wide eyes (3x3 pixels each)
            for dx in range(3):
                for dy in range(3):
                    # Left eye
                    x = start_x + (left_eye_x - 1 + dx) * pixel_size
                    y = start_y + (15 - (left_eye_y - 1 + dy)) * pixel_size
                    Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
                    # Right eye
                    x = start_x + (right_eye_x - 1 + dx) * pixel_size
                    y = start_y + (15 - (right_eye_y - 1 + dy)) * pixel_size
                    Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
        
        elif self.emotion == "Angry":
            # Normal eyes with angry eyebrows
            for dx in range(2):
                for dy in range(2):
                    # Left eye
                    x = start_x + (left_eye_x + dx) * pixel_size
                    y = start_y + (15 - (left_eye_y + dy)) * pixel_size
                    Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
                    # Right eye
                    x = start_x + (right_eye_x + dx) * pixel_size
                    y = start_y + (15 - (right_eye_y + dy)) * pixel_size
                    Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
            # Angry eyebrows
            x = start_x + 4 * pixel_size
            y = start_y + (15 - 4) * pixel_size
            Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
            x = start_x + 5 * pixel_size
            y = start_y + (15 - 5) * pixel_size
            Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
            x = start_x + 10 * pixel_size
            y = start_y + (15 - 5) * pixel_size
            Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
            x = start_x + 11 * pixel_size
            y = start_y + (15 - 4) * pixel_size
            Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
        
        elif self.emotion == "Fear":
            # Wide worried eyes (same as surprise)
            for dx in range(3):
                for dy in range(3):
                    # Left eye
                    x = start_x + (left_eye_x - 1 + dx) * pixel_size
                    y = start_y + (15 - (left_eye_y - 1 + dy)) * pixel_size
                    Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
                    # Right eye
                    x = start_x + (right_eye_x - 1 + dx) * pixel_size
                    y = start_y + (15 - (right_eye_y - 1 + dy)) * pixel_size
                    Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
        
        elif self.emotion == "Disgust":
            # Squinted eyes (horizontal lines)
            for dx in range(2):
                # Left eye
                x = start_x + (left_eye_x + dx) * pixel_size
                y = start_y + (15 - (left_eye_y + 1)) * pixel_size
                Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
                # Right eye
                x = start_x + (right_eye_x + dx) * pixel_size
                y = start_y + (15 - (right_eye_y + 1)) * pixel_size
                Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
    
    def _draw_mouth(self, start_x, start_y, pixel_size):
        """Draw mouth based on current emotion"""
        mouth_y = 10
        
        if self.emotion == "Happy":
            # Smile
            smile_pixels = [(6, mouth_y), (7, mouth_y + 1), (8, mouth_y + 1), (9, mouth_y)]
            for px, py in smile_pixels:
                x = start_x + px * pixel_size
                y = start_y + (15 - py) * pixel_size
                Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
        
        elif self.emotion == "Sad":
            # Frown
            frown_pixels = [(6, mouth_y + 1), (7, mouth_y), (8, mouth_y), (9, mouth_y + 1)]
            for px, py in frown_pixels:
                x = start_x + px * pixel_size
                y = start_y + (15 - py) * pixel_size
                Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
        
        elif self.emotion == "Surprise":
            # Open mouth (small square)
            for dx in range(2):
                for dy in range(2):
                    x = start_x + (7 + dx) * pixel_size
                    y = start_y + (15 - (mouth_y + dy)) * pixel_size
                    Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
        
        elif self.emotion == "Angry":
            # Angry mouth (horizontal line)
            for px in range(6, 10):
                x = start_x + px * pixel_size
                y = start_y + (15 - mouth_y) * pixel_size
                Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
        
        elif self.emotion == "Fear":
            # Small worried mouth
            for px in range(7, 9):
                x = start_x + px * pixel_size
                y = start_y + (15 - mouth_y) * pixel_size
                Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
        
        elif self.emotion == "Disgust":
            # Wavy disgusted mouth
            disgust_pixels = [(6, mouth_y), (7, mouth_y - 1), (8, mouth_y), (9, mouth_y - 1)]
            for px, py in disgust_pixels:
                x = start_x + px * pixel_size
                y = start_y + (15 - py) * pixel_size
                Rectangle(pos=(x, y), size=(pixel_size, pixel_size))
        
        else:  # Neutral
            # Straight line mouth
            for px in range(6, 10):
                x = start_x + px * pixel_size
                y = start_y + (15 - mouth_y) * pixel_size
                Rectangle(pos=(x, y), size=(pixel_size, pixel_size))


class KivyCompanionGUI(BoxLayout):
    """Main Kivy Companion GUI class"""
    
    def __init__(self, **kwargs):
        super(KivyCompanionGUI, self).__init__(**kwargs)
        
        # Set layout orientation with more negative top margin
        self.orientation = 'vertical'
        self.padding = [0, -35, 0, 0]  # more negative top padding to pull content higher
        self.spacing = 0
        
        # Companion state
        self.companion_name = "EMO"
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
        
        # Chat mode
        self.chat_mode = False
        
        # Test camera availability
        self.test_camera()
        
        # Setup GUI
        self.init_ui()
        
        # Start companion behavior
        self.start_companion_behavior()
    
    def init_detector(self):
        """Safely initialize the emotion detector"""
        if not EMOTION_DETECTION_AVAILABLE:
            print("⚠️ Emotion detection not available")
            return
            
        try:
            self.detector = SimpleEmotionDetector()
            print("✅ Emotion detector initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize emotion detector: {e}")
            self.detector = None
    
    def test_camera(self):
        """Test if camera is available"""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap is not None and cap.isOpened():
                cap.release()
                self.camera_available = True
                return True
        except Exception as e:
            print(f"Camera test failed: {e}")
        self.camera_available = False
        return False
    
    def init_ui(self):
        """Initialize the retro-style user interface"""
        # Title with minimal height
        title_label = Label(
            text=f">>> {self.companion_name} <<<",
            size_hint_y=None,
            height=20,
            color=(1, 1, 1, 1),
            font_size='14sp',
            bold=True
        )
        self.add_widget(title_label)
        
        # Status display
        status_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=20,
            spacing=10
        )
        
        self.emotion_display = Label(
            text="EMOTION: NEUTRAL",
            color=(1, 1, 1, 1),
            font_size='11sp',
            bold=True
        )
        status_layout.add_widget(self.emotion_display)
        
        self.add_widget(status_layout)
        
        # Main companion display area (larger to accommodate bigger face)
        companion_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=400,  # Increased height to match larger canvas
            padding=[0, 0, 0, 0],
            spacing=0
        )

        # Make the canvas larger to fill the available space
        self.companion_canvas = CompanionCanvas(
            size_hint=(None, None),
            size=(400, 400)  # Larger canvas to fill more space
        )

        # Center the canvas horizontally only (no vertical spacers)
        row = BoxLayout(orientation='horizontal', size_hint_y=None, height=400)
        row.add_widget(Widget(size_hint_x=1))  # Left spacer
        row.add_widget(self.companion_canvas)
        row.add_widget(Widget(size_hint_x=1))  # Right spacer
        companion_layout.add_widget(row)
        self.add_widget(companion_layout)
        
        # Speech area
        speech_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=120,
            spacing=5
        )
        
        speech_label = Label(
            text=f"{self.companion_name} says:",
            size_hint_y=None,
            height=20,
            color=(1, 1, 1, 1),
            font_size='10sp',
            bold=True
        )
        speech_layout.add_widget(speech_label)
        
        self.speech_text = Label(
            text="SYSTEM INITIALIZED. EMOTION DETECTION READY.",
            size_hint_y=None,
            height=80,
            color=(1, 1, 1, 1),
            font_size='16sp',
            text_size=(None, None),
            halign='center',
            valign='middle'
        )
        speech_layout.add_widget(self.speech_text)
        
        self.add_widget(speech_layout)
        
        # Control buttons - larger to fill more space
        button_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=60,  # Increased from 40 to 60
            spacing=8
        )
        
        # Camera button
        self.camera_btn = Button(
            text="[ START CAMERA ]" if self.camera_available else "[ CAMERA N/A ]",
            disabled=not self.camera_available,
            font_size='14sp'  # Increased font size
        )
        self.camera_btn.bind(on_press=self.toggle_camera)
        button_layout.add_widget(self.camera_btn)
        
        # Chat button
        chat_btn = Button(
            text="[ CHAT MODE ]",
            font_size='14sp'  # Increased font size
        )
        chat_btn.bind(on_press=self.toggle_chat_mode)
        button_layout.add_widget(chat_btn)
        
        # Activities button
        activities_btn = Button(
            text="[ ACTIVITIES ]",
            font_size='14sp'  # Increased font size
        )
        activities_btn.bind(on_press=self.show_activities)
        button_layout.add_widget(activities_btn)
        
        # Stats button
        stats_btn = Button(
            text="[ STATS ]",
            font_size='14sp'  # Increased font size
        )
        stats_btn.bind(on_press=self.show_stats)
        button_layout.add_widget(stats_btn)
        
        # Settings button
        settings_btn = Button(
            text="[ SETTINGS ]",
            font_size='14sp'  # Increased font size
        )
        settings_btn.bind(on_press=self.show_settings)
        button_layout.add_widget(settings_btn)
        
        self.add_widget(button_layout)
        
        # Chat input (initially hidden)
        self.chat_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=0,  # Initially hidden
            spacing=5,
            opacity=0
        )
        
        chat_label = Label(
            text="Input:",
            size_hint_x=None,
            width=50,
            color=(1, 1, 1, 1),
            font_size='10sp'
        )
        self.chat_layout.add_widget(chat_label)
        
        self.chat_entry = TextInput(
            multiline=False,
            font_size='12sp'
        )
        self.chat_entry.bind(on_text_validate=self.send_chat_message)
        self.chat_layout.add_widget(self.chat_entry)
        
        send_btn = Button(
            text="[ SEND ]",
            size_hint_x=None,
            width=80,
            font_size='10sp'
        )
        send_btn.bind(on_press=self.send_chat_message)
        self.chat_layout.add_widget(send_btn)
        
        self.add_widget(self.chat_layout)
        
        # Initialize display
        self.update_companion_display()
    
    def update_companion_display(self):
        """Update the companion's visual display"""
        self.companion_canvas.set_emotion(self.current_emotion)
        self.emotion_display.text = f"EMOTION: {self.current_emotion.upper()}"
    
    def companion_speak(self, message):
        """Make the companion speak"""
        timestamp = time.strftime("%H:%M:%S")
        retro_message = f"[{timestamp}] >>> {message.upper()}"
        self.speech_text.text = retro_message
        self.speech_text.text_size = (self.speech_text.width, None)
        self.last_interaction = time.time()
    
    def toggle_chat_mode(self, instance):
        """Toggle chat input visibility"""
        if self.chat_mode:
            # Hide chat
            self.chat_layout.height = 0
            self.chat_layout.opacity = 0
            self.chat_mode = False
        else:
            # Show chat
            self.chat_layout.height = 40
            self.chat_layout.opacity = 1
            self.chat_mode = True
            self.chat_entry.focus = True
    
    def send_chat_message(self, instance=None):
        """Send a chat message"""
        message = self.chat_entry.text.strip()
        if message:
            response = self.generate_chat_response(message)
            self.companion_speak(f"YOU SAID: {message}. {response}")
            self.chat_entry.text = ""
            
            # Update companion based on chat
            self.last_interaction = time.time()
            self.happiness_level = min(100, self.happiness_level + 2)
            self.update_companion_display()
    
    def generate_chat_response(self, user_message):
        """Generate a response to user's chat message"""
        message_lower = user_message.lower()
        
        # Emotion-related responses
        if any(word in message_lower for word in ['sad', 'down', 'depressed', 'upset']):
            return "I'M SORRY YOU'RE FEELING THAT WAY. I'M HERE FOR YOU!"
        elif any(word in message_lower for word in ['happy', 'great', 'awesome', 'good']):
            return "WONDERFUL! YOUR HAPPINESS MAKES ME HAPPY TOO!"
        elif any(word in message_lower for word in ['angry', 'mad', 'frustrated']):
            return "TAKE DEEP BREATHS. THIS FEELING WILL PASS."
        elif any(word in message_lower for word in ['scared', 'afraid', 'worried']):
            return "YOU'RE BRAVER THAN YOU THINK! I BELIEVE IN YOU!"
        
        # General conversation
        elif any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return "HELLO! GREAT TO SEE YOU! HOW ARE YOU TODAY?"
        elif any(word in message_lower for word in ['how', 'you']):
            return f"I'M GREAT! HAPPINESS: {self.happiness_level}% ENERGY: {self.energy_level}%"
        elif 'joke' in message_lower:
            jokes = [
                "WHY DON'T SCIENTISTS TRUST ATOMS? THEY MAKE UP EVERYTHING!",
                "WHAT DO YOU CALL A FAKE NOODLE? AN IMPASTA!",
                "WHY DID THE SCARECROW WIN AN AWARD? OUTSTANDING IN HIS FIELD!",
                "WHAT DO YOU CALL A BEAR WITH NO TEETH? A GUMMY BEAR!"
            ]
            return random.choice(jokes)
        else:
            responses = [
                "THAT'S INTERESTING! TELL ME MORE!",
                "I LOVE CHATTING WITH YOU!",
                "THANKS FOR SHARING WITH ME!",
                "YOU'RE GOOD COMPANY! WHAT ELSE?",
                "I'M ALL EARS!"
            ]
            return random.choice(responses)
    
    def toggle_camera(self, instance):
        """Toggle camera detection on/off"""
        if not self.detection_active:
            self.start_camera_detection()
        else:
            self.stop_camera_detection()
    
    def start_camera_detection(self):
        """Start camera-based emotion detection"""
        if not self.camera_available:
            self.companion_speak("ERROR: CAMERA NOT AVAILABLE ON SYSTEM")
            return
        
        if not self.detector:
            self.companion_speak("ERROR: EMOTION DETECTOR NOT AVAILABLE")
            return
        
        self.detection_active = True
        self.camera_btn.text = "[ STOP CAMERA ]"
        
        self.camera_thread = threading.Thread(target=self.camera_detection_loop, daemon=True)
        self.camera_thread.start()
        
        self.companion_speak("CAMERA ACTIVATED. EMOTION DETECTION ONLINE.")
    
    def stop_camera_detection(self):
        """Stop camera detection"""
        self.detection_active = False
        self.camera_btn.text = "[ START CAMERA ]"
        self.companion_speak("CAMERA DEACTIVATED. EMOTION DETECTION OFFLINE.")
    
    def camera_detection_loop(self):
        """Main camera detection loop"""
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                Clock.schedule_once(lambda dt: self.companion_speak("SORRY, COULDN'T ACCESS CAMERA!"))
                self.detection_active = False
                return
            
            last_emotion_time = 0
            last_face_time = time.time()
            consecutive_errors = 0
            max_errors = 10
            min_face_size = 60
            emotion_cooldown = 1.0
            
            while self.detection_active and consecutive_errors < max_errors:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        consecutive_errors += 1
                        time.sleep(0.1)
                        continue
                    
                    consecutive_errors = 0
                    frame = cv2.flip(frame, 1)  # Mirror the frame
                    
                    faces = self.detector.detect_faces(frame)
                    face_found = False
                    
                    if faces and len(faces) > 0:
                        x, y, w, h = faces[0]
                        if w > min_face_size and h > min_face_size:
                            face_found = True
                            last_face_time = time.time()
                            face_img = frame[y:y+h, x:x+w]
                            
                            if face_img.size > 0:
                                emotion, confidence = self.detector.predict_emotion(face_img)
                                
                                current_time = time.time()
                                if (confidence > 0.7 and 
                                    current_time - last_emotion_time > emotion_cooldown):
                                    if emotion != self.current_emotion:
                                        Clock.schedule_once(
                                            lambda dt, e=emotion: self.on_emotion_detected(e)
                                        )
                                        last_emotion_time = current_time
                            
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{self.current_emotion}", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Fallback to Neutral if no face detected for 3 seconds
                    if not face_found and (time.time() - last_face_time > 3.0):
                        if self.current_emotion != "Neutral":
                            Clock.schedule_once(
                                lambda dt: self.on_emotion_detected("Neutral")
                            )
                            last_emotion_time = time.time()
                    
                    cv2.putText(frame, "Press 'q' to stop camera", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow(f'{self.companion_name} - Camera View', frame)
                    
                    if cv2.getWindowProperty(f'{self.companion_name} - Camera View', cv2.WND_PROP_VISIBLE) < 1:
                        # Window was closed by user
                        Clock.schedule_once(
                            lambda dt: self.companion_speak("CAMERA STOPPED. YOU CAN STILL CHAT MANUALLY!")
                        )
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        Clock.schedule_once(
                            lambda dt: self.companion_speak("CAMERA STOPPED. YOU CAN STILL CHAT MANUALLY!")
                        )
                        break
                        
                except Exception as e:
                    consecutive_errors += 1
                    time.sleep(0.05)
                    continue
                
                time.sleep(0.03)
            
            if consecutive_errors >= max_errors:
                Clock.schedule_once(
                    lambda dt: self.companion_speak("CAMERA HAD TOO MANY ERRORS. STOPPING DETECTION.")
                )
                
        except Exception as e:
            Clock.schedule_once(
                lambda dt: self.companion_speak(f"CAMERA ERROR: {str(e)[:50]}...")
            )
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
    
    def respond_to_emotion(self, emotion, manual=False):
        """Respond to detected or manual emotion"""
        responses = {
            "Happy": [
                "YAY! I LOVE SEEING YOU HAPPY! YOUR JOY IS CONTAGIOUS!",
                "YOUR HAPPINESS MAKES MY CIRCUITS SPARKLE WITH JOY!",
                "WHAT A WONDERFUL SMILE! YOU'RE ABSOLUTELY GLOWING!",
                "HAPPINESS LOOKS AMAZING ON YOU! KEEP SHINING!"
            ],
            "Sad": [
                "I'M HERE FOR YOU, ALWAYS. IT'S OKAY TO FEEL SAD SOMETIMES.",
                "WOULD YOU LIKE TO TALK ABOUT WHAT'S BOTHERING YOU? I'M LISTENING.",
                "REMEMBER, AFTER EVERY STORM COMES A RAINBOW. YOU'RE STRONGER THAN YOU KNOW!",
                "SENDING YOU THE BIGGEST VIRTUAL HUGS! YOU'RE NOT ALONE."
            ],
            "Angry": [
                "I CAN SEE YOU'RE UPSET. LET'S TAKE SOME DEEP BREATHS TOGETHER...",
                "IT'S COMPLETELY OKAY TO FEEL ANGRY. LET'S WORK THROUGH THIS TOGETHER.",
                "WOULD SOME CALMING THOUGHTS HELP? I'M HERE TO SUPPORT YOU.",
                "THIS FEELING WILL PASS. YOU'RE INCREDIBLY STRONG AND RESILIENT!"
            ],
            "Fear": [
                "HEY, I'M RIGHT HERE WITH YOU. YOU'RE COMPLETELY SAFE!",
                "FEAR IS NORMAL, BUT YOU'RE BRAVER THAN YOU THINK! YOU'RE A LION!",
                "LET'S FACE THIS TOGETHER. WHAT'S WORRYING YOU? I BELIEVE IN YOU!",
                "TAKE SLOW, DEEP BREATHS WITH ME. YOU'VE GOT THIS!"
            ],
            "Surprise": [
                "WOW! SOMETHING EXCITING HAPPENED? TELL ME EVERYTHING!",
                "I LOVE SURPRISES! YOUR EXPRESSION IS ABSOLUTELY PRICELESS!",
                "LIFE IS FULL OF AMAZING SURPRISES, ISN'T IT? HOW WONDERFUL!",
                "THAT LOOK OF WONDER IS ABSOLUTELY BEAUTIFUL!"
            ],
            "Disgust": [
                "EW, SOMETHING BOTHERING YOU? I'M HERE TO HELP MAKE IT BETTER!",
                "NOT EVERYTHING IN LIFE IS PLEASANT, BUT WE'll GET THROUGH IT TOGETHER!",
                "WANT TO TALK ABOUT WHAT'S BOTHERING YOU? I'M ALL EARS!",
                "LET'S FOCUS ON SOMETHING MORE PLEASANT TOGETHER!"
            ],
            "Neutral": [
                "HOW ARE YOU FEELING TODAY? I'M HERE AND READY TO CHAT!",
                "YOU SEEM CALM AND PEACEFUL. THAT'S REALLY NICE!",
                "JUST TAKING THINGS AS THEY COME? I REALLY RESPECT THAT!",
                "SOMETIMES NEUTRAL IS THE PERFECT WAY TO BE! BALANCE IS BEAUTIFUL!"
            ]
        }
        
        response = random.choice(responses.get(emotion, responses["Neutral"]))
        
        if manual:
            response = f"THANKS FOR TELLING ME! {response}"
        else:
            response = f"I CAN SEE YOU'RE FEELING {emotion.upper()}. {response}"
        
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
    
    def show_activities(self, instance):
        """Show activities popup"""
        content = BoxLayout(orientation='vertical', spacing=10, padding=20)
        
        title = Label(
            text=">>> SELECT ACTIVITY <<<",
            size_hint_y=None,
            height=40,
            font_size='14sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        content.add_widget(title)
        
        activities = [
            ("MOOD BOOST", self.random_mood),
            ("GIVE GIFT", self.give_gift),
            ("PLAY MUSIC", self.play_music),
            ("EXERCISE", self.exercise),
            ("MEDITATE", self.meditate),
            ("TELL STORY", self.tell_story)
        ]
        
        for name, action in activities:
            btn = Button(
                text=f"[ {name} ]",
                size_hint_y=None,
                height=40,
                font_size='12sp'
            )
            btn.bind(on_press=lambda x, a=action: self.run_activity(a, popup))
            content.add_widget(btn)
        
        close_btn = Button(
            text="[ CLOSE ]",
            size_hint_y=None,
            height=40,
            font_size='12sp'
        )
        
        popup = Popup(
            title=">>> ACTIVITIES <<<",
            content=content,
            size_hint=(0.8, 0.8),
            auto_dismiss=False
        )
        
        close_btn.bind(on_press=popup.dismiss)
        content.add_widget(close_btn)
        
        popup.open()
    
    def run_activity(self, activity_func, popup):
        """Run an activity and close popup"""
        activity_func()
        popup.dismiss()
    
    def show_stats(self, instance):
        """Show stats popup"""
        stats_info = f"""COMPANION STATUS:
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
DETECTION MODEL: {'LOADED' if self.detector else 'NOT AVAILABLE'}
ACTION UNITS: {'ENABLED' if self.detector else 'DISABLED'}
"""
        
        content = BoxLayout(orientation='vertical', spacing=10, padding=20)
        
        title = Label(
            text=">>> COMPANION STATISTICS <<<",
            size_hint_y=None,
            height=40,
            font_size='14sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        content.add_widget(title)
        
        stats_label = Label(
            text=stats_info,
            text_size=(None, None),
            halign='left',
            valign='top',
            font_size='10sp',
            color=(1, 1, 1, 1)
        )
        content.add_widget(stats_label)
        
        close_btn = Button(
            text="[ CLOSE ]",
            size_hint_y=None,
            height=40,
            font_size='12sp'
        )
        
        popup = Popup(
            title=">>> STATISTICS <<<",
            content=content,
            size_hint=(0.8, 0.8)
        )
        
        close_btn.bind(on_press=popup.dismiss)
        content.add_widget(close_btn)
        
        popup.open()
    
    def show_settings(self, instance):
        """Show settings popup"""
        content = BoxLayout(orientation='vertical', spacing=10, padding=20)
        
        title = Label(
            text=">>> COMPANION SETTINGS <<<",
            size_hint_y=None,
            height=40,
            font_size='14sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        content.add_widget(title)
        
        # Name setting
        name_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        name_layout.add_widget(Label(text="NAME:", size_hint_x=None, width=60, color=(1, 1, 1, 1)))
        
        name_entry = TextInput(
            text=self.companion_name,
            multiline=False,
            size_hint_x=0.7
        )
        name_layout.add_widget(name_entry)
        
        def update_name(instance):
            new_name = name_entry.text.strip().upper()
            if new_name and new_name != self.companion_name:
                old_name = self.companion_name
                self.companion_name = new_name
                self.companion_speak(f"NAME CHANGED FROM {old_name} TO {self.companion_name}")
        
        name_btn = Button(
            text="[ UPDATE ]",
            size_hint_x=None,
            width=80,
            font_size='10sp'
        )
        name_btn.bind(on_press=update_name)
        name_layout.add_widget(name_btn)
        
        content.add_widget(name_layout)
        
        # Reset button
        reset_btn = Button(
            text="[ RESET COMPANION ]",
            size_hint_y=None,
            height=40,
            font_size='12sp'
        )
        
        def reset_companion(instance):
            self.happiness_level = 75
            self.energy_level = 80
            self.current_emotion = "Neutral"
            self.conversation_history = []
            self.update_companion_display()
            self.companion_speak("SYSTEM RESET COMPLETE. ALL PARAMETERS RESTORED.")
            popup.dismiss()
        
        reset_btn.bind(on_press=reset_companion)
        content.add_widget(reset_btn)
        
        close_btn = Button(
            text="[ CLOSE ]",
            size_hint_y=None,
            height=40,
            font_size='12sp'
        )
        
        popup = Popup(
            title=">>> SETTINGS <<<",
            content=content,
            size_hint=(0.8, 0.6)
        )
        
        close_btn.bind(on_press=popup.dismiss)
        content.add_widget(close_btn)
        
        popup.open()
    
    # Activity methods
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
    
    def start_companion_behavior(self):
        """Start the companion's autonomous behavior"""
        def behavior_loop(dt):
            try:
                current_time = time.time()
                
                # Periodic check-ins
                if current_time - self.last_interaction > self.question_interval:
                    self.periodic_checkin()
                    self.last_interaction = current_time
                
                # Gradual mood changes
                if random.random() < 0.1:  # 10% chance
                    self.gradual_mood_change()
                    
            except Exception as e:
                print(f"Behavior loop error: {e}")
        
        # Schedule the behavior loop to run every 5 seconds
        Clock.schedule_interval(behavior_loop, 5)
    
    def periodic_checkin(self):
        """Periodic check-in with the user"""
        checkin_messages = [
            "HOW ARE YOU FEELING RIGHT NOW?",
            "JUST CHECKING IN! HOW'S YOUR DAY GOING?",
            "I HAVEN'T HEARD FROM YOU IN A WHILE. EVERYTHING OKAY?",
            "WANT TO CHAT? I'M HERE IF YOU NEED ME!",
            "HOW'S YOUR MOOD TODAY? I'M HERE TO LISTEN!",
            "JUST WANTED TO SAY HI! HOW ARE YOU DOING?"
        ]
        
        message = random.choice(checkin_messages)
        self.companion_speak(message)
    
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
        self.update_companion_display()


class CompanionApp(App):
    """Main Kivy application"""
    
    def build(self):
        # Set window properties - smaller height to reduce empty space
        Window.size = (700, 400)
        Window.minimum_width = 700
        Window.minimum_height = 400
        Window.title = "EMO - Emotion Companion"
        
        # Set background color to dark grey
        Window.clearcolor = (0.16, 0.16, 0.16, 1)
        
        # Create the main widget normally
        main_widget = KivyCompanionGUI()
        return main_widget


# Mark first todo as completed since we have the basic structure
if __name__ == "__main__":
    if not KIVY_AVAILABLE:
        print("❌ Kivy not available. Please install Kivy:")
        print("pip install kivy")
        exit(1)
    
    print("🎭 Starting Kivy Interactive Companion GUI...")
    CompanionApp().run()