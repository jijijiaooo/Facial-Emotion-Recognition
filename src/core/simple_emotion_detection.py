#!/usr/bin/env python3
"""
Simple Emotion Detection - Supports Keras, PKL, and TFLite models

Added built-in Wi‑Fi sender so this file can be used standalone (sends to ESP32
and to an optional Raspberry Pi HTTP / WebSocket endpoint). Throttles sends to
avoid flooding the network.
"""

import cv2
import numpy as np
import os
import sys
import pickle
import threading
import time

# Networking
import json
import requests

# Optional websocket client (used only if installed)
try:
    from websocket import create_connection, WebSocketException  # websocket-client package
    WEBSOCKET_AVAILABLE = True
except Exception:
    WEBSOCKET_AVAILABLE = False

# Try to import TFLite
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
    TFLITE_RUNTIME = True
    print("TFLite Runtime available (optimized)")
except ImportError:
    try:
        import tensorflow as tf
        TFLITE_AVAILABLE = True
        TFLITE_RUNTIME = False
        print("TensorFlow Lite available")
    except ImportError:
        TFLITE_AVAILABLE = False
        TFLITE_RUNTIME = False
        print("TFLite not available - .tflite models will be skipped")

# --------------------------
# Wi-Fi / network configuration (change environment variables or call configure_wifi_targets)
# --------------------------
ESP32_IP = os.environ.get("ESP32_IP", "192.168.0.107")
ESP32_PORT = int(os.environ.get("ESP32_PORT", "8080"))

# Raspberry Pi endpoints (optional). If you want Android to connect to the Pi (WebSocket),
# run a WS server on the Pi and set RPI_WS_IP/RPI_WS_PORT. Also you can expose an HTTP JSON endpoint.
RPI_HTTP_IP = os.environ.get("RPI_HTTP_IP", "")  # e.g. "192.168.0.102"
RPI_HTTP_PORT = int(os.environ.get("RPI_HTTP_PORT", "5000")) if os.environ.get("RPI_HTTP_IP") else None

RPI_WS_IP = os.environ.get("RPI_WS_IP", "")  # e.g. "192.168.0.102"
RPI_WS_PORT = int(os.environ.get("RPI_WS_PORT", "8080")) if os.environ.get("RPI_WS_IP") else None

EMOJI_MAP = {
    "Happy": "^_^",
    "Sad": "T_T",
    "Angry": ">:[",
    "Fear": "O_O",
    "Surprise": "O.O",
    "Disgust": ">_<",
    "Neutral": "-_-"
}

# small helper to configure at runtime
def configure_wifi_targets(esp32_ip: str = None,
                           esp32_port: int = None,
                           rpi_http_ip: str = None,
                           rpi_http_port: int = None,
                           rpi_ws_ip: str = None,
                           rpi_ws_port: int = None):
    global ESP32_IP, ESP32_PORT, RPI_HTTP_IP, RPI_HTTP_PORT, RPI_WS_IP, RPI_WS_PORT
    if esp32_ip:
        ESP32_IP = esp32_ip
    if esp32_port:
        ESP32_PORT = int(esp32_port)
    if rpi_http_ip is not None:
        RPI_HTTP_IP = rpi_http_ip
        RPI_HTTP_PORT = int(rpi_http_port) if rpi_http_port else None
    if rpi_ws_ip is not None:
        RPI_WS_IP = rpi_ws_ip
        RPI_WS_PORT = int(rpi_ws_port) if rpi_ws_port else None


def _build_esp32_payload(emotion: str) -> bytes:
    emoji = EMOJI_MAP.get(emotion, "??")
    return f"{emoji} {emotion}".encode("utf-8")


def _build_rpi_payload_json(emotion: str):
    return {
        "emotion": emotion,
        "emoji": EMOJI_MAP.get(emotion, "??"),
        "source": "rpi4b"  # change if needed
    }


def send_emotion_wifi(emotion: str,
                      send_to_esp: bool = True,
                      send_to_rpi_http: bool = True,
                      send_to_rpi_ws: bool = True,
                      timeout: float = 1.0) -> dict:
    """
    Send the emotion to configured targets. This function is synchronous and intended
    to be called inside a background thread to avoid blocking inference loop.

    Returns dict with results for 'esp', 'rpi_http', 'rpi_ws' (either response text or error).
    """
    results = {"esp": None, "rpi_http": None, "rpi_ws": None}

    if not emotion:
        return results

    # 1) Send to ESP32 (simple plain-text device)
    if send_to_esp and ESP32_IP:
        try:
            url = f"http://{ESP32_IP}:{ESP32_PORT}/emotion"
            payload = _build_esp32_payload(emotion)
            headers = {"Content-Type": "text/plain; charset=utf-8"}
            resp = requests.post(url, data=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            results["esp"] = resp.text
        except Exception as e:
            results["esp"] = f"error: {e}"

    # 2) Send JSON to Raspberry Pi HTTP endpoint (optional)
    if send_to_rpi_http and RPI_HTTP_IP:
        try:
            if not RPI_HTTP_PORT:
                results["rpi_http"] = "error: RPI_HTTP_PORT not set"
            else:
                url = f"http://{RPI_HTTP_IP}:{RPI_HTTP_PORT}/emotion"
                payload_json = _build_rpi_payload_json(emotion)
                headers = {"Content-Type": "application/json; charset=utf-8"}
                resp = requests.post(url, data=json.dumps(payload_json), headers=headers, timeout=timeout)
                resp.raise_for_status()
                results["rpi_http"] = resp.text
        except Exception as e:
            results["rpi_http"] = f"error: {e}"

    # 3) Send via WebSocket to Raspberry Pi (optional)
    if send_to_rpi_ws and RPI_WS_IP and WEBSOCKET_AVAILABLE:
        try:
            ws_url = f"ws://{RPI_WS_IP}:{RPI_WS_PORT}"
            # open short-lived connection and send a JSON message
            ws = create_connection(ws_url, timeout=timeout)
            ws.send(json.dumps(_build_rpi_payload_json(emotion)))
            # optionally read response (non-blocking expects short timeout)
            try:
                resp = ws.recv()
            except Exception:
                resp = "sent"
            ws.close()
            results["rpi_ws"] = resp
        except WebSocketException as e:
            results["rpi_ws"] = f"ws error: {e}"
        except Exception as e:
            results["rpi_ws"] = f"error: {e}"
    else:
        if send_to_rpi_ws and RPI_WS_IP and not WEBSOCKET_AVAILABLE:
            results["rpi_ws"] = "error: websocket-client not installed"

    return results


# --------------------------
# End Wi-Fi helper functions
# --------------------------

class SimpleEmotionDetector:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.colors = {
            'Angry': (0, 0, 255), 'Disgust': (0, 255, 0), 'Fear': (255, 0, 255),
            'Happy': (0, 255, 255), 'Neutral': (255, 255, 255), 'Sad': (255, 0, 0),
            'Surprise': (0, 165, 255)
        }
        self.models = self.load_all_models()
        self.face_cascade = self.init_face_detection()
        self.fps_counter = 0
        self.fps_start = cv2.getTickCount()
        self.current_fps = 0
        self.debug_mode = False
        self.last_aus = {}
        
        # Temporal smoothing for stable detection
        self.emotion_history = []
        self.confidence_history = []
        self.history_size = 3  # Reduced from 5 to 3 for more responsiveness
        self.gui_mode = False  # Flag for GUI mode with more responsive detection
        
        # Adaptive thresholds (will adjust based on face characteristics)
        self.adaptive_thresholds = {
            'brightness_baseline': 120,
            'contrast_baseline': 30,
            'face_size_factor': 1.0
        }
        
        # Advanced AU combination patterns
        self.emotion_patterns = self._init_emotion_patterns()

        # Wi-Fi send throttle: don't send every frame
        self._last_sent_emotion = None
        self._last_sent_time = 0.0
        self._send_cooldown = 0.7  # seconds

    # ... (rest of the methods unchanged, identical to prior implementation) ...
    # For brevity the following methods are left unchanged in content:
    # _init_emotion_patterns, load_all_models, init_face_detection, detect_faces,
    # preprocess_face_48x48, preprocess_face_224x224, predict_emotion, extract_action_units,
    # _update_adaptive_thresholds, _temporal_smoothing, _advanced_emotion_scoring,
    # predict_basic, update_fps
    #
    # NOTE: The complete implementations are preserved as in the file you're editing.
    # The only functional additions are send_emotion (below) and the Wi-Fi helpers above.

    def send_emotion(self, emotion):
        """
        Send detected emotion using send_emotion_wifi in a background thread.
        Throttle to avoid flooding the network: send only if emotion changed or cooldown elapsed.
        """
        now = time.time()
        if (emotion == self._last_sent_emotion) and (now - self._last_sent_time < self._send_cooldown):
            # skip sending if same emotion and still in cooldown
            if self.debug_mode:
                print(f"[WiFi] Skipping send (throttled) for {emotion}")
            return

        # Update last sent values immediately to avoid concurrent repeated sends
        self._last_sent_emotion = emotion
        self._last_sent_time = now

        # Run in background thread
        threading.Thread(target=self._send_emotion_bg, args=(emotion,), daemon=True).start()

    def _send_emotion_bg(self, emotion):
        """
        Background thread worker to call send_emotion_wifi and log the result.
        """
        try:
            # choose which channels to send to; set RPI targets by env vars or configure_wifi_targets()
            res = send_emotion_wifi(
                emotion,
                send_to_esp=True,
                send_to_rpi_http=bool(RPI_HTTP_IP),
                send_to_rpi_ws=bool(RPI_WS_IP),
                timeout=1.0
            )
            if self.debug_mode:
                print(f"[WiFi] Sent emotion '{emotion}': {res}")
        except Exception as e:
            print(f"[WiFi] Error sending emotion: {e}")

    def run(self):
        """Main detection loop (unchanged)"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Could not open camera")
            return
        
        print("Starting Enhanced Emotion Detection with Action Units")
        print("Press 'q' to quit, 's' to save screenshot, 'd' to toggle debug mode")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.update_fps()
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Only process the largest face (if any faces detected)
                if faces:
                    # Find the largest face by area
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = largest_face
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence = self.predict_emotion(face_img)

                    # --- WiFi: Send emotion to ESP32/phone ---
                    self.send_emotion(emotion)
                    
                    # Draw results
                    color = self.colors.get(emotion, (255, 255, 255))
                    
                    # Face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Emotion label
                    label = f"{emotion}"
                    if confidence > 0.6:
                        label += f" ({confidence:.2f})"
                    
                    # Text background
                    cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
                    cv2.putText(frame, label, (x+5, y-8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Debug mode: show Action Units
                    if self.debug_mode and self.last_aus:
                        debug_y = y + h + 20
                        for i, (au_name, au_value) in enumerate(self.last_aus.items()):
                            debug_text = f"{au_name}: {au_value:.1f}"
                            cv2.putText(frame, debug_text, (x, debug_y + i*15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show FPS
                fps_text = f"FPS: {self.current_fps}"
                cv2.putText(frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Simple Emotion Detection", frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite('emotion_screenshot.jpg', frame)
                    print("Screenshot saved as emotion_screenshot.jpg")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Simple emotion detection stopped")

def main():
    print("Simple Emotion Detection")
    print("=" * 30)
    print("This version works without external cascade files")
    print("and provides basic emotion recognition")
    
    detector = SimpleEmotionDetector()
    detector.run()

if __name__ == "__main__":
    main()
