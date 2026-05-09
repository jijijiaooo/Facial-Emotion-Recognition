"""
Video Stream Processing (RTSP/SRT)
Connects to video streams, processes frames, and detects emotions
"""

import cv2
import asyncio
import numpy as np
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class VideoStreamProcessor:
    """Process video stream (RTSP/SRT) and detect emotions"""
    
    def __init__(self, stream_url: str, detector, fps: int = 2):
        """
        Args:
            stream_url: Video stream URL 
                       (e.g., rtsp://192.168.0.100:8554/camera1 
                        or srt://57.158.24.176:8890?streamid=read:camera1)
            detector: Emotion detector instance
            fps: Frames per second to process (default 2 = every 0.5 seconds)
        """
        self.stream_url = stream_url
        self.detector = detector
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.is_running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_result = None
        self.protocol = "SRT" if stream_url.startswith("srt://") else "RTSP"
        
    async def connect(self) -> bool:
        """Connect to video stream (RTSP/SRT)"""
        try:
            logger.info(f"Connecting to {self.protocol} stream: {self.stream_url}")
            
            # Use CAP_FFMPEG backend for SRT support
            if self.protocol == "SRT":
                self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.stream_url)
            
            # Set buffer size to 1 to get latest frame
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open {self.protocol} stream")
                return False
            
            logger.info(f"✓ {self.protocol} stream connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to {self.protocol}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from video stream"""
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f"{self.protocol} stream disconnected")
    
    async def process_stream(self):
        """Main processing loop - reads frames and detects emotions"""
        if not await self.connect():
            logger.error(f"Failed to connect to {self.protocol} stream")
            return
        
        self.is_running = True
        last_process_time = 0
        
        logger.info(f"Starting {self.protocol} stream processing at {self.fps} FPS")
        
        try:
            while self.is_running:
                current_time = asyncio.get_event_loop().time()
                
                # Check if enough time has passed
                if current_time - last_process_time < self.frame_interval:
                    await asyncio.sleep(0.01)  # Small sleep to prevent CPU overload
                    continue
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame from {self.protocol} stream, reconnecting...")
                    self.disconnect()
                    if not await self.connect():
                        break
                    continue
                
                # Process frame
                try:
                    result = await self._process_frame(frame)
                    self.latest_result = result
                    last_process_time = current_time
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
        finally:
            self.disconnect()
            self.is_running = False
    
    async def _process_frame(self, frame):
        """Process a single frame and detect emotions"""
        # Use detector-native API when available
        if hasattr(self.detector, 'detect_emotion'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.detector.detect_emotion(rgb_frame)
        else:
            # Compatibility path for detectors exposing detect_faces + predict_emotion
            faces = self.detector.detect_faces(frame, single_face=False)
            detections = []

            for i, (x, y, w, h) in enumerate(faces):
                face_img = frame[y:y+h, x:x+w]
                emotion, confidence = self.detector.predict_emotion(face_img)
                detections.append({
                    'face_id': i,
                    'emotion': emotion,
                    'confidence': float(confidence),
                    'bbox': {
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h)
                    }
                })

            result = {
                'success': True,
                'faces_detected': len(detections),
                'results': detections
            }
        
        # Add timestamp and metadata
        result['timestamp'] = datetime.now().isoformat()
        result['camera'] = f'{self.protocol.lower()}_stream'
        result['stream_url'] = self.stream_url
        
        return result
    
    def stop(self):
        """Stop processing"""
        logger.info(f"Stopping {self.protocol} stream processing")
        self.is_running = False
    
    def get_latest_result(self):
        """Get the latest emotion detection result"""
        return self.latest_result
