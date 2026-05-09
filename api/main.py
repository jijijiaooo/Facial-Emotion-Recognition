#!/usr/bin/env python3
"""
FastAPI Application for Emotion Detection
Serves the Enhanced Hybrid Emotion Detection model via REST API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import io
from PIL import Image
import logging
from typing import Dict, List, Set
import os
import asyncio
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the emotion detector
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.revised_emotion_detection import RevisedEmotionDetector
from api.rtsp_stream import VideoStreamProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Facial Emotion Recognition API",
    description="Real-time facial emotion detection using Enhanced Hybrid CNN",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = None

DEFAULT_EMOTIONS = ['angry', 'disgust', 'shocked', 'happy', 'neutral', 'sad']


def get_display_emotions(detector_instance) -> List[str]:
    """Return user-visible emotion labels."""
    if not detector_instance:
        return DEFAULT_EMOTIONS

    labels = getattr(detector_instance, "EMOTION_LABELS", DEFAULT_EMOTIONS)
    if hasattr(detector_instance, "get_display_label"):
        return [detector_instance.get_display_label(label) for label in labels]
    return labels

# Video stream processor (RTSP/SRT)
stream_processor: VideoStreamProcessor = None

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        self.active_connections -= disconnected

manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize the emotion detector on startup"""
    global detector
    logger.info("🚀 Starting Emotion Detection API...")
    
    try:
        # Configure TensorFlow for better performance
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        
        # Initialize detector (uses default revised model path)
        detector = RevisedEmotionDetector()
        logger.info("✅ Emotion detector initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global stream_processor
    logger.info("🛑 Shutting down Emotion Detection API...")
    
    # Stop video stream if running
    if stream_processor:
        stream_processor.stop()


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Facial Emotion Recognition API",
        "version": "1.0.0",
        "status": "running",
        "model": "Revised CNN 2026",
        "emotions": get_display_emotions(detector)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_loaded": detector is not None
    }


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
    
    Returns:
        JSON with detected faces and their emotions
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect faces
        faces = detector.detect_faces(img, single_face=False)
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_img = img[y:y+h, x:x+w]
            
            # Predict emotion
            emotion, confidence = detector.predict_emotion(face_img)
            
            results.append({
                "face_id": i,
                "emotion": emotion,
                "confidence": float(confidence),
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                }
            })
        
        return {
            "success": True,
            "faces_detected": len(faces),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_emotions_batch(files: List[UploadFile] = File(...)):
    """
    Predict emotions from multiple images
    
    Args:
        files: List of image files
    
    Returns:
        JSON with results for each image
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    batch_results = []
    
    for file_idx, file in enumerate(files):
        try:
            # Read image file
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                batch_results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid image file"
                })
                continue
            
            # Detect faces
            faces = detector.detect_faces(img, single_face=False)
            
            results = []
            for i, (x, y, w, h) in enumerate(faces):
                face_img = img[y:y+h, x:x+w]
                emotion, confidence = detector.predict_emotion(face_img)
                
                results.append({
                    "face_id": i,
                    "emotion": emotion,
                    "confidence": float(confidence),
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    }
                })
            
            batch_results.append({
                "filename": file.filename,
                "success": True,
                "faces_detected": len(faces),
                "results": results
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            batch_results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_images": len(files),
        "batch_results": batch_results
    }


@app.get("/emotions")
async def get_emotions():
    """Get list of supported emotions"""
    emotions = get_display_emotions(detector)
    return {
        "emotions": emotions,
        "count": len(emotions)
    }


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    emotions = get_display_emotions(detector)

    return {
        "model_type": "Revised CNN 2026",
        "model_path": getattr(detector, "model_path", None),
        "architecture": "Single-branch CNN (revised dataset)",
        "input_size": "112x112 grayscale",
        "emotions": emotions,
        "face_detection": "Haar Cascade",
        "landmark_detection": "not used"
    }


# ==================== VIDEO STREAM ENDPOINTS (RTSP/SRT) ====================

@app.post("/stream/start")
async def start_video_stream(stream_url: str, fps: int = 2):
    """
    Start processing video stream (RTSP/SRT)
    
    Args:
        stream_url: Video stream URL
                   RTSP: rtsp://192.168.0.100:8554/camera1
                   SRT: srt://57.158.24.176:8890?streamid=read:camera1
        fps: Frames per second to process (default: 2)
    
    Returns:
        Status of stream processing
    """
    global stream_processor
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    if stream_processor and stream_processor.is_running:
        raise HTTPException(status_code=400, detail="Video stream already running")
    
    try:
        # Create processor
        stream_processor = VideoStreamProcessor(stream_url, detector, fps=fps)
        
        # Start processing in background
        asyncio.create_task(stream_and_broadcast())
        
        protocol = "SRT" if stream_url.startswith("srt://") else "RTSP"
        
        return {
            "success": True,
            "message": f"{protocol} stream processing started",
            "stream_url": stream_url,
            "protocol": protocol,
            "fps": fps
        }
    
    except Exception as e:
        logger.error(f"Error starting video stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream/stop")
async def stop_video_stream():
    """Stop processing video stream"""
    global stream_processor
    
    if not stream_processor:
        raise HTTPException(status_code=400, detail="No video stream running")
    
    stream_processor.stop()
    stream_processor = None
    
    return {
        "success": True,
        "message": "Video stream processing stopped"
    }


@app.get("/stream/status")
async def get_stream_status():
    """Get status of video stream processing"""
    if not stream_processor:
        return {
            "running": False,
            "protocol": None,
            "latest_result": None
        }
    
    return {
        "running": stream_processor.is_running,
        "protocol": stream_processor.protocol,
        "stream_url": stream_processor.stream_url,
        "latest_result": stream_processor.get_latest_result()
    }


# ==================== BACKWARD COMPATIBILITY (RTSP ENDPOINTS) ====================

@app.post("/rtsp/start")
async def start_rtsp_stream_compat(rtsp_url: str, fps: int = 2):
    """Backward compatibility: redirect to /stream/start"""
    return await start_video_stream(stream_url=rtsp_url, fps=fps)


@app.post("/rtsp/stop")
async def stop_rtsp_stream_compat():
    """Backward compatibility: redirect to /stream/stop"""
    return await stop_video_stream()


@app.get("/rtsp/status")
async def get_rtsp_status_compat():
    """Backward compatibility: redirect to /stream/status"""
    return await get_stream_status()


# ==================== BACKGROUND TASK ====================

async def stream_and_broadcast():
    """Background task: process video stream and broadcast results"""
    global stream_processor
    
    logger.info(f"Starting {stream_processor.protocol} stream processing and broadcasting...")
    
    # Start processing
    await stream_processor.process_stream()
    
    # Broadcast results to WebSocket clients
    while stream_processor and stream_processor.is_running:
        result = stream_processor.get_latest_result()
        if result:
            await manager.broadcast(result)
        await asyncio.sleep(0.1)  # Check for new results every 100ms


# ==================== WEBSOCKET ENDPOINT ====================

@app.websocket("/ws/emotions")
async def websocket_emotions(websocket: WebSocket):
    """
    WebSocket endpoint for real-time emotion updates
    
    Clients connect here to receive continuous emotion detection results
    from the video stream (RTSP/SRT)
    """
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to emotion detection stream",
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (ping/pong for keep-alive)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Echo back (for ping/pong)
                if data == "ping":
                    await websocket.send_text("pong")
                
            except asyncio.TimeoutError:
                # Send keep-alive ping
                await websocket.send_json({"type": "ping"})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Azure sets PORT)
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
