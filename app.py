#!/usr/bin/env python3
"""
Flask API for Emotion Detection
Provides REST API endpoints for the Enhanced Hybrid Emotion Detection system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'core'))

from hybrid_emotion_detection_enhanced import EnhancedHybridEmotionDetector

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize SocketIO for real-time streaming
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize the emotion detector (loads model once on startup)
print("Initializing emotion detector...")
detector = None

try:
    detector = EnhancedHybridEmotionDetector()
    print("✅ Emotion detector initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize detector: {e}")
    print("   The API will return errors until a model is available")


@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    return jsonify({
        'name': 'Emotion Detection API',
        'version': '3.0',
        'status': 'running' if detector else 'error - model not loaded',
        'features': {
            'rest_api': True,
            'websocket_streaming': True,
            'multi_face_detection': True
        },
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Predict emotion from image',
            '/predict_batch': 'POST - Predict emotions from multiple faces',
            'ws://stream': 'WebSocket - Real-time video streaming'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if detector is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'emotions': detector.emotions
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict emotion from a single image
    
    Accepts:
    - JSON with base64 encoded image: {"image": "base64_string"}
    - Multipart form-data with image file: file field named 'image'
    
    Returns:
    - JSON with emotion prediction and confidence
    """
    if detector is None:
        return jsonify({
            'error': 'Model not initialized',
            'message': 'The emotion detection model failed to load'
        }), 503
    
    try:
        # Get image from request
        image = None
        
        # Check if it's JSON with base64
        if request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'No image provided in JSON'}), 400
            
            # Decode base64 image
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Check if it's multipart form-data
        elif 'image' in request.files:
            file = request.files['image']
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Detect face and predict emotion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if detector.face_cascade is None:
            return jsonify({'error': 'Face detection not available'}), 500
        
        faces = detector.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(48, 48)
        )
        
        if len(faces) == 0:
            return jsonify({
                'emotion': None,
                'confidence': 0,
                'message': 'No face detected in image'
            })
        
        # Use the first detected face
        x, y, w, h = faces[0]
        face_img = gray[y:y+h, x:x+w]
        
        # Predict emotion
        emotion, confidence = detector.predict_emotion(face_img)
        
        return jsonify({
            'emotion': emotion,
            'confidence': float(confidence),
            'face_location': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict emotions from an image containing multiple faces
    
    Accepts:
    - JSON with base64 encoded image: {"image": "base64_string"}
    - Multipart form-data with image file: file field named 'image'
    
    Returns:
    - JSON with array of predictions for all detected faces
    """
    if detector is None:
        return jsonify({
            'error': 'Model not initialized',
            'message': 'The emotion detection model failed to load'
        }), 503
    
    try:
        # Get image from request
        image = None
        
        # Check if it's JSON with base64
        if request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'No image provided in JSON'}), 400
            
            # Decode base64 image
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Check if it's multipart form-data
        elif 'image' in request.files:
            file = request.files['image']
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Detect all faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if detector.face_cascade is None:
            return jsonify({'error': 'Face detection not available'}), 500
        
        faces = detector.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(48, 48)
        )
        
        if len(faces) == 0:
            return jsonify({
                'faces': [],
                'count': 0,
                'message': 'No faces detected in image'
            })
        
        # Predict emotion for each face
        predictions = []
        for i, (x, y, w, h) in enumerate(faces):
            face_img = gray[y:y+h, x:x+w]
            emotion, confidence = detector.predict_emotion(face_img)
            
            predictions.append({
                'face_id': i,
                'emotion': emotion,
                'confidence': float(confidence),
                'face_location': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                }
            })
        
        return jsonify({
            'faces': predictions,
            'count': len(predictions)
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/emotions', methods=['GET'])
def get_emotions():
    """Get list of supported emotions"""
    if detector is None:
        return jsonify({'error': 'Model not initialized'}), 503
    
    return jsonify({
        'emotions': detector.emotions,
        'count': len(detector.emotions)
    })


# ============================================================================
# WebSocket Endpoints for Real-time Streaming
# ============================================================================

# Store active connections and their session data
active_sessions = {}


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    session_id = request.sid
    active_sessions[session_id] = {
        'connected_at': time.time(),
        'frame_count': 0,
        'last_emotion': None
    }
    print(f"✅ Client connected: {session_id}")
    emit('connected', {
        'status': 'connected',
        'session_id': session_id,
        'message': 'Ready for real-time emotion detection'
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    session_id = request.sid
    if session_id in active_sessions:
        session = active_sessions[session_id]
        duration = time.time() - session['connected_at']
        print(f"❌ Client disconnected: {session_id}")
        print(f"   Duration: {duration:.1f}s, Frames: {session['frame_count']}")
        del active_sessions[session_id]


@socketio.on('video_frame')
def handle_video_frame(data):
    """Handle incoming video frame for real-time emotion detection
    
    Expected data format:
    {
        'image': 'base64_encoded_image',
        'timestamp': optional_client_timestamp,
        'frame_id': optional_frame_identifier
    }
    """
    session_id = request.sid
    
    if detector is None:
        emit('error', {
            'error': 'Model not initialized',
            'message': 'The emotion detection model failed to load'
        })
        return
    
    try:
        # Update session stats
        if session_id in active_sessions:
            active_sessions[session_id]['frame_count'] += 1
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces and predict emotions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if detector.face_cascade is None:
            emit('error', {'error': 'Face detection not available'})
            return
        
        faces = detector.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
        
        # Process all detected faces
        predictions = []
        for i, (x, y, w, h) in enumerate(faces):
            face_img = frame[y:y+h, x:x+w]
            emotion, confidence = detector.predict_emotion(face_img)
            
            predictions.append({
                'face_id': i,
                'emotion': emotion,
                'confidence': float(confidence),
                'face_location': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                }
            })
        
        # Update session with latest emotion
        if predictions and session_id in active_sessions:
            active_sessions[session_id]['last_emotion'] = predictions[0]['emotion']
        
        # Send prediction back to client
        response = {
            'timestamp': data.get('timestamp', time.time()),
            'frame_id': data.get('frame_id'),
            'faces': predictions,
            'face_count': len(predictions),
            'processing_time': time.time()
        }
        
        emit('emotion_result', response)
        
    except Exception as e:
        emit('error', {
            'error': 'Processing failed',
            'message': str(e)
        })


@socketio.on('get_session_stats')
def handle_get_stats():
    """Get statistics for current session"""
    session_id = request.sid
    if session_id in active_sessions:
        session = active_sessions[session_id]
        duration = time.time() - session['connected_at']
        fps = session['frame_count'] / duration if duration > 0 else 0
        
        emit('session_stats', {
            'session_id': session_id,
            'duration': duration,
            'frame_count': session['frame_count'],
            'fps': fps,
            'last_emotion': session['last_emotion']
        })
    else:
        emit('error', {'error': 'Session not found'})


if __name__ == '__main__':
    # For local development with WebSocket support
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting Emotion Detection API with WebSocket support on port {port}")
    print(f"   REST API: http://0.0.0.0:{port}")
    print(f"   WebSocket: ws://0.0.0.0:{port}/socket.io")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
