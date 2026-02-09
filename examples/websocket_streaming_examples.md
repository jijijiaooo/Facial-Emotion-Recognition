# Real-Time Video Streaming Examples

## Overview
The API now supports real-time video streaming via WebSockets, allowing continuous emotion detection from live camera feeds.

**API Endpoint:** `ws://your-app.onrender.com/socket.io`

---

## 🎯 JavaScript/React Native Example

```javascript
import io from 'socket.io-client';
import { Camera } from 'expo-camera';

// Connect to WebSocket server
const socket = io('https://your-app.onrender.com', {
  transports: ['websocket'],
  reconnection: true
});

// Handle connection
socket.on('connected', (data) => {
  console.log('Connected to emotion detection server:', data);
});

// Handle emotion results
socket.on('emotion_result', (result) => {
  console.log('Detected emotions:', result.faces);
  
  result.faces.forEach(face => {
    console.log(`Face ${face.face_id}: ${face.emotion} (${face.confidence})`);
    // Update UI with emotion data
    updateEmotionDisplay(face.emotion, face.confidence);
  });
});

// Handle errors
socket.on('error', (error) => {
  console.error('Error:', error);
});

// Capture and send video frames
const sendVideoFrame = async () => {
  try {
    // Capture frame from camera
    const photo = await camera.takePictureAsync({
      base64: true,
      quality: 0.7,
      skipProcessing: true
    });
    
    // Send to server
    socket.emit('video_frame', {
      image: photo.base64,
      timestamp: Date.now(),
      frame_id: frameCounter++
    });
  } catch (error) {
    console.error('Frame capture failed:', error);
  }
};

// Start streaming (send frames at 10 FPS)
const startStreaming = () => {
  streamInterval = setInterval(sendVideoFrame, 100); // 10 FPS
};

// Stop streaming
const stopStreaming = () => {
  if (streamInterval) {
    clearInterval(streamInterval);
  }
  socket.disconnect();
};

// Get session statistics
socket.emit('get_session_stats');
socket.on('session_stats', (stats) => {
  console.log('Session stats:', stats);
  console.log(`FPS: ${stats.fps.toFixed(2)}, Frames: ${stats.frame_count}`);
});
```

---

## 📱 React Native Complete Component

```jsx
import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Camera } from 'expo-camera';
import io from 'socket.io-client';

const EmotionDetectorStream = () => {
  const [emotion, setEmotion] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [faceCount, setFaceCount] = useState(0);
  
  const cameraRef = useRef(null);
  const socketRef = useRef(null);
  const streamIntervalRef = useRef(null);

  useEffect(() => {
    // Initialize socket connection
    socketRef.current = io('https://your-app.onrender.com', {
      transports: ['websocket']
    });

    socketRef.current.on('connected', (data) => {
      console.log('Connected:', data.message);
    });

    socketRef.current.on('emotion_result', (result) => {
      if (result.faces.length > 0) {
        const mainFace = result.faces[0];
        setEmotion(mainFace.emotion);
        setConfidence(mainFace.confidence);
        setFaceCount(result.face_count);
      } else {
        setFaceCount(0);
      }
    });

    return () => {
      stopStreaming();
      socketRef.current.disconnect();
    };
  }, []);

  const sendFrame = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.6,
        skipProcessing: true
      });

      socketRef.current.emit('video_frame', {
        image: photo.base64,
        timestamp: Date.now()
      });
    }
  };

  const startStreaming = () => {
    setIsStreaming(true);
    streamIntervalRef.current = setInterval(sendFrame, 100); // 10 FPS
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
    }
  };

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={styles.camera}
        type={Camera.Constants.Type.front}
      />
      
      <View style={styles.overlay}>
        {emotion && (
          <View style={styles.emotionBox}>
            <Text style={styles.emotionText}>{emotion}</Text>
            <Text style={styles.confidenceText}>
              {(confidence * 100).toFixed(1)}%
            </Text>
            <Text style={styles.faceCountText}>
              {faceCount} face{faceCount !== 1 ? 's' : ''} detected
            </Text>
          </View>
        )}
        
        <TouchableOpacity
          style={[styles.button, isStreaming && styles.buttonActive]}
          onPress={isStreaming ? stopStreaming : startStreaming}
        >
          <Text style={styles.buttonText}>
            {isStreaming ? 'Stop Streaming' : 'Start Streaming'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  camera: { flex: 1 },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'space-between',
    padding: 20
  },
  emotionBox: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 20,
    borderRadius: 10,
    alignItems: 'center'
  },
  emotionText: {
    color: 'white',
    fontSize: 32,
    fontWeight: 'bold'
  },
  confidenceText: {
    color: '#4CAF50',
    fontSize: 24,
    marginTop: 5
  },
  faceCountText: {
    color: '#FFF',
    fontSize: 14,
    marginTop: 5
  },
  button: {
    backgroundColor: '#2196F3',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center'
  },
  buttonActive: {
    backgroundColor: '#f44336'
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold'
  }
});

export default EmotionDetectorStream;
```

---

## 🐍 Python Client Example

```python
import socketio
import cv2
import base64
import time
from io import BytesIO
from PIL import Image

# Create Socket.IO client
sio = socketio.Client()

# Event handlers
@sio.on('connected')
def on_connect(data):
    print(f"Connected: {data['message']}")

@sio.on('emotion_result')
def on_emotion_result(data):
    print(f"\n📊 Frame {data['frame_id']}")
    print(f"   Faces detected: {data['face_count']}")
    
    for face in data['faces']:
        print(f"   Face {face['face_id']}: {face['emotion']} ({face['confidence']:.2f})")

@sio.on('error')
def on_error(data):
    print(f"❌ Error: {data['message']}")

# Connect to server
sio.connect('https://your-app.onrender.com')

# Open webcam
cap = cv2.VideoCapture(0)
frame_id = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Send frame to server
        sio.emit('video_frame', {
            'image': base64_image,
            'timestamp': time.time(),
            'frame_id': frame_id
        })
        
        frame_id += 1
        
        # Control frame rate (10 FPS)
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n\nStopping...")
finally:
    cap.release()
    sio.disconnect()
```

---

## 🌐 Web Browser Example (HTML/JavaScript)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Real-Time Emotion Detection</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <h1>Real-Time Emotion Detection</h1>
    <video id="video" width="640" height="480" autoplay></video>
    <canvas id="canvas" style="display:none;"></canvas>
    
    <div id="results">
        <h2>Detected Emotion: <span id="emotion">-</span></h2>
        <h3>Confidence: <span id="confidence">-</span></h3>
        <p>Faces: <span id="faceCount">0</span></p>
    </div>
    
    <button onclick="startStreaming()">Start</button>
    <button onclick="stopStreaming()">Stop</button>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        let socket;
        let streamInterval;
        let frameId = 0;

        // Initialize webcam
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            });

        // Connect to WebSocket
        socket = io('https://your-app.onrender.com');

        socket.on('connected', (data) => {
            console.log('Connected:', data);
        });

        socket.on('emotion_result', (result) => {
            if (result.faces.length > 0) {
                const face = result.faces[0];
                document.getElementById('emotion').textContent = face.emotion;
                document.getElementById('confidence').textContent = 
                    (face.confidence * 100).toFixed(1) + '%';
            }
            document.getElementById('faceCount').textContent = result.face_count;
        });

        function captureAndSend() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            
            // Convert to base64
            const base64Image = canvas.toDataURL('image/jpeg', 0.7)
                .split(',')[1]; // Remove data:image/jpeg;base64, prefix
            
            // Send to server
            socket.emit('video_frame', {
                image: base64Image,
                timestamp: Date.now(),
                frame_id: frameId++
            });
        }

        function startStreaming() {
            streamInterval = setInterval(captureAndSend, 100); // 10 FPS
        }

        function stopStreaming() {
            clearInterval(streamInterval);
        }
    </script>
</body>
</html>
```

---

## 📊 Performance Tips

### Optimal Frame Rates
- **Mobile Apps:** 5-10 FPS (100-200ms intervals)
- **Web Apps:** 10-15 FPS (67-100ms intervals)
- **Desktop Apps:** 15-30 FPS (33-67ms intervals)

### Image Quality
- Use JPEG compression (quality: 60-80%)
- Resize images before sending (max 640x480 for mobile)
- Lower quality for slower connections

### Example with Optimization:

```javascript
const sendOptimizedFrame = async () => {
  const photo = await camera.takePictureAsync({
    base64: true,
    quality: 0.6,           // 60% quality
    skipProcessing: true,
    exif: false,
    width: 640,             // Resize to 640px width
  });
  
  socket.emit('video_frame', {
    image: photo.base64,
    timestamp: Date.now()
  });
};
```

---

## 🔧 Testing Locally

```bash
# Install dependencies
pip install -r requirements_api.txt

# Run the server
python app.py

# Server will start on:
# REST API: http://localhost:5000
# WebSocket: ws://localhost:5000/socket.io
```

---

## 🚀 Deploy to Render

Update your `Procfile`:
```
web: gunicorn --worker-class gevent -w 1 app:app
```

The WebSocket will automatically work on Render at:
```
wss://your-app.onrender.com/socket.io
```

---

## 📝 Event Reference

### Client → Server Events

| Event | Data | Description |
|-------|------|-------------|
| `video_frame` | `{image, timestamp, frame_id}` | Send video frame for processing |
| `get_session_stats` | - | Request session statistics |

### Server → Client Events

| Event | Data | Description |
|-------|------|-------------|
| `connected` | `{status, session_id, message}` | Connection successful |
| `emotion_result` | `{faces[], face_count, timestamp}` | Emotion detection results |
| `error` | `{error, message}` | Error occurred |
| `session_stats` | `{duration, frame_count, fps}` | Session statistics |

---

## 🎯 Response Format

```json
{
  "timestamp": 1704067200.123,
  "frame_id": 42,
  "faces": [
    {
      "face_id": 0,
      "emotion": "Happy",
      "confidence": 0.87,
      "face_location": {
        "x": 120,
        "y": 80,
        "width": 150,
        "height": 150
      }
    }
  ],
  "face_count": 1,
  "processing_time": 1704067200.456
}
```
