# Mobile App Integration Guide

## 🎯 How It Works

```
Azure VM Camera → SRT Stream → Your API → Face Detection → Emotion Model → Mobile App
     (edge)         (video)      (Azure)    (OpenCV)      (TensorFlow)    (results)
```

## 📝 Step-by-Step Logic

### 1. **SRT Stream Processing** (api/rtsp_stream.py)

```python
class VideoStreamProcessor:
    def __init__(self, stream_url, detector, fps=2):
        self.stream_url = stream_url  # srt://57.158.24.176:8890?streamid=read:camera1
        self.detector = detector      # Your emotion detection model
        self.fps = fps                # Process 2 frames per second
        self.protocol = "SRT" if stream_url.startswith("srt://") else "RTSP"
    
    async def connect(self):
        # Connect to SRT stream using OpenCV with FFmpeg backend
        self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        return self.cap.isOpened()
    
    async def process_stream(self):
        while self.is_running:
            # 1. Read frame from SRT stream
            ret, frame = self.cap.read()
            
            # 2. Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 3. Detect faces and emotions
            result = self.detector.detect_emotion(rgb_frame)
            
            # 4. Store latest result
            self.latest_result = {
                "dominant_emotion": "happy",  # or sad, angry, etc.
                "confidence": 0.89,
                "all_emotions": {...},
                "timestamp": "2026-02-09T10:30:45",
                "camera": "srt_stream"
            }
```

### 2. **API Endpoints** (api/main.py)

```python
@app.post("/stream/start")
async def start_video_stream(stream_url: str, fps: int = 2):
    # Create stream processor
    stream_processor = VideoStreamProcessor(stream_url, detector, fps)
    
    # Start processing in background (non-blocking)
    asyncio.create_task(stream_and_broadcast())
    
    return {"success": True, "protocol": "SRT"}

@app.get("/stream/status")
async def get_stream_status():
    # Return latest emotion result
    return {
        "running": True,
        "protocol": "SRT",
        "latest_result": {
            "dominant_emotion": "happy",
            "confidence": 0.89,
            "all_emotions": {
                "happy": 0.89,
                "neutral": 0.06,
                "sad": 0.03,
                "angry": 0.01,
                "fear": 0.01,
                "disgust": 0.00
            }
        }
    }
```

## 📱 Mobile App Integration

### Option 1: HTTP Polling (Simple)

```kotlin
// Android Kotlin Example
class EmotionActivity : AppCompatActivity() {
    private val apiUrl = "https://emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net"
    private val srtUrl = "srt://57.158.24.176:8890?streamid=read:camera1"
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // 1. Start SRT stream processing (only once when app opens)
        lifecycleScope.launch {
            startStream()
            pollEmotions()
        }
    }
    
    private suspend fun startStream() {
        val response = apiClient.post("$apiUrl/stream/start") {
            parameter("stream_url", srtUrl)
            parameter("fps", 2)  // Process 2 frames/sec
        }
        Log.d("Emotion", "Stream started: ${response.message}")
    }
    
    private suspend fun pollEmotions() {
        while (isActive) {
            // Get latest emotion result
            val status = apiClient.get("$apiUrl/stream/status")
            
            if (status.running && status.latest_result != null) {
                val emotion = status.latest_result.dominant_emotion
                val confidence = status.latest_result.confidence
                
                // Update UI
                runOnUiThread {
                    emotionText.text = emotion
                    confidenceText.text = "${(confidence * 100).toInt()}%"
                    updateEmotionIcon(emotion)
                }
            }
            
            delay(1000)  // Poll every 1 second
        }
    }
}
```

### Option 2: WebSocket (Real-time, Better)

```kotlin
// Android Kotlin WebSocket Example
class EmotionActivity : AppCompatActivity() {
    private val apiUrl = "emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net"
    private val srtUrl = "srt://57.158.24.176:8890?streamid=read:camera1"
    private lateinit var webSocket: WebSocket
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        lifecycleScope.launch {
            // 1. Start stream processing
            startStream()
            
            // 2. Connect to WebSocket for real-time updates
            connectWebSocket()
        }
    }
    
    private suspend fun startStream() {
        apiClient.post("https://$apiUrl/stream/start") {
            parameter("stream_url", srtUrl)
            parameter("fps", 2)
        }
    }
    
    private fun connectWebSocket() {
        val client = OkHttpClient()
        val request = Request.Builder()
            .url("wss://$apiUrl/ws/emotions")
            .build()
        
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, text: String) {
                val result = JSONObject(text)
                
                // Parse emotion data
                val emotion = result.getString("dominant_emotion")
                val confidence = result.getDouble("confidence")
                val allEmotions = result.getJSONObject("all_emotions")
                
                // Update UI on main thread
                runOnUiThread {
                    emotionText.text = emotion
                    confidenceText.text = "${(confidence * 100).toInt()}%"
                    updateEmotionIcon(emotion)
                    updateEmotionChart(allEmotions)
                }
            }
            
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e("WebSocket", "Connection failed: ${t.message}")
            }
        })
    }
    
    override fun onDestroy() {
        super.onDestroy()
        webSocket.close(1000, "Activity destroyed")
    }
}
```

### Option 3: React Native (iOS/Android)

```javascript
import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';

const API_URL = 'https://emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net';
const SRT_URL = 'srt://57.158.24.176:8890?streamid=read:camera1';

export default function EmotionScreen() {
  const [emotion, setEmotion] = useState('neutral');
  const [confidence, setConfidence] = useState(0);

  useEffect(() => {
    // 1. Start stream when component mounts
    startStream();
    
    // 2. Poll for emotions
    const interval = setInterval(async () => {
      const status = await fetch(`${API_URL}/stream/status`);
      const data = await status.json();
      
      if (data.running && data.latest_result) {
        setEmotion(data.latest_result.dominant_emotion);
        setConfidence(data.latest_result.confidence);
      }
    }, 1000);  // Every 1 second
    
    return () => clearInterval(interval);
  }, []);
  
  const startStream = async () => {
    await fetch(`${API_URL}/stream/start?stream_url=${encodeURIComponent(SRT_URL)}&fps=2`, {
      method: 'POST'
    });
  };
  
  return (
    <View style={styles.container}>
      <Text style={styles.emotion}>{emotion}</Text>
      <Text style={styles.confidence}>{(confidence * 100).toFixed(0)}%</Text>
    </View>
  );
}
```

## 🔄 Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     MOBILE APP                              │
│                                                             │
│  1. On App Start:                                           │
│     POST /stream/start                                      │
│     - stream_url: srt://57.158.24.176:8890                  │
│     - fps: 2                                                │
│                                                             │
│  2. Every 1 second:                                         │
│     GET /stream/status                                      │
│     ↓                                                       │
│     Receive: {                                              │
│       dominant_emotion: "happy",                            │
│       confidence: 0.89,                                     │
│       all_emotions: {...}                                   │
│     }                                                       │
│     ↓                                                       │
│     Update UI with emotion                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                   AZURE API SERVICE                          │
│                                                             │
│  Background Process (runs continuously):                    │
│                                                             │
│  while stream is running:                                   │
│    1. Read frame from SRT stream                            │
│       ↓                                                     │
│    2. Detect faces using OpenCV                             │
│       ↓                                                     │
│    3. Run emotion detection model                           │
│       ↓                                                     │
│    4. Store latest result                                   │
│       ↓                                                     │
│    5. Broadcast to WebSocket clients (if any)               │
│       ↓                                                     │
│    Wait 0.5 seconds (fps=2)                                 │
│    Repeat...                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                  AZURE VM CAMERA                            │
│                                                             │
│  MediaMTX Server                                            │
│  - Captures video from camera                               │
│  - Streams via SRT protocol                                 │
│  - srt://57.158.24.176:8890?streamid=read:camera1           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## ✅ Key Points

1. **Your mobile app NEVER handles video** - It only receives emotion results
2. **API does all the heavy work** - Video processing, face detection, emotion recognition
3. **Lightweight mobile app** - Just HTTP requests and UI updates
4. **Real-time updates** - Results available within 0.5-1 second
5. **Low bandwidth** - Mobile only receives small JSON responses (~200 bytes)

## 🧪 Test It Now

```bash
# Test if API is running
curl https://emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net/health

# Start stream processing
curl -X POST "https://emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net/stream/start?stream_url=srt://57.158.24.176:8890?streamid=read:camera1&fps=2"

# Get emotion results
curl https://emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net/stream/status
```

## 📊 Response Example

```json
{
  "running": true,
  "protocol": "SRT",
  "stream_url": "srt://57.158.24.176:8890?streamid=read:camera1",
  "latest_result": {
    "dominant_emotion": "happy",
    "confidence": 0.8945,
    "all_emotions": {
      "happy": 0.8945,
      "neutral": 0.0623,
      "sad": 0.0287,
      "angry": 0.0089,
      "fear": 0.0034,
      "disgust": 0.0022
    },
    "timestamp": "2026-02-09T10:30:45.123456",
    "camera": "srt_stream",
    "stream_url": "srt://57.158.24.176:8890?streamid=read:camera1"
  }
}
```
