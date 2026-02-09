# SRT Stream Setup and Usage Guide

## Overview
The emotion detection API now supports **SRT (Secure Reliable Transport)** streaming from your Azure VM, providing reliable video transport over the internet with automatic error correction and low latency.

## Architecture

```
Azure VM (edge-camera-vm)              Your Mac / Phone
┌─────────────────────┐               ┌──────────────────┐
│                     │               │                  │
│  MediaMTX Server    │   SRT Stream  │  Emotion API     │
│  (Port 8890)        │──────────────>│  (FastAPI)       │
│                     │               │                  │
│  57.158.24.176      │               │  Processes and   │
│                     │               │  Returns Results │
└─────────────────────┘               └──────────────────┘
                                              │
                                              │ HTTP/WebSocket
                                              ▼
                                      ┌──────────────────┐
                                      │  Mobile Phone    │
                                      │  (Your App)      │
                                      └──────────────────┘
```

## Setup Instructions

### 1. Start MediaMTX on Azure VM

SSH into your Azure VM:
```bash
ssh PelioScope@57.158.24.176
# Password: PELi0Sc0p3_2025
```

Start the MediaMTX server:
```bash
./mediamtx
```

This will start the SRT server on port 8890 with stream ID `camera1`.

### 2. SRT Stream URL

Your SRT stream URL is:
```
srt://57.158.24.176:8890?streamid=read:camera1
```

**Components:**
- `srt://` - Protocol
- `57.158.24.176` - Azure VM public IP
- `8890` - SRT port
- `?streamid=read:camera1` - Stream identifier

## API Usage

### Start SRT Stream Processing

**Endpoint:** `POST /stream/start`

```bash
curl -X POST "http://localhost:8000/stream/start?stream_url=srt://57.158.24.176:8890?streamid=read:camera1&fps=2"
```

**Response:**
```json
{
  "success": true,
  "message": "SRT stream processing started",
  "stream_url": "srt://57.158.24.176:8890?streamid=read:camera1",
  "protocol": "SRT",
  "fps": 2
}
```

### Check Stream Status

**Endpoint:** `GET /stream/status`

```bash
curl "http://localhost:8000/stream/status"
```

**Response:**
```json
{
  "running": true,
  "protocol": "SRT",
  "stream_url": "srt://57.158.24.176:8890?streamid=read:camera1",
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
    },
    "timestamp": "2026-02-07T10:30:45.123456",
    "camera": "srt_stream",
    "stream_url": "srt://57.158.24.176:8890?streamid=read:camera1"
  }
}
```

### Stop Stream Processing

**Endpoint:** `POST /stream/stop`

```bash
curl -X POST "http://localhost:8000/stream/stop"
```

### WebSocket for Real-Time Updates

Connect to WebSocket for continuous emotion updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/emotions');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Emotion:', data.dominant_emotion);
    console.log('Confidence:', data.confidence);
    console.log('All emotions:', data.all_emotions);
};

ws.onopen = () => {
    console.log('Connected to emotion stream');
};
```

## Python Client Example

```python
import requests
import json

# API base URL
API_URL = "http://localhost:8000"

# SRT stream configuration
SRT_URL = "srt://57.158.24.176:8890?streamid=read:camera1"
FPS = 2  # Process 2 frames per second

def start_stream():
    """Start SRT stream processing"""
    response = requests.post(
        f"{API_URL}/stream/start",
        params={
            "stream_url": SRT_URL,
            "fps": FPS
        }
    )
    print("Start response:", response.json())

def get_status():
    """Get current stream status and latest emotion"""
    response = requests.get(f"{API_URL}/stream/status")
    data = response.json()
    
    if data["running"]:
        result = data["latest_result"]
        if result:
            print(f"Emotion: {result['dominant_emotion']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"All emotions: {result['all_emotions']}")
    else:
        print("Stream not running")

def stop_stream():
    """Stop stream processing"""
    response = requests.post(f"{API_URL}/stream/stop")
    print("Stop response:", response.json())

if __name__ == "__main__":
    # Start the stream
    start_stream()
    
    # Check status
    import time
    time.sleep(2)  # Wait for first results
    get_status()
    
    # Stop when done
    # stop_stream()
```

## Mobile App Integration

### For Your Android/iOS App

**1. Start the stream when app opens:**
```kotlin
// Kotlin/Android example
val apiUrl = "https://your-api.azurewebsites.net"
val srtUrl = "srt://57.158.24.176:8890?streamid=read:camera1"

lifecycleScope.launch {
    val response = apiClient.post("$apiUrl/stream/start") {
        parameter("stream_url", srtUrl)
        parameter("fps", 2)
    }
    // Stream is now processing
}
```

**2. Get emotion results via HTTP polling:**
```kotlin
// Poll every second
launch {
    while (isActive) {
        val status = apiClient.get("$apiUrl/stream/status")
        val emotion = status["latest_result"]["dominant_emotion"]
        updateUI(emotion)
        delay(1000)  // 1 second
    }
}
```

**3. Or use WebSocket for real-time updates:**
```kotlin
val webSocket = OkHttpClient().newWebSocket(
    Request.Builder().url("ws://your-api.azurewebsites.net/ws/emotions").build(),
    object : WebSocketListener() {
        override fun onMessage(webSocket: WebSocket, text: String) {
            val result = JSONObject(text)
            val emotion = result.getString("dominant_emotion")
            runOnUiThread { updateUI(emotion) }
        }
    }
)
```

## Backward Compatibility

The old `/rtsp/` endpoints still work for backward compatibility:
- `POST /rtsp/start?rtsp_url=...`
- `POST /rtsp/stop`
- `GET /rtsp/status`

These redirect to the new `/stream/` endpoints.

## Troubleshooting

### Stream fails to connect
1. Verify MediaMTX is running on Azure VM: `ssh PelioScope@57.158.24.176` then check `./mediamtx`
2. Check firewall allows port 8890
3. Test SRT URL with ffplay: `ffplay srt://57.158.24.176:8890?streamid=read:camera1`

### No results returned
- Wait 1-2 seconds for first frame processing
- Check `/stream/status` to see if stream is running
- Verify FPS setting isn't too high (recommend 1-2 FPS)

### High latency
- Reduce FPS to 1 (process 1 frame per second)
- Check network bandwidth
- MediaMTX includes automatic buffering and error correction

## Azure VM Details

- **Resource Group:** rg-edge-camera  
- **VM Name:** edge-camera-vm  
- **Public IP:** 57.158.24.176  
- **Location:** East Asia  
- **Size:** Standard B2ats v2 (2 cores, 1 GB memory)  
- **OS:** Linux  

## Benefits of SRT vs RTSP

1. **Better for Internet:** SRT handles packet loss and network jitter
2. **Lower Latency:** Optimized for real-time streaming
3. **Encryption:** Built-in security (optional)
4. **Firewall Friendly:** Uses UDP, easier to configure
5. **Auto Recovery:** Automatic error correction

## Next Steps

1. ✅ Start MediaMTX on Azure VM
2. ✅ Start your emotion detection API (locally or on Azure)
3. ✅ Call `/stream/start` with SRT URL
4. ✅ Get results via `/stream/status` or WebSocket
5. ✅ Display emotions in your mobile app

---

**Last Updated:** February 7, 2026  
**API Version:** 1.0.0
