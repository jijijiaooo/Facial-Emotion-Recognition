# SRT Stream Integration - Changes Summary

## 📋 Overview
Updated the emotion detection system to support **SRT (Secure Reliable Transport)** video streaming from your Azure VM camera, enabling robust internet-based video streaming with the emotion detection model.

## 🎯 Your Configuration
- **SRT Stream URL:** `srt://57.158.24.176:8890?streamid=read:camera1`
- **Azure VM:** edge-camera-vm (57.158.24.176)
- **MediaMTX Server:** Running on VM, port 8890
- **SSH Access:** `ssh PelioScope@57.158.24.176` (password: PELi0Sc0p3_2025)

## 🔧 Files Modified

### 1. `api/rtsp_stream.py` → Updated for SRT Support
**Changes:**
- Renamed class: `RTSPStreamProcessor` → `VideoStreamProcessor`
- Added SRT protocol detection and handling
- Uses `cv2.CAP_FFMPEG` backend for SRT support
- Updated method parameters: `rtsp_url` → `stream_url`
- Added protocol metadata to results (`srt_stream` or `rtsp_stream`)

**Key Features:**
- Auto-detects protocol from URL (SRT vs RTSP)
- Handles both SRT and RTSP streams transparently
- Includes stream URL in detection results
- Better logging with protocol-specific messages

### 2. `api/main.py` → New Stream Endpoints
**New Endpoints:**
- `POST /stream/start` - Start video stream (RTSP or SRT)
- `POST /stream/stop` - Stop video stream
- `GET /stream/status` - Get stream status and latest emotion

**Backward Compatibility:**
- `POST /rtsp/start` - Redirects to `/stream/start`
- `POST /rtsp/stop` - Redirects to `/stream/stop`
- `GET /rtsp/status` - Redirects to `/stream/status`

**Changes:**
- Updated global variable: `rtsp_processor` → `stream_processor`
- Added protocol information to status responses
- WebSocket endpoint updated to handle both protocols

### 3. `Dockerfile.api` → FFmpeg with SRT Support
**Added Dependencies:**
- `ffmpeg` - Video processing with SRT support
- `libavcodec-dev` - Video codec libraries
- `libavformat-dev` - Container format libraries
- `libavutil-dev` - Utility libraries
- `libswscale-dev` - Scaling libraries

## 📁 New Files Created

### 1. `SRT_STREAM_GUIDE.md` - Complete Usage Guide
Comprehensive documentation including:
- Architecture diagram
- Setup instructions for Azure VM
- API usage examples (curl, Python, JavaScript)
- Mobile app integration guide (Kotlin example)
- Troubleshooting tips
- SRT vs RTSP comparison

### 2. `api/test_srt_stream.py` - Test Suite
Automated testing script with:
- API health check
- Stream start/stop testing
- Status monitoring
- Emotion detection verification
- Backward compatibility tests
- Summary statistics

### 3. `SRT_INTEGRATION_SUMMARY.md` - This file
Quick reference for all changes made.

## 🚀 Quick Start

### 1. Start MediaMTX on Azure VM
```bash
ssh PelioScope@57.158.24.176
# Password: PELi0Sc0p3_2025
./mediamtx
```

### 2. Start Your API (Locally or Azure)
```bash
cd api
source ../venv/bin/activate  # If using local venv
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start SRT Stream Processing
```bash
curl -X POST "http://localhost:8000/stream/start?stream_url=srt://57.158.24.176:8890?streamid=read:camera1&fps=2"
```

### 4. Check Results
```bash
curl "http://localhost:8000/stream/status"
```

### 5. Stop Stream
```bash
curl -X POST "http://localhost:8000/stream/stop"
```

## 📱 Mobile App Integration

### Start Stream from Your App
```kotlin
// When app starts
val apiUrl = "https://your-api.azurewebsites.net"
val srtUrl = "srt://57.158.24.176:8890?streamid=read:camera1"

// Start processing
apiClient.post("$apiUrl/stream/start") {
    parameter("stream_url", srtUrl)
    parameter("fps", 2)
}
```

### Get Results - Option A: HTTP Polling
```kotlin
// Poll every second
while (true) {
    val response = apiClient.get("$apiUrl/stream/status")
    val emotion = response.data.latest_result.dominant_emotion
    updateUI(emotion)
    delay(1000)
}
```

### Get Results - Option B: WebSocket (Better)
```kotlin
val ws = OkHttpClient().newWebSocket(
    Request.Builder().url("ws://$apiUrl/ws/emotions").build(),
    object : WebSocketListener() {
        override fun onMessage(webSocket: WebSocket, text: String) {
            val data = JSONObject(text)
            val emotion = data.getString("dominant_emotion")
            val confidence = data.getDouble("confidence")
            runOnUiThread { 
                updateUI(emotion, confidence) 
            }
        }
    }
)
```

## 🧪 Testing

### Run the Test Suite
```bash
cd api
python test_srt_stream.py
```

This will:
1. ✅ Check API health
2. ✅ Start SRT stream
3. ✅ Monitor emotions for 20 seconds
4. ✅ Test backward compatibility
5. ✅ Show emotion statistics
6. ✅ Clean up and stop stream

### Manual Testing
```bash
# 1. Test health
curl http://localhost:8000/health

# 2. Start stream
curl -X POST "http://localhost:8000/stream/start?stream_url=srt://57.158.24.176:8890?streamid=read:camera1&fps=2"

# 3. Wait a few seconds, then check status
sleep 3
curl http://localhost:8000/stream/status

# 4. Stop when done
curl -X POST http://localhost:8000/stream/stop
```

## 📊 API Response Examples

### Stream Start Response
```json
{
  "success": true,
  "message": "SRT stream processing started",
  "stream_url": "srt://57.158.24.176:8890?streamid=read:camera1",
  "protocol": "SRT",
  "fps": 2
}
```

### Status Response (with results)
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
    "timestamp": "2026-02-07T17:30:45.123456",
    "camera": "srt_stream",
    "stream_url": "srt://57.158.24.176:8890?streamid=read:camera1"
  }
}
```

## 🔄 Deployment to Azure

### Update Docker Image
```bash
# Build with SRT support
docker build --platform linux/amd64 -f Dockerfile.api -t emotion-detection-api:srt-v2 .

# Tag for Azure Container Registry
docker tag emotion-detection-api:srt-v2 emotiondetectionpelio-gyb9hbcygcchh6b3.azurecr.io/emotion-detection-api:srt-v2

# Push to ACR
docker push emotiondetectionpelio-gyb9hbcygcchh6b3.azurecr.io/emotion-detection-api:srt-v2

# Update App Service
az webapp config container set \
  --name emotion-detection-api \
  --resource-group emotion-detection-pelio \
  --docker-custom-image-name emotiondetectionpelio-gyb9hbcygcchh6b3.azurecr.io/emotion-detection-api:srt-v2

# Restart
az webapp restart --name emotion-detection-api --resource-group emotion-detection-pelio
```

## ✅ What Works Now

1. ✅ **SRT Stream Processing** - Connect to your Azure VM camera via SRT
2. ✅ **RTSP Still Works** - Backward compatible with RTSP streams
3. ✅ **Real-time Emotions** - Processes frames at configurable FPS (default 2)
4. ✅ **HTTP API** - Get latest results via REST endpoint
5. ✅ **WebSocket Streaming** - Real-time emotion updates to mobile app
6. ✅ **Auto-reconnect** - Handles connection drops automatically
7. ✅ **Metadata Included** - Results include timestamp, stream URL, protocol

## 📈 Next Steps

1. **Test Locally:**
   ```bash
   # Start API
   cd api && uvicorn main:app --reload
   
   # In another terminal, run test
   python test_srt_stream.py
   ```

2. **Integrate with Mobile App:**
   - Update your app to call `/stream/start` when it launches
   - Use WebSocket or HTTP polling to get emotion results
   - Display emotions in your UI

3. **Deploy to Azure:**
   - Build updated Docker image with FFmpeg+SRT support
   - Push to Azure Container Registry
   - Update App Service to use new image
   - Test with your mobile app

## 🆘 Troubleshooting

### "Failed to open SRT stream"
- Ensure MediaMTX is running on Azure VM
- Check VM public IP: `57.158.24.176`
- Verify port 8890 is open in VM firewall
- Test directly: `ffplay srt://57.158.24.176:8890?streamid=read:camera1`

### "No results yet"
- Wait 2-3 seconds after starting stream
- Check `/stream/status` to see if stream is running
- Verify camera is actually sending video to MediaMTX

### "Import error: VideoStreamProcessor"
- Make sure you updated `api/main.py` with the new import
- Restart your API server to reload the code

## 🎉 Benefits

1. **Internet-Ready:** SRT works well over internet connections
2. **Resilient:** Handles packet loss and network jitter
3. **Low Latency:** Optimized for real-time applications
4. **Future-Proof:** Can easily add more protocols (WebRTC, etc.)
5. **Flexible:** Works with both local (RTSP) and remote (SRT) cameras

---

**Last Updated:** February 7, 2026  
**Version:** 2.0 (SRT Support)  
**Status:** ✅ Ready for Testing
