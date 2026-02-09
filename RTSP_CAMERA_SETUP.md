# RTSP Camera Stream to Azure - Complete Setup Guide

## Architecture
```
RTSP Camera (192.168.0.100) 
    ↓ RTSP stream
Azure Server (continuous processing)
    ↓ WebSocket (real-time)
Android App (displays emotions)
```

---

## Step 1: Update Azure Server

### 1.1 Add Dependencies
Already updated in `api/requirements.txt`:
- `websockets==12.0` (WebSocket support)

### 1.2 Rebuild Docker Image
```bash
cd /Users/jiaoshihlo/Codes/Facial-Emotion-Recognition-version-revised-dataset

# Build for Azure (AMD64)
docker build --platform linux/amd64 -f Dockerfile.api -t emotion-detection-api .

# Tag for Azure Container Registry
docker tag emotion-detection-api emotiondetectionpelio-gyb9hbcygcchh6b3.azurecr.io/emotion-detection-api:rtsp

# Push to ACR
docker push emotiondetectionpelio-gyb9hbcygcchh6b3.azurecr.io/emotion-detection-api:rtsp
```

### 1.3 Update Azure App Service
```bash
# Update container
az webapp config container set \
  --name emotion-detection-api-g9budncvekdgewbk \
  --resource-group emotion-detection-pelio \
  --docker-custom-image-name emotiondetectionpelio-gyb9hbcygcchh6b3.azurecr.io/emotion-detection-api:rtsp

# Restart app
az webapp restart \
  --name emotion-detection-api-g9budncvekdgewbk \
  --resource-group emotion-detection-pelio
```

---

## Step 2: Start RTSP Stream Processing

### Option A: Using API (Recommended)

Send POST request to start RTSP stream:

```bash
curl -X POST "https://emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net/rtsp/start?rtsp_url=rtsp://admin:admin@192.168.0.100:8554/live&fps=2"
```

Expected response:
```json
{
  "success": true,
  "message": "RTSP stream processing started",
  "rtsp_url": "rtsp://admin:admin@192.168.0.100:8554/live",
  "fps": 2
}
```

### Option B: Android App Auto-Start

The Android app automatically starts the RTSP stream when launched (see `EmotionStreamActivity.kt`)

---

## Step 3: Android App Setup

### 3.1 Add Dependencies to `build.gradle`

```gradle
dependencies {
    implementation 'com.squareup.okhttp3:okhttp:4.12.0'
    implementation 'androidx.cardview:cardview:1.0.0'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}
```

### 3.2 Add Permissions to `AndroidManifest.xml`

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

### 3.3 Copy Files

1. **Kotlin Code**: Copy [android/EmotionStreamActivity.kt](android/EmotionStreamActivity.kt) to your Android project
2. **Layout**: Create `res/layout/activity_emotion_stream.xml` (XML included in comments)
3. **Colors**: Create `res/values/colors.xml` (included in comments)

### 3.4 Update Server URL

In `EmotionStreamActivity.kt`, update:
```kotlin
private val SERVER_URL = "https://emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net"
private val RTSP_URL = "rtsp://admin:admin@192.168.0.100:8554/live"
```

---

## Step 4: Network Configuration

### 4.1 Camera Network Setup

**Option 1: Camera and Azure on Same Network (Best)**
- Camera: Connect to your Wi-Fi
- Azure: Can access camera's local IP (if allowed)
- ⚠️ **Problem**: Azure may not reach local 192.168.x.x IP

**Option 2: Port Forwarding (Recommended)**
1. Open router settings
2. Forward port `8554` to camera IP `192.168.0.100`
3. Use public IP in RTSP URL:
   ```
   rtsp://admin:admin@YOUR_PUBLIC_IP:8554/live
   ```

**Option 3: VPN/Tunnel (Most Secure)**
- Use ngrok or similar to expose RTSP stream
- Connect Azure to VPN endpoint

### 4.2 Update RTSP URL

After port forwarding, update:
```kotlin
// In Android app
private val RTSP_URL = "rtsp://admin:admin@YOUR_PUBLIC_IP:8554/live"
```

```bash
# When starting via API
curl -X POST "https://...azurewebsites.net/rtsp/start?rtsp_url=rtsp://admin:admin@YOUR_PUBLIC_IP:8554/live&fps=2"
```

---

## Step 5: Testing

### 5.1 Test RTSP Connection Locally First

```bash
# Install ffplay (on Mac)
brew install ffmpeg

# Test RTSP stream
ffplay rtsp://admin:admin@192.168.0.100:8554/live
```

### 5.2 Test WebSocket Connection

```bash
# Install wscat
npm install -g wscat

# Connect to WebSocket
wscat -c wss://emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net/ws/emotions
```

You should see:
```json
{"type":"connected","message":"Connected to emotion detection stream",...}
```

### 5.3 Check RTSP Status

```bash
curl https://emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net/rtsp/status
```

---

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rtsp/start` | POST | Start RTSP stream processing |
| `/rtsp/stop` | POST | Stop RTSP stream processing |
| `/rtsp/status` | GET | Get stream status and latest result |
| `/ws/emotions` | WebSocket | Real-time emotion updates |
| `/health` | GET | Server health check |
| `/predict` | POST | Single image prediction (existing) |

---

## How It Works

1. **Android app launches** → Calls `/rtsp/start`
2. **Azure server** → Connects to RTSP camera at `rtsp://admin:admin@192.168.0.100:8554/live`
3. **Server processes** → 2 frames per second (configurable via `fps` parameter)
4. **Server detects emotions** → Uses Enhanced Hybrid CNN model
5. **Server broadcasts** → Sends results to all connected WebSocket clients
6. **Android receives** → Updates UI with emotion + confidence in real-time

---

## Troubleshooting

### Issue: "Failed to open RTSP stream"
**Solutions**:
- ✅ Check camera is powered on
- ✅ Verify RTSP URL credentials (`admin:admin`)
- ✅ Test with ffplay locally first
- ✅ Check firewall/network restrictions
- ✅ Ensure Azure can reach camera IP (use port forwarding)

### Issue: "WebSocket disconnected"
**Solutions**:
- ✅ Check internet connection
- ✅ Verify server is running (`/health` endpoint)
- ✅ App automatically reconnects after 3 seconds

### Issue: "No faces detected"
**Solutions**:
- ✅ Point camera at face (proper lighting)
- ✅ Check camera angle/position
- ✅ Reduce FPS if processing is too slow

### Issue: "Slow performance"
**Solutions**:
- ✅ Reduce FPS: `fps=1` (1 frame per second)
- ✅ Upgrade Azure tier (currently B2)
- ✅ Lower camera resolution

---

## Performance

| Azure Tier | FPS | Response Time | Cost/Month |
|------------|-----|---------------|------------|
| B1 (1 core) | 0.5 | 2-3 seconds | $13 |
| **B2 (2 cores)** | **2** | **0.5-1 second** | **$26** ✓ Current |
| B3 (4 cores) | 4-5 | 0.2-0.5 second | $52 |

---

## Stop RTSP Stream

### Via API:
```bash
curl -X POST https://emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net/rtsp/stop
```

### Via Android:
App automatically stops stream in `onDestroy()`

---

## Security Notes

⚠️ **RTSP credentials in URL**:
- Current: `rtsp://admin:admin@...` (hardcoded)
- Production: Use environment variables or secure storage

⚠️ **Public IP exposure**:
- If using port forwarding, your camera is exposed
- Consider VPN or secure tunnel instead

⚠️ **WebSocket security**:
- Currently accepts all connections
- Add authentication for production

---

## Next Steps

1. ✅ Rebuild and deploy Docker image
2. ✅ Test RTSP connection locally
3. ✅ Set up port forwarding or VPN
4. ✅ Build Android app
5. ✅ Test end-to-end flow
6. 🚀 Deploy and enjoy real-time emotion detection!

---

**Your RTSP URL**: `rtsp://admin:admin@192.168.0.100:8554/live`  
**Azure Server**: `emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net`  
**WebSocket**: `wss://emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net/ws/emotions`
