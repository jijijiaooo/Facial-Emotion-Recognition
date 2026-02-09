// EmotionStreamActivity.kt
// Android app that receives real-time emotion updates from Azure server

package com.example.emotiondetection

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import okhttp3.*
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*

class EmotionStreamActivity : AppCompatActivity() {
    
    // Server configuration
    private val SERVER_URL = "https://emotion-detection-api-g9budncvekdgewbk.eastasia-01.azurewebsites.net"
    private val RTSP_URL = "rtsp://admin:admin@192.168.0.100:8554/live"
    
    private val client = OkHttpClient()
    private var webSocket: WebSocket? = null
    private val scope = CoroutineScope(Dispatchers.Main + Job())
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_emotion_stream)
        
        // Start RTSP stream on server
        startRTSPStream()
        
        // Connect to WebSocket for real-time updates
        connectWebSocket()
    }
    
    private fun startRTSPStream() {
        scope.launch(Dispatchers.IO) {
            try {
                val url = "$SERVER_URL/rtsp/start?rtsp_url=${RTSP_URL}&fps=2"
                
                val request = Request.Builder()
                    .url(url)
                    .post(RequestBody.create(null, ByteArray(0)))
                    .build()
                
                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()
                
                withContext(Dispatchers.Main) {
                    if (response.isSuccessful) {
                        Log.i(TAG, "✓ RTSP stream started: $responseBody")
                        statusText.text = "Camera connected - Detecting emotions..."
                    } else {
                        Log.e(TAG, "Failed to start RTSP: $responseBody")
                        statusText.text = "Error: Failed to connect to camera"
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error starting RTSP: ${e.message}")
                withContext(Dispatchers.Main) {
                    statusText.text = "Error: ${e.message}"
                }
            }
        }
    }
    
    private fun connectWebSocket() {
        val wsUrl = SERVER_URL.replace("https://", "wss://") + "/ws/emotions"
        
        val request = Request.Builder()
            .url(wsUrl)
            .build()
        
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(TAG, "WebSocket connected")
                runOnUiThread {
                    connectionStatus.text = "● Connected"
                    connectionStatus.setTextColor(getColor(android.R.color.holo_green_dark))
                }
            }
            
            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d(TAG, "Received: $text")
                
                try {
                    val json = JSONObject(text)
                    
                    // Check message type
                    val type = json.optString("type", "")
                    if (type == "connected" || type == "ping") {
                        // Ignore connection messages
                        if (type == "ping") {
                            webSocket.send("pong")  // Respond to keep-alive
                        }
                        return
                    }
                    
                    // Parse emotion detection result
                    val success = json.optBoolean("success", false)
                    
                    if (success) {
                        val facesDetected = json.optInt("faces_detected", 0)
                        val timestamp = json.optString("timestamp", "")
                        
                        if (facesDetected > 0) {
                            val results = json.getJSONArray("results")
                            val firstResult = results.getJSONObject(0)
                            
                            val emotion = firstResult.getString("emotion")
                            val confidence = firstResult.getDouble("confidence")
                            
                            runOnUiThread {
                                updateUI(emotion, confidence, timestamp)
                            }
                        } else {
                            runOnUiThread {
                                emotionText.text = "No face detected"
                                confidenceText.text = ""
                                emotionEmoji.text = "👤"
                            }
                        }
                    }
                    
                } catch (e: Exception) {
                    Log.e(TAG, "Error parsing message: ${e.message}")
                }
            }
            
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket error: ${t.message}")
                runOnUiThread {
                    connectionStatus.text = "● Disconnected"
                    connectionStatus.setTextColor(getColor(android.R.color.holo_red_dark))
                    statusText.text = "Connection lost - Retrying..."
                }
                
                // Reconnect after 3 seconds
                scope.launch {
                    delay(3000)
                    connectWebSocket()
                }
            }
            
            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "WebSocket closed: $reason")
            }
        })
    }
    
    private fun updateUI(emotion: String, confidence: Double, timestamp: String) {
        // Update emotion display
        emotionText.text = emotion
        confidenceText.text = "${(confidence * 100).toInt()}%"
        emotionEmoji.text = getEmoji(emotion)
        
        // Update timestamp
        try {
            val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.getDefault())
            val date = sdf.parse(timestamp.substring(0, 19))
            val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
            timestampText.text = "Last updated: ${timeFormat.format(date)}"
        } catch (e: Exception) {
            timestampText.text = "Last updated: $timestamp"
        }
        
        // Visual feedback
        containerCard.setCardBackgroundColor(getColorForEmotion(emotion))
    }
    
    private fun getEmoji(emotion: String): String {
        return when (emotion) {
            "Happy" -> "😊"
            "Sad" -> "😢"
            "Angry" -> "😠"
            "Surprise" -> "😲"
            "Fear" -> "😨"
            "Disgust" -> "🤢"
            "Neutral" -> "😐"
            else -> "😊"
        }
    }
    
    private fun getColorForEmotion(emotion: String): Int {
        return when (emotion) {
            "Happy" -> getColor(R.color.happy_green)
            "Sad" -> getColor(R.color.sad_blue)
            "Angry" -> getColor(R.color.angry_red)
            "Surprise" -> getColor(R.color.surprise_yellow)
            "Fear" -> getColor(R.color.fear_purple)
            "Disgust" -> getColor(R.color.disgust_brown)
            else -> getColor(R.color.neutral_gray)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        
        // Stop RTSP stream
        scope.launch(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$SERVER_URL/rtsp/stop")
                    .post(RequestBody.create(null, ByteArray(0)))
                    .build()
                
                client.newCall(request).execute()
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping RTSP: ${e.message}")
            }
        }
        
        // Close WebSocket
        webSocket?.close(1000, "App closed")
        scope.cancel()
    }
    
    companion object {
        private const val TAG = "EmotionStream"
    }
}


// Layout XML (res/layout/activity_emotion_stream.xml)
/*
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:background="@android:color/background_dark"
    android:padding="24dp">

    <TextView
        android:id="@+id/connectionStatus"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="● Connecting..."
        android:textColor="@android:color/holo_orange_light"
        android:textSize="14sp"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintTop_toTopOf="parent" />

    <TextView
        android:id="@+id/titleText"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Real-Time Emotion Detection"
        android:textColor="@android:color/white"
        android:textSize="24sp"
        android:textStyle="bold"
        android:layout_marginTop="40dp"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintTop_toTopOf="parent" />

    <androidx.cardview.widget.CardView
        android:id="@+id/containerCard"
        android:layout_width="0dp"
        android:layout_height="300dp"
        android:layout_marginTop="40dp"
        app:cardCornerRadius="20dp"
        app:cardElevation="8dp"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toBottomOf="@id/titleText">

        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="match_parent"
            android:gravity="center"
            android:orientation="vertical"
            android:padding="24dp">

            <TextView
                android:id="@+id/emotionEmoji"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="😊"
                android:textSize="120sp" />

            <TextView
                android:id="@+id/emotionText"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="Detecting..."
                android:textColor="@android:color/black"
                android:textSize="32sp"
                android:textStyle="bold"
                android:layout_marginTop="16dp" />

            <TextView
                android:id="@+id/confidenceText"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text=""
                android:textColor="@android:color/darker_gray"
                android:textSize="24sp"
                android:layout_marginTop="8dp" />

        </LinearLayout>

    </androidx.cardview.widget.CardView>

    <TextView
        android:id="@+id/timestampText"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Last updated: --:--:--"
        android:textColor="@android:color/white"
        android:textSize="14sp"
        android:layout_marginTop="24dp"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintTop_toBottomOf="@id/containerCard" />

    <TextView
        android:id="@+id/statusText"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Initializing camera..."
        android:textColor="@android:color/holo_blue_light"
        android:textSize="16sp"
        android:layout_marginBottom="40dp"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintBottom_toBottomOf="parent" />

</androidx.constraintlayout.widget.ConstraintLayout>
*/

// Colors (res/values/colors.xml)
/*
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <color name="happy_green">#C8E6C9</color>
    <color name="sad_blue">#BBDEFB</color>
    <color name="angry_red">#FFCDD2</color>
    <color name="surprise_yellow">#FFF9C4</color>
    <color name="fear_purple">#E1BEE7</color>
    <color name="disgust_brown">#D7CCC8</color>
    <color name="neutral_gray">#E0E0E0</color>
</resources>
*/
