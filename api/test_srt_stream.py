#!/usr/bin/env python3
"""
Test script for SRT stream emotion detection
Tests the API with SRT stream from Azure VM
"""

import requests
import time
import json

# Configuration
API_URL = "http://localhost:8000"
SRT_URL = "srt://57.158.24.176:8890?streamid=read:camera1"
FPS = 2

def test_health():
    """Test if API is running"""
    print("\n🔍 Testing API health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"✅ API Status: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ API not responding: {e}")
        return False

def start_stream():
    """Start SRT stream processing"""
    print(f"\n▶️  Starting SRT stream...")
    print(f"   URL: {SRT_URL}")
    print(f"   FPS: {FPS}")
    
    try:
        response = requests.post(
            f"{API_URL}/stream/start",
            params={
                "stream_url": SRT_URL,
                "fps": FPS
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Stream started successfully")
            print(f"   Protocol: {result.get('protocol', 'Unknown')}")
            print(f"   Message: {result.get('message', '')}")
            return True
        else:
            print(f"❌ Failed to start stream: {response.status_code}")
            print(f"   {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting stream: {e}")
        return False

def check_status():
    """Check stream status and get latest emotion result"""
    try:
        response = requests.get(f"{API_URL}/stream/status", timeout=5)
        data = response.json()
        
        print(f"\n📊 Stream Status:")
        print(f"   Running: {data.get('running', False)}")
        print(f"   Protocol: {data.get('protocol', 'N/A')}")
        
        if data.get("running") and data.get("latest_result"):
            result = data["latest_result"]
            print(f"\n😊 Latest Emotion Detection:")
            print(f"   Emotion: {result.get('dominant_emotion', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
            print(f"   Timestamp: {result.get('timestamp', 'N/A')}")
            
            if 'all_emotions' in result:
                print(f"\n   All Emotions:")
                for emotion, score in sorted(result['all_emotions'].items(), 
                                            key=lambda x: x[1], reverse=True):
                    print(f"      {emotion:10s}: {score:.2%}")
            
            return result
        else:
            print("   No results yet (stream may still be connecting...)")
            return None
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return None

def monitor_stream(duration: int = 30):
    """Monitor stream for specified duration"""
    print(f"\n👁️  Monitoring stream for {duration} seconds...")
    print("   (Press Ctrl+C to stop early)\n")
    
    start_time = time.time()
    last_emotion = None
    emotion_count = {}
    
    try:
        while time.time() - start_time < duration:
            result = check_status()
            
            if result:
                emotion = result.get('dominant_emotion')
                if emotion != last_emotion:
                    print(f"   🔄 Emotion changed: {last_emotion} → {emotion}")
                    last_emotion = emotion
                
                # Count emotions
                emotion_count[emotion] = emotion_count.get(emotion, 0) + 1
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    
    # Summary
    if emotion_count:
        print(f"\n📈 Emotion Summary:")
        total = sum(emotion_count.values())
        for emotion, count in sorted(emotion_count.items(), 
                                     key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            print(f"   {emotion:10s}: {count:3d} times ({percentage:.1f}%)")

def stop_stream():
    """Stop stream processing"""
    print(f"\n⏹️  Stopping stream...")
    
    try:
        response = requests.post(f"{API_URL}/stream/stop", timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Stream stopped successfully")
            return True
        else:
            print(f"❌ Failed to stop stream: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error stopping stream: {e}")
        return False

def test_backward_compatibility():
    """Test that old /rtsp/ endpoints still work"""
    print("\n🔄 Testing backward compatibility (/rtsp/ endpoints)...")
    
    try:
        # Test old endpoint
        response = requests.post(
            f"{API_URL}/rtsp/start",
            params={
                "rtsp_url": SRT_URL,  # Using SRT URL with old endpoint
                "fps": FPS
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Old /rtsp/start endpoint works")
            time.sleep(2)
            
            # Check status with old endpoint
            response = requests.get(f"{API_URL}/rtsp/status", timeout=5)
            if response.status_code == 200:
                print("✅ Old /rtsp/status endpoint works")
            
            # Stop with old endpoint
            response = requests.post(f"{API_URL}/rtsp/stop", timeout=5)
            if response.status_code == 200:
                print("✅ Old /rtsp/stop endpoint works")
            
            return True
        else:
            print(f"❌ Backward compatibility issue: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing backward compatibility: {e}")
        return False

def main():
    """Main test sequence"""
    print("=" * 60)
    print("SRT Stream Emotion Detection - Test Suite")
    print("=" * 60)
    
    # 1. Check API health
    if not test_health():
        print("\n❌ API is not running. Please start the API first:")
        print("   cd api && uvicorn main:app --reload")
        return
    
    # 2. Start stream
    if not start_stream():
        print("\n❌ Could not start stream. Check:")
        print("   1. MediaMTX is running on Azure VM")
        print("   2. SRT URL is correct")
        print("   3. Network/firewall allows connection")
        return
    
    # 3. Wait for first results
    print("\n⏳ Waiting for stream to connect and process first frames...")
    time.sleep(5)
    
    # 4. Check initial status
    check_status()
    
    # 5. Monitor for a bit
    monitor_stream(duration=20)
    
    # 6. Test backward compatibility
    stop_stream()
    time.sleep(1)
    test_backward_compatibility()
    
    # 7. Cleanup
    print("\n🧹 Cleaning up...")
    stop_stream()
    
    print("\n" + "=" * 60)
    print("✅ Test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        print("Stopping stream...")
        stop_stream()
