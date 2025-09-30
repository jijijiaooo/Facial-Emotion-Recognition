import requests

ESP32_IP = "192.168.0.107"  # <-- Replace with your ESP32's IP address
EMOJI_MAP = {
    "Happy": "??",
    "Sad": "??",
    "Angry": "??",
    "Fear": "??",
    "Surprise": "??",
    "Disgust": "??",
    "Neutral": "??"
}

def send_emotion_wifi(emotion):
    emoji = EMOJI_MAP.get(emotion, "??")
    msg = f"{emoji} {emotion}"
    url = f"http://{ESP32_IP}:8080/emotion"
    try:
        response = requests.post(url, data=msg.encode())
        print(f"Sent to ESP32: {msg}, Response: {response.text}")
    except Exception as e:
        print(f"WiFi send error: {e}")

# Example usage:
if __name__ == "__main__":
    send_emotion_wifi("Happy")