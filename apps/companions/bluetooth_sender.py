import asyncio
from bleak import BleakClient

ESP32_ADDRESS = "58:8c:81:a4:9c:b0"  # <-- Replace with your ESP32 MAC
CHARACTERISTIC_UUID_RX = "04db1a64-c4c2-4c8f-9065-c0b16f8298d5"

EMOJI_MAP = {
    "Happy": "😄",
    "Sad": "😢",
    "Angry": "😠",
    "Fear": "😱",
    "Surprise": "😲",
    "Disgust": "🤢",
    "Neutral": "😐"
}

async def send_emotion_ble(emotion):
    """Send emotion and emoji to ESP32-C3 via BLE"""
    emoji = EMOJI_MAP.get(emotion, "🙂")
    msg = f"{emoji} {emotion}"
    try:
        async with BleakClient(ESP32_ADDRESS) as client:
            await client.write_gatt_char(CHARACTERISTIC_UUID_RX, msg.encode())
            print(f"Sent to ESP32: {msg}")
    except Exception as e:
        print(f"BLE send error: {e}")