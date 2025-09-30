import asyncio
from bleak import BleakClient

ESP32_ADDRESS = "58:8C:81:A4:9C:B2"  # <-- Replace with your ESP32 MAC
CHARACTERISTIC_UUID_RX = "492024F-H4P4-6971-0AE02MBL039"

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
