import os
import json
import requests
from typing import Optional, Dict, Any

# Default targets - can be overridden via environment variables or configure_targets()
ESP32_IP = os.environ.get("ESP32_IP", "192.168.0.107")
ESP32_PORT = int(os.environ.get("ESP32_PORT", "8080"))

# Optional Raspberry Pi (will forward to Android / phone UI). Leave empty to disable.
RPI_IP = os.environ.get("RPI_IP", "")  # e.g. "192.168.0.123"
RPI_PORT = int(os.environ.get("RPI_PORT", "5000"))

# Emoji map used for human-readable / display purposes
EMOJI_MAP = {
    "Happy": "^_^",
    "Sad": "T_T",
    "Angry": ">:[",
    "Fear": "O_O",
    "Surprise": "O.O",
    "Disgust": ">_<",
    "Neutral": "-_-"
}


def configure_targets(esp32_ip: Optional[str] = None,
                      rpi_ip: Optional[str] = None,
                      esp32_port: Optional[int] = None,
                      rpi_port: Optional[int] = None) -> None:
    """
    Override module-level target addresses at runtime.
    Call this before sending if you want to change targets programmatically.
    """
    global ESP32_IP, RPI_IP, ESP32_PORT, RPI_PORT
    if esp32_ip is not None:
        ESP32_IP = esp32_ip
    if rpi_ip is not None:
        RPI_IP = rpi_ip
    if esp32_port is not None:
        ESP32_PORT = int(esp32_port)
    if rpi_port is not None:
        RPI_PORT = int(rpi_port)


def _build_payload_text(emotion: str) -> bytes:
    """Legacy plain-text payload for ESP32 endpoints expecting raw body."""
    emoji = EMOJI_MAP.get(emotion, "??")
    msg = f"{emoji} {emotion}"
    return msg.encode("utf-8")


def _build_payload_json(emotion: str) -> Dict[str, Any]:
    """JSON payload suitable for RPi/phone endpoints."""
    return {
        "emotion": emotion,
        "emoji": EMOJI_MAP.get(emotion, "??"),
        "source": "rpi4b"  # consumer can change if needed
    }


def _post(url: str, data: Any, headers: Dict[str, str], timeout: float) -> requests.Response:
    """Wrapper to post data with a short timeout and raise on bad status."""
    resp = requests.post(url, data=data, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def send_emotion_wifi(emotion: str,
                      send_to_esp: bool = True,
                      send_to_rpi: bool = True,
                      timeout: float = 1.0) -> Dict[str, Optional[str]]:
    """
    Send detected emotion to configured devices.

    - send_to_esp: POSTs plain text to ESP32 at http://{ESP32_IP}:{ESP32_PORT}/emotion
                  (keeps legacy behavior for simple ESP32 HTTP servers)
    - send_to_rpi: POSTs JSON to Raspberry Pi at http://{RPI_IP}:{RPI_PORT}/emotion
                  (useful if your Pi hosts a small webserver to forward to an Android phone)
    - timeout: request timeout in seconds (kept small to avoid blocking UI threads)

    Returns a dict with 'esp' and 'rpi' entries containing response text or error message.
    """
    results = {"esp": None, "rpi": None}

    # Validate inputs quickly
    if not emotion:
        return {"esp": "no emotion provided", "rpi": "no emotion provided"}

    # Send to ESP32 (legacy plain-text endpoint)
    if send_to_esp and ESP32_IP:
        try:
            url = f"http://{ESP32_IP}:{ESP32_PORT}/emotion"
            payload = _build_payload_text(emotion)
            headers = {"Content-Type": "text/plain; charset=utf-8"}
            resp = _post(url, data=payload, headers=headers, timeout=timeout)
            results["esp"] = resp.text
        except Exception as e:
            results["esp"] = f"error: {e}"

    # Send to Raspberry Pi (JSON) - useful to forward to Android phone or web UI
    if send_to_rpi and RPI_IP:
        try:
            url = f"http://{RPI_IP}:{RPI_PORT}/emotion"
            payload_json = _build_payload_json(emotion)
            headers = {"Content-Type": "application/json; charset=utf-8"}
            resp = _post(url, data=json.dumps(payload_json), headers=headers, timeout=timeout)
            results["rpi"] = resp.text
        except Exception as e:
            results["rpi"] = f"error: {e}"

    return results


# Example usage (module can be executed directly for quick test)
if __name__ == "__main__":
    # You can override targets here, or set env vars before running.
    # Example to set RPI IP programmatically:
    # configure_targets(rpi_ip="192.168.0.200", rpi_port=5000)
    print("Sending test emotion 'Happy' to configured targets...")
    res = send_emotion_wifi("Happy")
    print("Results:", res)