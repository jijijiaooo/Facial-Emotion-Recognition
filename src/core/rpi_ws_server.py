import asyncio
import websockets
import json

connected = set()
latest_emotion = {"emotion": "Neutral", "emoji": "-_-", "confidence": 0}

async def emotion_handler(websocket, path):
    connected.add(websocket)
    try:
        # Send the latest emotion on connect
        await websocket.send(json.dumps(latest_emotion))
        async for message in websocket:
            try:
                data = json.loads(message)
                if "emotion" in data:
                    global latest_emotion
                    # Accept and store all fields (emotion, emoji, confidence, etc.)
                    latest_emotion = data
                    print("Received emotion:", data)
                    # Broadcast to all clients
                    for ws in connected:
                        if ws.open:
                            await ws.send(json.dumps(latest_emotion))
            except Exception as e:
                print("Error:", e)
    finally:
        connected.remove(websocket)

if __name__ == "__main__":
    import sys
    port = 8080 if len(sys.argv) < 2 else int(sys.argv[1])
    print(f"WebSocket server running on ws://0.0.0.0:{port}")
    asyncio.get_event_loop().run_until_complete(
        websockets.serve(emotion_handler, "0.0.0.0", port)
    )
    asyncio.get_event_loop().run_forever()