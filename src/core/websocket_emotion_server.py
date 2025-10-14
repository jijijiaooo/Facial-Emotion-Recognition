import asyncio
import websockets
import json

clients = set()
latest_emotion = {"emotion": "Neutral", "emoji": "-_-", "confidence": 0}

async def handler(websocket, path):
    clients.add(websocket)
    try:
        await websocket.send(json.dumps(latest_emotion))
        async for _ in websocket:
            pass  # No incoming messages expected from clients
    finally:
        clients.remove(websocket)

def broadcast_emotion(emotion, emoji, confidence):
    global latest_emotion
    latest_emotion = {"emotion": emotion, "emoji": emoji, "confidence": confidence}
    asyncio.run(_broadcast())

async def _broadcast():
    if clients:
        message = json.dumps(latest_emotion)
        await asyncio.gather(*(client.send(message) for client in clients if client.open))

def start_server(host='0.0.0.0', port=8080):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = websockets.serve(handler, host, port)
    loop.run_until_complete(server)
    print(f"WebSocket server running on ws://{host}:{port}")
    loop.run_forever()

