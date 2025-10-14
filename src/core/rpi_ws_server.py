import asyncio
import websockets
import threading
from simple_emotion_detection import SimpleEmotionDetector

connected_clients = set()

async def emotion_broadcaster(detector):
    """Continuously detect emotion and broadcast to clients."""
    cap = detector.init_camera()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detector.detect_faces(frame)
        if faces:
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            emotion, confidence = detector.predict_emotion(face_img)
            # Broadcast emotion to all clients
            for ws in connected_clients.copy():
                try:
                    await ws.send(emotion)
                except Exception:
                    connected_clients.discard(ws)
        await asyncio.sleep(0.2)  # Adjust as needed

async def handler(websocket, path):
    connected_clients.add(websocket)
    try:
        async for _ in websocket:
            pass  # You can handle incoming messages here if needed
    finally:
        connected_clients.discard(websocket)

def start_server(detector, host='0.0.0.0', port=8080):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(handler, host, port)
    loop.run_until_complete(start_server)
    loop.create_task(emotion_broadcaster(detector))
    loop.run_forever()

# REMOVE or COMMENT OUT this block:
# if __name__ == "__main__":
#     detector = SimpleEmotionDetector()
#     threading.Thread(target=start_server, args=(detector,), daemon=True).start()
#     print("WebSocket emotion SERVER running on ws://0.0.0.0:8080")
#     while True:
#         pass  # Keep main thread alive
