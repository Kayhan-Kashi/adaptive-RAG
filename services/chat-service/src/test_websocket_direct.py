# test_websocket_direct.py
import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8000/ws/12345678-1234-5678-1234-567812345678"
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ WebSocket connected!")
            
            # Send ping
            await ws.send(json.dumps({"type": "ping"}))
            response = await ws.recv()
            print(f"Ping response: {response}")
            
            # Send chat message
            await ws.send(json.dumps({
                "type": "chat",
                "conversation_id": "test-conv-123",
                "prompt": "Hello World"
            }))
            
            response = await ws.recv()
            print(f"Chat response: {response}")
            
            await ws.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")

asyncio.run(test())