from fastapi import FastAPI, WebSocket
from rag import ask_patient
import json

app = FastAPI()

# ✅ Health check
@app.get("/")
def home():
    return {"status": "AI Patient API running 🚀"}


# ✅ HTTP API (Browser / Postman test)
@app.get("/chat")
def chat(q: str):
    answer = ask_patient(q)
    return {"answer": answer}


# ✅ WebSocket (Unity use करेगा)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")

    try:
        while True:
            data = await websocket.receive_text()
            print("📥 Question:", data)

            answer = ask_patient(data)

            # 🔥 Always send JSON (important for Unity)
            response = {
                "answer": answer
            }

            await websocket.send_text(json.dumps(response))

    except Exception as e:
        print("❌ Disconnected:", e)