import os
import json
import base64
import asyncio
import io
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from rag import ask_patient, client as openai_client

app = FastAPI()


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "AI Patient API running 🚀"}


# ─── HTTP endpoint (unchanged) ────────────────────────────────────────────────
@app.get("/chat")
def chat(q: str):
    answer = ask_patient(q)
    return {"answer": answer}


# ─── Helper: PCM16 bytes → WAV bytes ─────────────────────────────────────────
# OpenAI Whisper API WAV file expect karta hai, raw PCM nahi
def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    import struct
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    buf = io.BytesIO()
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, num_channels, sample_rate,
                          byte_rate, block_align, bits_per_sample))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(pcm_bytes)
    return buf.getvalue()


# ─── STT: audio → text (OpenAI Whisper API ~0.5 sec) ─────────────────────────
async def speech_to_text(pcm_bytes: bytes) -> str:
    wav_bytes = pcm16_to_wav(pcm_bytes)
    wav_file = io.BytesIO(wav_bytes)
    wav_file.name = "audio.wav"

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_file,
            language="en"
        )
    )
    return result.text.strip()


# ─── TTS: text → PCM16 bytes (OpenAI TTS ~1 sec) ─────────────────────────────
async def text_to_speech(text: str) -> bytes:
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",           # alloy / echo / fable / onyx / nova / shimmer
            input=text,
            response_format="pcm"  # raw PCM16 — Unity AudioPlayer seedha use karta hai
        )
    )
    return result.content


# ─── PCM16 → base64 chunks ────────────────────────────────────────────────────
def pcm16_chunks(pcm_bytes: bytes, chunk_size: int = 4096):
    for i in range(0, len(pcm_bytes), chunk_size):
        yield base64.b64encode(pcm_bytes[i:i + chunk_size]).decode("utf-8")


# ─── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Unity client connected")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            # ── Audio input (Unity mic se) ─────────────────────────────────────
            if msg_type == "user.audio":
                try:
                    # 1. Base64 → PCM16
                    raw_bytes = base64.b64decode(data["audio"])

                    # 2. STT
                    print("🎤 STT...")
                    user_text = await speech_to_text(raw_bytes)
                    print(f"📥 Doctor: {user_text}")

                    if not user_text:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Could not detect speech. Please try again."
                        })
                        continue

                    # Doctor ka transcript Unity ko bhejo
                    await websocket.send_json({
                        "type": "user.transcript",
                        "text": user_text
                    })

                    # 3. RAG + LLM
                    print("🧠 RAG + LLM...")
                    answer = ask_patient(user_text)
                    print(f"💬 Reem: {answer}")

                    # Patient transcript Unity ko bhejo
                    await websocket.send_json({
                        "type": "transcript.delta",
                        "delta": answer
                    })

                    # 4. TTS
                    print("🔊 TTS...")
                    audio_pcm16 = await text_to_speech(answer)

                    # 5. Audio chunks Unity ko bhejo
                    for chunk_b64 in pcm16_chunks(audio_pcm16):
                        await websocket.send_json({
                            "type": "audio.delta",
                            "delta": chunk_b64
                        })

                    await websocket.send_json({"type": "response.done"})
                    print("✅ Done.\n")

                except Exception as e:
                    print(f"❌ Pipeline error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

            # ── Text input (Postman/testing ke liye) ──────────────────────────
            elif msg_type == "user.text":
                try:
                    user_text = data.get("text", "").strip()
                    if not user_text:
                        continue

                    answer = ask_patient(user_text)

                    await websocket.send_json({
                        "type": "transcript.delta",
                        "delta": answer
                    })

                    audio_pcm16 = await text_to_speech(answer)

                    for chunk_b64 in pcm16_chunks(audio_pcm16):
                        await websocket.send_json({
                            "type": "audio.delta",
                            "delta": chunk_b64
                        })

                    await websocket.send_json({"type": "response.done"})

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

    except WebSocketDisconnect:
        print("🔌 Unity client disconnected.")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
