"""FastAPI backend for Falcon Perception real-time agentic demo."""

import sys
import os
import base64
import io
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests

from app_state import AppState
from fp_client import check_health as fp_health
from orchestrator import run_orchestrator, reset_orchestrator, LLM_BACKEND, OLLAMA_URL, GEMINI_API_KEY

app = FastAPI(title="Agentic Perception")
state = AppState()

FP_URL = os.environ.get("FP_URL", "http://localhost:7860")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class ChatRequest(BaseModel):
    text: str
    frame: str  # base64-encoded JPEG


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Main chat endpoint. Decodes frame, runs orchestrator, returns reply + visual state."""
    img_bytes = base64.b64decode(req.frame)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    state.last_frame = np.array(img)

    try:
        reply = run_orchestrator(req.text, state, fp_url=FP_URL)
    except Exception as e:
        reply = f"Error: {e}"

    return {
        "reply": reply,
        "bboxes": state.bboxes,
        "zoom_region": list(state.zoom_region) if state.zoom_region else None,
    }


@app.post("/api/chat-audio")
async def chat_audio(audio: UploadFile = File(...)):
    """Audio chat endpoint (Gemini only). Transcribes and processes audio."""
    if LLM_BACKEND != "gemini":
        return {"reply": "Audio input is only supported with the Gemini backend.", "bboxes": [], "zoom_region": None}

    audio_bytes = await audio.read()
    mime_type = audio.content_type or "audio/webm"

    # We need a frame too — use the last captured frame
    if state.last_frame is None:
        return {"reply": "No camera frame available. Please wait for camera to start.", "bboxes": [], "zoom_region": None}

    try:
        reply = run_orchestrator(
            "The user sent an audio message. Listen to it and respond appropriately.",
            state,
            fp_url=FP_URL,
            audio_bytes=audio_bytes,
            audio_mime=mime_type,
        )
    except Exception as e:
        reply = f"Error: {e}"

    return {
        "reply": reply,
        "bboxes": state.bboxes,
        "zoom_region": list(state.zoom_region) if state.zoom_region else None,
    }


@app.post("/api/reset")
def reset():
    """Clear all state."""
    state.clear()
    reset_orchestrator()
    return {"status": "ok"}


@app.get("/api/health")
def health():
    """Check connectivity to FP server and LLM backend."""
    llm_ok = False
    llm_name = LLM_BACKEND

    if LLM_BACKEND == "ollama":
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            llm_ok = r.status_code == 200 and len(r.json().get("models", [])) > 0
        except Exception:
            pass
    elif LLM_BACKEND == "gemini":
        llm_ok = bool(GEMINI_API_KEY)

    return {
        "fp": fp_health(FP_URL),
        "llm": llm_ok,
        "llm_backend": llm_name,
    }


if __name__ == "__main__":
    import uvicorn

    print("Falcon Perception — Agentic Real-Time Demo")
    print(f"  FP server:    {FP_URL}")
    print(f"  LLM backend:  {LLM_BACKEND}")
    if not fp_health(FP_URL):
        print("  WARNING: FP server not reachable")
    print()
    uvicorn.run(app, host="0.0.0.0", port=7880)
