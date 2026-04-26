"""Agentic orchestrator with tool-calling support for Ollama and Gemini backends."""

import json
import base64
import io
import os

import numpy as np
import requests
from PIL import Image

from app_state import AppState
from fp_client import detect_object as fp_detect

# ── Configuration ──────────────────────────────────────────────────
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")  # "ollama" or "gemini"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "")  # auto-detect if empty

# Gemini (only used when LLM_BACKEND=gemini)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")


# ── Ollama model auto-detection ────────────────────────────────────
def _get_ollama_model() -> str:
    """Auto-detect the currently running model from Ollama."""
    if OLLAMA_MODEL:
        return OLLAMA_MODEL

    # First try running models
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        resp.raise_for_status()
        running = resp.json().get("models", [])
        if running:
            return running[0]["model"]
    except (requests.ConnectionError, requests.Timeout):
        pass

    # Fallback to first available model
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        if models:
            return models[0]["model"]
    except (requests.ConnectionError, requests.Timeout):
        pass

    raise RuntimeError(
        "No models found in Ollama. Load a model first: ollama run <model>"
    )


# ── System prompt & tools ──────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful visual assistant. A live camera feed is running in the background. "
    "You do NOT need the user to provide an image — the tools automatically capture "
    "the current camera frame when called. Just call the tool directly.\n\n"
    "You can have normal conversations with the user. "
    "Only use tools when the user asks you to find, detect, zoom, or inspect something. "
    "Do NOT use tools for greetings, general questions, or chitchat.\n\n"
    "Available tools:\n"
    "- detect_object: captures the current camera frame and finds an object in it\n"
    "- zoom_to_object: zooms the camera into the last detected object\n"
    "- inspect_object: visually examines the detected object to answer a question about it\n\n"
    "Rules:\n"
    "- Only call one tool per turn, then describe the result.\n"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "detect_object",
            "description": (
                "Detect an object in the current camera frame. Supports both "
                "generic queries ('cat', 'bottle') and specific descriptive ones "
                "('the red cup on the left', 'the person wearing glasses'). "
                "Pass the user's description directly as the query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to detect — can be generic or descriptive",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_to_object",
            "description": "Zoom the camera view into a detected object for a closer look",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Which detection to zoom to (0-based). Defaults to 0 (first detection).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_object",
            "description": (
                "Visually inspect the last detected object to answer questions "
                "about it — e.g. identify its type, read text on it, describe its "
                "color, brand, condition, or any detail the user is asking about. "
                "Only works after detect_object has found something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What to look for or answer about the object",
                    }
                },
                "required": ["question"],
            },
        },
    },
]


# ── Utility helpers ────────────────────────────────────────────────
def _log_tokens(data: dict, label: str = "") -> None:
    """Log token usage from an Ollama response."""
    prompt_tokens = data.get("prompt_eval_count", 0)
    gen_tokens = data.get("eval_count", 0)
    total_ms = data.get("total_duration", 0) / 1e6  # ns → ms
    print(f"  [tokens{' ' + label if label else ''}] "
          f"prompt={prompt_tokens} gen={gen_tokens} "
          f"total={prompt_tokens + gen_tokens} "
          f"time={total_ms:.0f}ms")


def _frame_to_b64(frame: np.ndarray) -> str:
    """Encode numpy frame (RGB) as base64 JPEG for Ollama vision."""
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _prune_history(messages: list[dict], max_images: int = 2) -> list[dict]:
    """Remove older images from chat history to stay within context limits."""
    image_count = 0
    for msg in reversed(messages):
        if "images" in msg:
            image_count += 1
            if image_count > max_images:
                del msg["images"]
    return messages


# ── Tool execution ─────────────────────────────────────────────────
def _execute_tool(name: str, args: dict, state: AppState, fp_url: str) -> str:
    """Execute a tool call and return the result string."""
    print(f"  [tool] executing: {name}({json.dumps(args)})")

    if name == "detect_object":
        query = args.get("query", "")
        if state.last_frame is None:
            return json.dumps({"error": "No camera frame available"})
        pil_image = Image.fromarray(state.last_frame)
        detections = fp_detect(pil_image, query, fp_url=fp_url)
        state.bboxes = detections
        print(f"  [tool] detect_object → {len(detections)} detection(s)")
        for i, d in enumerate(detections):
            print(f"         [{i}] label={d['label']} bbox={[round(c,1) for c in d['bbox']]}")
        return json.dumps(detections)

    elif name == "zoom_to_object":
        idx = int(args.get("index", 0))
        if state.bboxes:
            if idx < 0 or idx >= len(state.bboxes):
                idx = 0
            bbox = state.bboxes[idx]["bbox"]
            print(f"  [tool] zooming to detection [{idx}] label={state.bboxes[idx].get('label','')}")
            frame_h, frame_w = state.last_frame.shape[:2]
            frame_aspect = frame_w / frame_h

            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            obj_w, obj_h = x2 - x1, y2 - y1

            pad = 0.25
            region_w = obj_w * (1 + pad * 2)
            region_h = obj_h * (1 + pad * 2)

            if region_w / region_h > frame_aspect:
                region_h = region_w / frame_aspect
            else:
                region_w = region_h * frame_aspect

            zx1 = max(0, int(cx - region_w / 2))
            zy1 = max(0, int(cy - region_h / 2))
            zx2 = min(frame_w, int(cx + region_w / 2))
            zy2 = min(frame_h, int(cy + region_h / 2))

            state.zoom_region = (zx1, zy1, zx2, zy2)
            print(f"  [tool] zoom_to_object → region={[zx1,zy1,zx2,zy2]}")
            return json.dumps({"status": "zoomed", "bbox": bbox, "zoom_region": [zx1, zy1, zx2, zy2]})
        print("  [tool] zoom_to_object → no bboxes to zoom to")
        return json.dumps({"error": "No detected objects to zoom to"})

    elif name == "inspect_object":
        question = args.get("question", "Describe this object in detail.")
        if not state.bboxes or state.last_frame is None:
            return json.dumps({"error": "No detected object to inspect"})

        # Use zoom region if active, otherwise pad the bbox
        if state.zoom_region:
            crop_box = list(state.zoom_region)
        else:
            bbox = state.bboxes[0]["bbox"]
            frame_h, frame_w = state.last_frame.shape[:2]
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 0.25
            crop_box = [
                max(0, bbox[0] - bw * pad),
                max(0, bbox[1] - bh * pad),
                min(frame_w, bbox[2] + bw * pad),
                min(frame_h, bbox[3] + bh * pad),
            ]

        description = _inspect_crop(state.last_frame, crop_box, question)
        return json.dumps({"description": description})

    return json.dumps({"error": f"Unknown tool: {name}"})


def _inspect_crop(frame: np.ndarray, bbox: list, question: str) -> str:
    """One-shot VLM call on the cropped bbox region."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return "Could not crop the detected area."
    crop = frame[y1:y2, x1:x2]
    img_b64 = _frame_to_b64(crop)
    print(f"  [inspect] crop {x2-x1}x{y2-y1}, question: {question}")

    if LLM_BACKEND == "gemini":
        return _inspect_gemini(crop, question)

    # Ollama VLM path
    model = _get_ollama_model()
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": question, "images": [img_b64]}
            ],
            "stream": False,
            "options": {"num_predict": 512, "temperature": 0},
        },
        timeout=60,
    )
    resp.raise_for_status()
    inspect_data = resp.json()
    _log_tokens(inspect_data, "inspect")
    answer = inspect_data.get("message", {}).get("content", "")
    print(f"  [inspect] answer: {answer[:120]}")
    return answer


def _inspect_gemini(crop: np.ndarray, question: str) -> str:
    """Inspect using Gemini vision."""
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    img = Image.fromarray(crop)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}},
            question,
        ],
    )
    answer = response.text.strip()
    print(f"  [inspect-gemini] answer: {answer[:120]}")
    return answer


# ── Ollama orchestrator ────────────────────────────────────────────
def _run_ollama(user_text: str, state: AppState, fp_url: str) -> str:
    """Run Ollama tool-calling orchestrator."""
    user_msg = {"role": "user", "content": user_text}

    if not state.chat_history:
        state.chat_history.append({"role": "system", "content": SYSTEM_PROMPT})

    state.chat_history.append(user_msg)

    model = _get_ollama_model()
    print(f"[model] {model}")

    # Round 1: call with tools — let LLM decide if it needs one
    print("[round 1] tool selection")
    _prune_history(state.chat_history)

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": state.chat_history,
            "tools": TOOLS,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message", {})
    _log_tokens(data, "round1")

    tool_calls = msg.get("tool_calls")
    content = msg.get("content", "")

    # No tool needed — just a text reply
    if not tool_calls:
        state.chat_history.append({"role": "assistant", "content": content})
        print(f"[assistant] {content}")
        return content

    # Execute exactly one tool
    tool_call = tool_calls[0]
    func = tool_call.get("function", {})
    name = func.get("name", "")
    args = func.get("arguments", {})
    print(f"  [llm] tool call: {name}")

    state.chat_history.append(msg)
    result = _execute_tool(name, args, state, fp_url)
    state.chat_history.append({"role": "tool", "content": result})

    # Round 2: call WITHOUT tools — force text-only response
    print("[round 2] generating response (no tools)")
    _prune_history(state.chat_history)

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": state.chat_history,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=60,
    )
    resp.raise_for_status()
    r2_data = resp.json()
    _log_tokens(r2_data, "round2")
    content = r2_data.get("message", {}).get("content", "")
    state.chat_history.append({"role": "assistant", "content": content})
    print(f"[assistant] {content}")
    return content


# ── Gemini text extraction helper ──────────────────────────────────
def _extract_gemini_text(response) -> str:
    """Safely extract text from a Gemini response, skipping function_call parts."""
    try:
        return response.text.strip()
    except Exception:
        # Fallback: manually extract text from parts
        texts = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)
        result = " ".join(texts).strip()
        if result:
            return result
        return "I found the object and marked it on screen."


# ── Gemini orchestrator ────────────────────────────────────────────
_gemini_client = None  # Persistent client (must outlive the chat session)
_gemini_chat = None  # Persistent chat session for multi-turn context


def _run_gemini(
    user_text: str,
    state: AppState,
    fp_url: str,
    audio_bytes: bytes | None = None,
    audio_mime: str = "audio/webm",
) -> str:
    """Run Gemini function-calling orchestrator with optional audio."""
    global _gemini_client, _gemini_chat
    from google import genai
    from google.genai import types

    # Create or reuse persistent client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    # Build tool declarations for the new SDK
    tool_declarations = []
    for t in TOOLS:
        func = t["function"]
        props = {}
        for pname, pinfo in func["parameters"].get("properties", {}).items():
            props[pname] = types.Schema(
                type=pinfo["type"].upper(),
                description=pinfo.get("description", ""),
            )
        tool_declarations.append(types.FunctionDeclaration(
            name=func["name"],
            description=func["description"],
            parameters=types.Schema(
                type="OBJECT",
                properties=props,
                required=func["parameters"].get("required", []),
            ),
        ))

    gemini_tools = [types.Tool(function_declarations=tool_declarations)]

    chat_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=gemini_tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Create or reuse persistent chat session for multi-turn context
    if _gemini_chat is None:
        _gemini_chat = _gemini_client.chats.create(
            model=GEMINI_MODEL,
            config=chat_config,
        )

    # Build user content parts
    parts = []
    if audio_bytes:
        parts.append(types.Part(inline_data=types.Blob(
            mime_type=audio_mime,
            data=audio_bytes,
        )))
    parts.append(types.Part(text=user_text))

    # Round 1: send message with tools
    print("[gemini round 1] sending with tools")
    response = _gemini_chat.send_message(parts)

    # Check for function call
    candidate = response.candidates[0]
    part = candidate.content.parts[0]

    if hasattr(part, "function_call") and part.function_call and part.function_call.name:
        fc = part.function_call
        name = fc.name
        args = dict(fc.args) if fc.args else {}
        print(f"  [gemini] tool call: {name}({json.dumps(args)})")

        result_str = _execute_tool(name, args, state, fp_url)

        # Wrap result in a dict — Gemini requires function_response to be a dict, not a list
        result_obj = json.loads(result_str)
        if isinstance(result_obj, list):
            result_obj = {"detections": result_obj}

        # Round 2: send function response back through the chat session
        print("[gemini round 2] generating response")
        function_response_part = types.Part.from_function_response(
            name=name,
            response=result_obj,
        )
        response2 = _gemini_chat.send_message(function_response_part)
        content = _extract_gemini_text(response2)
    else:
        content = _extract_gemini_text(response)

    print(f"[assistant-gemini] {content}")
    return content


# ── Reset ──────────────────────────────────────────────────────────
def reset_orchestrator():
    """Reset the orchestrator state (clears Gemini chat session)."""
    global _gemini_client, _gemini_chat
    _gemini_client = None
    _gemini_chat = None


# ── Public entry point ─────────────────────────────────────────────
def run_orchestrator(
    user_text: str,
    state: AppState,
    fp_url: str = "http://localhost:7860",
    audio_bytes: bytes | None = None,
    audio_mime: str = "audio/webm",
) -> str:
    """
    Run the orchestrator. Dispatches to Ollama or Gemini based on LLM_BACKEND.
    Returns the final assistant text response.
    """
    print(f"\n{'='*60}")
    print(f"[user] {user_text}")
    print(f"[backend] {LLM_BACKEND}")

    if LLM_BACKEND == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        result = _run_gemini(user_text, state, fp_url, audio_bytes, audio_mime)
    else:
        result = _run_ollama(user_text, state, fp_url)

    print(f"{'='*60}\n")
    return result
