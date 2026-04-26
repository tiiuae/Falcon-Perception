# Agentic Perception

A real-time agentic perception system built on top of [Falcon Perception](https://github.com/tiiuae/Falcon-Perception). An LLM orchestrator decides when to detect objects, zoom into them, and visually inspect them — all through natural conversation over a live camera feed.

## Features

- **Tool-calling agent**: The LLM decides which tools to invoke (`detect_object`, `zoom_to_object`, `inspect_object`) based on conversation context
- **On-demand detection**: Frames are only sent to Falcon Perception when the agent decides to look — no continuous polling
- **Visual zoom**: CSS-based camera zoom into detected objects, preserving aspect ratio
- **Visual inspection**: Crops the detected region and sends it to the VLM for detailed Q&A (read text, identify brand, describe details)
- **Multi-turn chat**: Full conversation history so the agent remembers previous detections
- **Dual LLM backend**: Switch between local Ollama and cloud Gemini with a single env var
- **Audio input**: Voice commands via microphone (Gemini backend only)

## Architecture

```
User (text/voice) → FastAPI Server → Orchestrator (Ollama or Gemini)
                                          ↓ tool call
                                    Falcon Perception (detect)
                                    Ollama VLM / Gemini (inspect)
                                          ↓
                              bboxes + zoom + reply → Browser UI
```

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (for Falcon Perception engine)
- **Ollama** (for local LLM) or **Gemini API key** (for cloud LLM)

## Setup

### 1. Set up Falcon Perception base environment

From the **repository root** (`Falcon-Perception/`):

```bash
# Check your CUDA version
nvidia-smi
# **Note**: If your CUDA version differs from the default in `pyproject.toml`, update the PyTorch index URL in `[tool.uv.index]` before running `uv pip install`.

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install uv (fast package installer)
pip install uv

# Install Falcon Perception as a library (with server dependencies)
uv pip install -e ".[server]"
```



### 2. Install Agentic Perception dependencies

Still from the repository root, with the venv activated:

```bash
pip install -r agentic_perception/requirements.txt
```

### 3. Choose your LLM backend

#### Option A: Ollama (local, free, private)

```bash
# Install Ollama: https://ollama.com
# Pull a vision-capable model
# I found out that the 4b version works best as smallest possible candidate
ollama pull qwen3.5:4b

# Start the Ollama server (runs on port 11434)
ollama serve
```

#### Option B: Gemini (cloud, fast, multimodal audio)

```bash
# Set your API key
export GEMINI_API_KEY="your-api-key-here"
```

## Running

### Terminal 1 — Start Falcon Perception engine

```bash
python -m falcon_perception.server
```

This starts the detection engine on `http://localhost:7860`.

### Terminal 2 — Start Agentic Perception UI

```bash
# With Ollama (default):
python agentic_perception/server.py

# With Gemini:
LLM_BACKEND=gemini  python agentic_perception/server.py 
or
LLM_BACKEND=ollama  python agentic_perception/server.py
```

Open **http://localhost:7880** in your browser.

## Usage

The UI shows your camera feed with a chat panel. Just type naturally:

| What you say | What happens |
|---|---|
| "hello" | Normal chat — no tools called |
| "find the coffee cup" | Agent calls `detect_object` → bbox drawn on screen |
| "zoom in" | Agent calls `zoom_to_object` → camera zooms into the detection |
| "what brand is it?" | Agent calls `inspect_object` → crops the object, VLM answers |
| "find the book on the left" | Specific spatial queries work too |
| Click **Reset** | Clears all detections, zoom, and chat history |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` or `gemini` |
| `GEMINI_API_KEY` | — | Required when `LLM_BACKEND=gemini` |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite-preview` | Which Gemini model to use |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | (auto-detect) | Override Ollama model name |
| `FP_URL` | `http://localhost:7860` | Falcon Perception server URL |

## File Structure

```
agentic_perception/
├── server.py           # FastAPI routes (chat, audio, reset, health)
├── orchestrator.py     # Agentic LLM orchestrator with tool-calling
├── fp_client.py        # Falcon Perception HTTP client
├── app_state.py        # Shared state (bboxes, zoom, chat history)
├── requirements.txt    # Extra pip dependencies
├── README.md           # This file
└── static/
    └── index.html      # Single-file frontend (HTML + CSS + JS)
```
