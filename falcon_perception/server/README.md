# Falcon Perception — Inference Server

Multi-GPU inference server with continuous batching, built on FastAPI and the `PagedInferenceEngine`.

## Code Layout

```
falcon_perception/server/
├── README.md            # This file
├── __init__.py
├── __main__.py          # CLI entry point (`python -m falcon_perception.server`)
├── config.py            # ServerConfig dataclass (model, engine, server params)
├── schemas.py           # Pydantic request/response models
├── engine_worker.py     # WorkerProxy (1 per GPU) + WorkerPool (least-load dispatch)
└── app.py               # FastAPI app factory, all HTTP endpoints

demo/streamlit_app.py    # Streamlit demo (API client, no GPU needed)
```

### Architecture

```
                     HTTP requests
                          │
                  ┌───────▼────────┐
                  │    FastAPI     │  uvicorn on :7860
                  │ /v1/predictions│
                  └───┬────┬───┬───┘
     least-loaded     │    │   │
                 ┌────▼┐ ┌─▼┐ ┌▼────┐
                 │ Q0  │ │Q1│ │ Q2  │  mp.Queue (cross-process)
                 └──┬──┘ └─┬┘ └──┬──┘
              ┌─────▼──┐┌──▼───┐┌▼─────┐
              │Worker 0││Wkr 1 ││Wkr 2 │  continuous batching loops
              │ GPU:0  ││ GPU:1││GPU:2 │  (run_one_step() in a while-true)
              └────────┘└──────┘└──────┘
```

Each `WorkerProxy` launches a **separate process** (`multiprocessing.Process`) with an
isolated CUDA context, running a `PagedInferenceEngine` (or `OCRInferenceEngine` for OCR
models) on one GPU. New requests are sent via `mp.Queue` and injected into the engine's
`waiting` deque between steps. Completion is signalled back through a shared response
queue, where a collector thread resolves the corresponding `asyncio.Future` in the
FastAPI event loop.

## Quick Start

```bash
# Install with server dependencies (once)
uv sync --extra server --extra demo  # or
pip install -e ".[server,demo]"

# Launch with defaults (auto-detect GPUs, port 7860)
python -m falcon_perception.server

# Or with explicit config
python -m falcon_perception.server \
    --config.num-gpus 2 \
    --config.hf-local-dir ./my_export/ \
    --config.no-cudagraph \
    --config.port 8000

# See all options
python -m falcon_perception.server --help
```

On startup the server will:

1. Download / load model weights (from HF Hub or local dir)
2. `torch.compile` the model (if `--config.compile`, default: on)
3. Capture CUDA graphs for decode (if `--config.cudagraph`, default: on)
4. Begin accepting requests on the configured port

Steps 1–3 take 1–3 minutes depending on GPU and whether compile caches exist.

### Configuration Reference

All fields live in `ServerConfig` (see `config.py`). Pass them as `--config.<field>`.

| Flag | Default | Description |
|------|---------|-------------|
| `hf-model-id` | `tiiuae/Falcon-Perception` | HF Hub model ID |
| `hf-revision` | `main` | HF Hub revision / branch |
| `hf-local-dir` | — | Load from a local export instead of HF Hub |
| `dtype` | `float32` | Model dtype (`float32` or `bfloat16`) |
| `num-gpus` | `-1` (auto) | Number of GPUs; `-1` = use all available |
| `compile` / `no-compile` | on | Enable `torch.compile` |
| `cudagraph` / `no-cudagraph` | on | Capture CUDA graphs for decode |
| `max-batch-size` | `128` | Max sequences in a single engine step |
| `max-seq-length` | `8192` | Max sequence length (tokens) |
| `n-pages` | `1024` | Number of KV-cache pages |
| `page-size` | `128` | Tokens per KV-cache page |
| `prefill-length-limit` | `16384` | Max prefill length |
| `temperature` | `0.0` | Sampling temperature |
| `top-k` | — | Top-k sampling (disabled by default) |
| `min-image-size` | `256` | Default min image dimension (px) |
| `max-image-size` | `1024` | Default max image dimension (px) |
| `max-tokens` | `8192` | Default max output tokens |
| `layout-threshold` | `0.3` | Layout detection confidence threshold (`ocr_layout`) |
| `host` | `0.0.0.0` | Bind address |
| `port` | `7860` | Bind port |
| `startup-timeout` | `600` | Max seconds to wait for engine init |
| `images-dir` | `./public/images` | Server-side image cache directory |

## API Endpoints

| Method | Path                      | Description                                        |
|--------|---------------------------|----------------------------------------------------|
| POST   | `/v1/predictions`         | JSON body (image as URL or base64)                 |
| POST   | `/v1/predictions/upload`  | Multipart form (file upload)                       |
| GET    | `/v1/health`              | Readiness probe, model info + GPU VRAM stats       |
| GET    | `/v1/status`              | Per-GPU queue depths                               |
| GET    | `/v1/models`              | OpenAI-compatible model listing                    |
| GET    | `/upload/check`           | Check if an image is in the server cache           |
| POST   | `/upload`                 | Upload an image to the server cache                |
| POST   | `/segment`                | Segment from a previously cached image             |
| POST   | `/detect`                 | Detect from a previously cached image              |
| POST   | `/ocr_plain`              | Plain OCR from a previously cached image           |
| POST   | `/ocr_layout`             | Layout-aware OCR from a previously cached image    |
| GET    | `/docs`                   | Interactive Swagger UI (auto-generated)            |

**Supported tasks** (via the `task` field): `segmentation`, `detection`, `ocr_plain`, `ocr_layout`.

## Sending Requests

### JSON body (base64 image)

```bash
# Encode image to base64
IMAGE_B64=$(base64 -w0 photo.jpg)

curl -X POST http://localhost:7860/v1/predictions \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": {\"base64\": \"$IMAGE_B64\"},
    \"query\": \"dumplings\",
    \"task\": \"segmentation\",
    \"max_tokens\": 8192,
    \"min_image_size\": 256,
    \"max_image_size\": 1024
  }"
```

### JSON body (image URL)

```bash
curl -X POST http://localhost:7860/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "image": {"url": "https://example.com/photo.jpg"},
    "query": "dog",
    "task": "segmentation"
  }'
```

### Multipart file upload

```bash
curl -X POST http://localhost:7860/v1/predictions/upload \
  -F "image=@photo.jpg" \
  -F "query=dog" \
  -F "task=segmentation"
```

### Python client

```python
import base64
import requests

# From URL
resp = requests.post("http://localhost:7860/v1/predictions", json={
    "image": {"url": "https://example.com/photo.jpg"},
    "query": "dog",
    "task": "segmentation",
})
result = resp.json()

# From local file
with open("photo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:7860/v1/predictions", json={
    "image": {"base64": image_b64},
    "query": "dog",
})
result = resp.json()
```

## Response Format

All prediction endpoints return a flat `Response`:

```json
{
  "id": "pred_a1b2c3d4e5f6",
  "model": "falcon-perception",
  "created": 1739451600,
  "masks": [
    {
      "label": "object 1",
      "bbox": [120.5, 80.3, 340.2, 260.7],
      "rle": {"counts": "...", "size": [768, 1024]},
      "height": 768,
      "width": 1024
    }
  ],
  "text": "<decoded model output>",
  "query": "dumplings",
  "image_width": 1024,
  "image_height": 768,
  "input_tokens": 1234,
  "output_tokens": 200,
  "inference_time_ms": 1250.5,
  "queue_ms": 2.1,
  "tokenize_time_ms": 5.2,
  "prefill_time_ms": 120.0,
  "decode_time_ms": 1100.0,
  "finalize_time_ms": 23.3,
  "num_decode_steps": 180,
  "avg_decode_batch_size": 1.0,
  "prefill_batch_size": 1,
  "prefill_tokens": 1234,
  "num_preemptions": 0,
  "layout_regions": []
}
```

### Response fields

| Field                    | Description                                                          |
|--------------------------|----------------------------------------------------------------------|
| `masks[].label`          | Human-readable label (e.g. `"object 1"`)                             |
| `masks[].bbox`           | Bounding box as `[x1, y1, x2, y2]` in pixels                         |
| `masks[].rle`            | COCO RLE encoding (`{counts, size}` dict)                            |
| `masks[].height/width`   | Mask dimensions in pixels                                            |
| `text`                   | Raw decoded token text from the model                                |
| `image_width/height`     | Original image dimensions the masks are relative to                  |
| `input_tokens`           | Number of tokens in the prefill (image + text prompt)                |
| `output_tokens`          | Number of tokens generated                                           |
| `inference_time_ms`      | Wall time from enqueue to completion (includes queue wait)           |
| `queue_ms`               | Time spent waiting before the engine picked up the request           |
| `tokenize_time_ms`       | Time to tokenize the input                                           |
| `prefill_time_ms`        | Time for the prefill (prompt processing) phase                       |
| `decode_time_ms`         | Time for the decode (generation) phase                               |
| `finalize_time_ms`       | Time for post-processing (mask decoding, etc.)                       |
| `num_decode_steps`       | Number of autoregressive decode steps                                |
| `layout_regions`         | For `ocr_layout`: list of `{category, bbox, score, text}` dicts      |

### Decoding RLE masks

Masks use **COCO RLE** format. Decode with `pycocotools`:

```python
import numpy as np
import pycocotools.mask as mask_util

def decode_coco_rle(rle: dict) -> np.ndarray:
    """Decode a COCO RLE dict to a binary mask."""
    if isinstance(rle["counts"], list):
        rle = mask_util.frPyObjects(rle, rle["size"][0], rle["size"][1])
    return mask_util.decode(rle).astype(np.uint8)

# Usage with the API response:
mask_entry = result["masks"][0]
mask = decode_coco_rle(mask_entry["rle"])  # shape: (height, width)
```

## Mask Quality: Bilinear Upsampling Before Binarization

The model's raw mask logits are produced at a resolution determined by the
`hr_upsample_ratio` (typically 16×), which equals the processing image size
but may be smaller than the original input image. Before binarizing the logit
masks (sigmoid > 0.5) and encoding them as COCO RLE, the server **bilinearly
upsamples** the logit tensor to the original image dimensions. This preserves
smooth mask boundaries that would otherwise be lost by nearest-neighbor resize
of a binary mask.

This happens transparently in `finalize_masks` — no client-side configuration
is needed.

## Health Check

```bash
curl http://localhost:7860/v1/health
```

```json
{
  "status": "ready",
  "num_gpus": 2,
  "model_id": "tiiuae/Falcon-Perception",
  "supported_tasks": ["segmentation", "detection"],
  "gpus": [
    {
      "gpu_id": 0,
      "device_name": "NVIDIA A100-SXM4-80GB",
      "waiting": 0,
      "running": 3,
      "vram_allocated_gib": 12.4,
      "vram_reserved_gib": 18.2
    },
    ...
  ]
}
```

Use `status` for readiness probes (Kubernetes, HF Spaces, etc.):
- `"ready"` — all engines loaded, accepting requests
- `"loading"` — engines still initializing

`model_id` echoes the loaded HF model ID; `supported_tasks` lists the tasks
the model can perform (`["segmentation", "detection"]` for the full perception model,
`["detection"]` for the perception-300m model, or `["ocr_plain", "ocr_layout"]`
for OCR models). Clients can use this to dynamically configure their UI.

## Error Handling

Prediction endpoints return standard HTTP error codes with a JSON body:

| Code | When |
|------|------|
| 400  | Missing or invalid image (`image.url` and `image.base64` both empty, bad file, etc.) |
| 503  | No healthy GPU workers available (all engines failed or still loading) |
| 500  | Unexpected inference error |

```json
{
  "detail": "No healthy engines available."
}
```

## CORS

The server enables unrestricted CORS (`allow_origins=["*"]`), so browser-based
frontends (like the Streamlit demo) can call the API directly without a proxy.

## Model Variants

The server auto-detects the loaded model variant from `config.json`:

| Variant | `perception_heads` | `do_segmentation` | Supported tasks |
|---------|--------------------|-------------------|-----------------------------|
| **perception** (full) | True | True | `segmentation`, `detection` |
| **perception-300m** | True | False | `detection` only |
| **ocr** | False | N/A | `ocr_plain`, `ocr_layout` |

The correct engine (`PagedInferenceEngine` or `OCRInferenceEngine`) is selected
automatically. Segmentation requests to a detection-only model return HTTP 400.

Two OCR task modes are available:

- **`ocr_plain`** — plain text extraction. Returns recognized text in the `text` field.
- **`ocr_layout`** — layout-aware OCR. First runs layout detection to identify text
  regions (headings, paragraphs, tables, etc.), then runs OCR on each crop. Results are
  returned in the `layout_regions` array, each entry containing `{category, bbox, score,
  text}`.


## Streamlit Demo (API Client)

`demo/streamlit_app.py` is a pure API client — no GPU, no model loading.
It calls the server and renders the results. The app auto-detects which model
is loaded and shows only the relevant tasks (segmentation/detection or OCR).

```bash
# Terminal 1: server
python -m falcon_perception.server

# Terminal 2: streamlit
streamlit run demo/streamlit_app.py
```

Features:

- Supports all four tasks: **segmentation**, **detection**, **ocr_plain**, **ocr_layout**
- Upload an image or paste a URL; set min/max image sizes in the sidebar
- Server health indicator (polls `/v1/health`)
- Mask overlay rendering with optional **NMS** (greedy IoU-based suppression)
- Per-prediction **pipeline breakdown**: tokenize, prefill, decode, finalize timings,
  decode batch size, preemption count
- OCR layout results rendered with category, confidence score, and text (tables shown as HTML)

Configure the server URL in the sidebar (defaults to `http://localhost:7860`).
