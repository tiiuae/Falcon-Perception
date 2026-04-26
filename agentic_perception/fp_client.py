"""Client for Falcon Perception inference server."""

import base64
import io

import requests
from PIL import Image

DEFAULT_FP_URL = "http://localhost:7860"


def detect_object(
    image: Image.Image, query: str, fp_url: str = DEFAULT_FP_URL
) -> list[dict]:
    """Send image + query to FP server, return list of detections.

    Each detection: {"label": str, "bbox": [x1, y1, x2, y2]}
    Bboxes are returned in the *original* image pixel coordinates.
    """
    orig_w, orig_h = image.size

    # Cap image size client-side (matching streamlit_app.py pattern)
    max_image_size = 512
    sent_w, sent_h = orig_w, orig_h
    if max(orig_w, orig_h) > max_image_size:
        scale = max_image_size / max(orig_w, orig_h)
        sent_w, sent_h = int(orig_w * scale), int(orig_h * scale)
        image = image.resize((sent_w, sent_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "image": {"base64": img_b64},
        "query": query,
        "task": "detection",
        "max_image_size": max_image_size,
    }

    resp = requests.post(
        f"{fp_url}/v1/predictions", json=payload, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()

    # Scale factor from FP response image back to original
    resp_w = data.get("image_width", sent_w) or sent_w
    resp_h = data.get("image_height", sent_h) or sent_h
    sx = orig_w / resp_w
    sy = orig_h / resp_h

    results = []
    for mask in data.get("masks", []):
        bbox = mask.get("bbox", [])
        if len(bbox) == 4:
            results.append({
                "label": query,
                "bbox": [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy],
            })
    return results


def check_health(fp_url: str = DEFAULT_FP_URL) -> bool:
    """Check if FP server is reachable."""
    try:
        resp = requests.get(f"{fp_url}/v1/health", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False
