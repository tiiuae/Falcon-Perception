import base64
import hashlib
import io
import re
import time
from typing import cast

import numpy as np
import requests
import streamlit as st
from PIL import Image

from falcon_perception.data import load_image
from falcon_perception.visualization_utils import (
    decode_coco_rle,
    mask_nms,
    overlay_detections_on_image_v2,
)


DEFAULT_SERVER_URL = "http://localhost:7860"


def call_prediction_api(
    server_url: str,
    image: Image.Image,
    query: str,
    task: str = "segmentation",
    max_tokens: int = 8192,
    min_image_size: int = 256,
    max_image_size: int = 1024,
    upscale_factor: int = 16,
) -> dict:
    """POST image + query to the inference server and return the JSON response."""
    # Resize client-side so masks are produced at capped resolution; avoids
    # large RLE transfers and keeps decode/render fast.
    w, h = image.size
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        image = image.resize(
            (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS,
        )

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "image": {"base64": image_b64},
        "query": query,
        "task": task,
        "max_tokens": max_tokens,
        "min_image_size": min_image_size,
        "max_image_size": max_image_size,
        "upscale_factor": upscale_factor,
    }
    resp = requests.post(
        f"{server_url.rstrip('/')}/v1/predictions",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def _pixel_bbox_to_normalized(bbox: list[float], img_w: int, img_h: int) -> dict:
    """Convert pixel [x1,y1,x2,y2] to the normalized center+size dicts
    that the overlay renderer expects."""
    x1, y1, x2, y2 = bbox
    return {
        "xy": {"x": (x1 + x2) / 2 / img_w, "y": (y1 + y2) / 2 / img_h},
        "hw": {"w": (x2 - x1) / img_w, "h": (y2 - y1) / img_h},
    }


def _cap_image_for_display(img: Image.Image, max_side: int) -> Image.Image:
    """Down-scale *img* so the longest side is at most *max_side* pixels."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def render_overlay_from_response(
    response: dict,
    original_image: Image.Image,
    *,
    task: str = "segmentation",
    nms_enabled: bool = False,
    nms_iou_threshold: float = 0.5,
    max_vis_size: int = 2048,
) -> tuple[np.ndarray | None, int, int]:
    """Decode API response masks + bboxes and render an overlay on the image.

    Returns (overlay_image_or_None, n_detections_drawn, n_suppressed_by_nms).
    """
    mask_results = response.get("masks", [])
    if not mask_results:
        return None, 0, 0

    img_w = response.get("image_width", original_image.width)
    img_h = response.get("image_height", original_image.height)

    detections = []
    for m in mask_results:
        rle = m.get("rle") or {}
        det: dict = {}

        if task != "detection":
            mask = decode_coco_rle(rle) if rle else None
            if mask is not None and mask.any():
                det["mask"] = mask

        bbox = m.get("bbox", [])
        if len(bbox) == 4:
            det.update(_pixel_bbox_to_normalized(bbox, img_w, img_h))

        if det:
            detections.append(det)

    if not detections:
        return None, 0, 0

    n_suppressed = 0
    if nms_enabled and task != "detection":
        detections, n_suppressed = mask_nms(detections, iou_threshold=nms_iou_threshold)

    if not detections:
        return None, 0, n_suppressed

    display_img = _cap_image_for_display(original_image, max_vis_size)
    overlay = overlay_detections_on_image_v2(
        display_img, detections, draw_bbox=True, masks_are_binary=True,
    )
    return overlay, len(detections), n_suppressed


def load_image_from_source(uploaded_file, url: str | None):
    if uploaded_file is not None:
        data = uploaded_file.getbuffer()
        key = f"upload:{uploaded_file.name}:{hashlib.md5(data).hexdigest()}"
        return Image.open(io.BytesIO(data)).convert("RGB"), key
    if url:
        img = cast(Image.Image, load_image(url)).convert("RGB")
        key = f"url:{url}"
        return img, key
    return None, None


_HTML_BLOCK_RE = re.compile(
    r"(<(table|div|ul|ol|dl|pre|blockquote|details|figure|section|article|aside|header|footer|nav|form)\b[^>]*>.*?</\2>)",
    re.DOTALL | re.IGNORECASE,
)


def _render_mixed_content(text: str):
    """Render text with markdown and inline HTML blocks.

    Splits on top-level HTML elements (e.g. <table>...</table>), renders
    blocks with st.html and surrounding text with st.markdown.
    """
    if not re.search(r"<(?:table|div|ul|ol|dl|pre)\b", text, re.IGNORECASE):
        stripped = text.strip()
        if stripped.startswith("<") and ">" in stripped:
            st.html(text)
        else:
            st.markdown(text)
        return

    # split() yields [text, block, tagname, text, block, tagname, ...]; odd indices are blocks
    parts = _HTML_BLOCK_RE.split(text)
    i = 0
    while i < len(parts):
        if i % 3 == 0:
            chunk = parts[i].strip()
            if chunk:
                st.markdown(chunk)
        elif i % 3 == 1:
            st.html(parts[i])
        i += 1


def main():
    st.set_page_config(page_title="Falcon Perception", layout="wide")
    st.title("Falcon Perception")

    if "predictions" not in st.session_state:
        st.session_state["predictions"] = []
    if "image_key" not in st.session_state:
        st.session_state["image_key"] = None

    with st.sidebar:
        st.header("Settings")
        server_url = st.text_input("API server URL", value=DEFAULT_SERVER_URL)

        all_tasks = ["segmentation", "detection", "ocr_plain", "ocr_layout"]
        supported_tasks = all_tasks
        model_id = ""
        try:
            health = requests.get(f"{server_url.rstrip('/')}/v1/health", timeout=3).json()
            status = health.get("status", "unknown")
            num_gpus = health.get("num_gpus", 0)
            model_id = health.get("model_id", "")
            server_tasks = health.get("supported_tasks", [])
            if server_tasks:
                supported_tasks = [t for t in all_tasks if t in server_tasks]
            if status == "ready":
                label = f"Server ready ({num_gpus} GPU{'s' if num_gpus != 1 else ''})"
                if model_id:
                    label += f" — {model_id.split('/')[-1]}"
                st.success(label)
            elif status == "loading":
                st.warning("Server is loading engines...")
            else:
                st.error(f"Server status: {status}")
        except Exception:
            st.error("Cannot reach server")

        st.markdown("---")
        task_type = st.selectbox("Task", supported_tasks, index=0)
        url = st.text_input("...or enter an image URL")
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if task_type in ("segmentation", "detection"):
            prompt_input = st.text_area("Text prompt", value="", height=80)
        else:
            prompt_input = None
        run_btn = st.button("Run inference", type="primary")

        if task_type == "segmentation":
            st.markdown("---")
            st.header("Post-processing")
            nms_enabled = st.toggle("Mask NMS", value=False, help="Suppress overlapping masks using greedy IoU-based NMS.")
            nms_threshold = st.slider(
                "NMS IoU threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05,
                disabled=not nms_enabled,
                help="Masks with pairwise IoU above this threshold are suppressed.",
            )
        else:
            nms_enabled = False
            nms_threshold = 0.5

        st.markdown("---")
        min_img_size = st.number_input(
            "Min image size (px)", min_value=64, max_value=2048, value=256, step=32,
            help="Minimum dimension the image is resized to before inference.",
        )
        max_img_size = st.number_input(
            "Max image size (px)", min_value=64, max_value=2048, value=1024, step=32,
            help="Maximum dimension the image is resized to before inference.",
        )
        if min_img_size > max_img_size:
            st.warning("Min image size should not exceed max image size.")

    image, image_key = load_image_from_source(uploaded_file, url)

    if image_key and image_key != st.session_state["image_key"]:
        st.session_state["image_key"] = image_key
        st.session_state["predictions"] = []

    if image:
        with st.expander("Input image", expanded=False):
            st.image(image, width="content")

    if run_btn:
        if not image:
            st.error("Please upload an image or provide a valid URL.")
            return
        if task_type in ("segmentation", "detection") and not (prompt_input or "").strip():
            st.error(f"Enter a text prompt for {task_type}.")
            return

        query = (prompt_input or "").strip()

        status_box = st.status("Running inference on server...", expanded=True)
        with status_box:
            try:
                response = call_prediction_api(
                    server_url=server_url,
                    image=image,
                    query=query,
                    task=task_type,
                    min_image_size=min_img_size,
                    max_image_size=max_img_size,
                )
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to server at {server_url}")
                status_box.update(label="Request failed", state="error")
                return
            except requests.exceptions.HTTPError as e:
                st.error(f"Server error: {e.response.status_code} — {e.response.text[:500]}")
                status_box.update(label="Request failed", state="error")
                return
            except Exception as e:
                st.error(f"Request failed: {e}")
                status_box.update(label="Request failed", state="error")
                return

            text = response.get("text", "")
            num_tokens = response.get("output_tokens", 0)
            num_masks = len(response.get("masks", []))
            server_ms = response.get("inference_time_ms", 0)

            st.write(f"Inference done — {num_tokens} tokens, {server_ms/1000:.2f}s server time")

            overlay = None
            n_drawn = 0
            n_suppressed = 0
            render_s = 0.0
            if task_type == "segmentation" and num_masks > 0:
                nms_label = f" (NMS IoU={nms_threshold:.2f})" if nms_enabled else ""
                status_box.update(label=f"Drawing {num_masks} masks{nms_label}...", expanded=True)
                t0 = time.perf_counter()
                overlay, n_drawn, n_suppressed = render_overlay_from_response(
                    response, image,
                    task=task_type,
                    nms_enabled=nms_enabled,
                    nms_iou_threshold=nms_threshold,
                )
                render_s = time.perf_counter() - t0
                nms_msg = f", {n_suppressed} suppressed by NMS" if n_suppressed else ""
                st.write(f"Rendering done — {n_drawn} masks drawn{nms_msg}, {render_s:.2f}s")
            elif task_type == "detection" and num_masks > 0:
                status_box.update(label=f"Drawing {num_masks} bboxes...", expanded=True)
                t0 = time.perf_counter()
                overlay, n_drawn, _ = render_overlay_from_response(
                    response, image, task="detection",
                )
                render_s = time.perf_counter() - t0
                st.write(f"Rendering done — {n_drawn} bboxes drawn, {render_s:.2f}s")

            status_box.update(label="Complete", state="complete", expanded=False)

        st.session_state["predictions"].insert(
            0,
            {
                "prompt": query if task_type in ("segmentation", "detection") else f"[{task_type.upper()}]",
                "task": task_type,
                "text": text,
                "overlay": overlay,
                "num_tokens": num_tokens,
                "num_masks": num_masks,
                "num_drawn": n_drawn,
                "num_suppressed": n_suppressed,
                "input_tokens": response.get("input_tokens", 0),
                "response": response,
                "render_time_s": render_s,
                "prediction_id": response.get("id", ""),
            },
        )

    predictions_container = st.container()
    with predictions_container:
        if st.session_state["predictions"]:
            st.divider()
            st.subheader("Predictions")
            for idx, pred in enumerate(st.session_state["predictions"], start=1):
                st.markdown(f"**Prediction #{idx} — {pred['prompt']}**")

                pred_task = pred.get("task", "segmentation")
                resp = pred.get("response", {})
                num_tokens = pred.get("num_tokens", 0)
                input_tokens = pred.get("input_tokens", 0)
                inference_ms = resp.get("inference_time_ms", 0)
                queue_ms = resp.get("queue_ms", 0)
                total_ms = resp.get("total_time_ms", 0) or (inference_ms + queue_ms)

                if pred_task == "ocr_layout":
                    regions = resp.get("layout_regions", [])
                    if regions:
                        for r_idx, region in enumerate(regions):
                            cat = region.get("category", "")
                            score = region.get("score", 0)
                            region_text = region.get("text", "")
                            st.markdown(f"**{r_idx + 1}. {cat}** (score: {score:.2f})")
                            if not region_text:
                                st.caption("(empty)")
                            elif cat.strip().lower() == "table":
                                st.html(region_text)
                            else:
                                _render_mixed_content(region_text)
                    else:
                        st.info("No layout regions detected.")
                elif pred_task == "ocr_plain":
                    ocr_text = pred.get("text", "")
                    if ocr_text:
                        st.markdown("**Rendered output**")
                        # in ocr_plain, single newline is not considered as a new line
                        # so we need to replace all single newlines with <br>
                        ocr_text = ocr_text.replace("\n", "<br>")
                        _render_mixed_content(ocr_text)
                        with st.expander("Raw text", expanded=False):
                            st.code(ocr_text, language=None)
                    else:
                        st.info("No text extracted.")
                elif pred_task == "detection":
                    num_masks = pred.get("num_masks", 0)
                    if num_masks == 0:
                        st.info("No detections found.")
                    elif pred.get("overlay") is not None:
                        st.image(pred["overlay"], caption="Overlay with bounding boxes", width="stretch")
                    raw_text = pred.get("text", "")
                    if raw_text:
                        with st.expander("Raw text", expanded=False):
                            st.code(raw_text, language=None)
                else:
                    num_masks = pred.get("num_masks", 0)
                    if num_masks == 0:
                        st.info("No detections found.")
                    elif pred.get("overlay") is not None:
                        st.image(pred["overlay"], caption="Overlay with boxes/masks", width="stretch")

                tok_ms = resp.get("tokenize_time_ms", 0)
                pf_ms = resp.get("prefill_time_ms", 0)
                dc_ms = resp.get("decode_time_ms", 0)
                fin_ms = resp.get("finalize_time_ms", 0)
                dc_steps = resp.get("num_decode_steps", 0)
                avg_bs = resp.get("avg_decode_batch_size", 0)
                pf_bs = resp.get("prefill_batch_size", 0)
                pf_tok = resp.get("prefill_tokens", 0)
                n_preempt = resp.get("num_preemptions", 0)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Input tokens", input_tokens)
                c2.metric("Output tokens", num_tokens)
                if pred_task in ("segmentation", "detection"):
                    n_drawn = pred.get("num_drawn", 0)
                    n_supp = pred.get("num_suppressed", 0)
                    label = "Masks" if pred_task == "segmentation" else "Detections"
                    c3.metric(label, f"{n_drawn}" if not n_supp else f"{n_drawn} ({n_supp} NMS)")
                else:
                    c3.metric("Queue", f"{queue_ms:.0f} ms")
                c4.metric("Inference", f"{inference_ms:.0f} ms")

                with st.expander("Pipeline breakdown", expanded=False):
                    p1, p2, p3, p4, p5 = st.columns(5)
                    p1.metric("Tokenize", f"{tok_ms:.0f} ms")
                    p2.metric("Prefill", f"{pf_ms:.0f} ms")
                    p3.metric("Decode", f"{dc_ms:.0f} ms")
                    p4.metric("Finalize", f"{fin_ms:.0f} ms")
                    render_ms = pred.get("render_time_s", 0) * 1000
                    p5.metric("Render", f"{render_ms:.0f} ms")

                    s1, s2, s3, s4, s5 = st.columns(5)
                    s1.metric("Prefill batch", pf_bs)
                    s2.metric("Prefill tokens", pf_tok)
                    s3.metric("Decode steps", dc_steps)
                    s4.metric("Avg decode BS", f"{avg_bs:.1f}")
                    s5.metric("Preemptions", n_preempt)

                st.markdown("---")


if __name__ == "__main__":
    main()
