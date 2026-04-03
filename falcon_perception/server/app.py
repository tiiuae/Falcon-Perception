# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""
FastAPI application for Falcon Perception inference.

Endpoints
---------
POST /v1/predictions        — JSON body (image as URL or base64)
POST /v1/predictions/upload — multipart form (file upload)
GET  /v1/health             — readiness probe + GPU info
GET  /v1/status             — per-GPU queue depths
GET  /v1/models             — list available models (OpenAI convention)
GET  /docs                  — auto-generated OpenAPI docs (Swagger UI)
"""

from __future__ import annotations

import asyncio
import base64
import io
import itertools
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import requests
import numpy as np
import pycocotools.mask as mask_util
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from falcon_perception import build_prompt_for_task
from falcon_perception.server.config import ServerConfig
from falcon_perception.server.engine_worker import InferenceResult, WorkerPool
from falcon_perception.server.schemas import (
    ErrorResponse,
    GPUStatus,
    HealthResponse,
    MaskResult,
    PredictionRequest,
    Response,
)
from falcon_perception.server.mask_smoother import smooth_mask_rle
from falcon_perception.server.mask_combiner import render_masks
logger = logging.getLogger(__name__)

_request_counter = itertools.count(1)

IMAGES_DIR: Path  # set in create_app from config


# ── Helpers ────────────────────────────────────────────────────────────



def _load_image_from_request(req: PredictionRequest) -> Image.Image:
    if req.image.base64:
        data = base64.b64decode(req.image.base64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    if req.image.url:
        from falcon_perception.data import load_image
        return load_image(req.image.url).convert("RGB")
    raise HTTPException(
        status_code=400,
        detail="Either image.url or image.base64 must be provided.",
    )
    
    


from falcon_perception.visualization_utils import pair_bbox_entries as _pair_bbox_entries


def _bbox_from_normalized(b: dict, w: int, h: int) -> list[float]:
    """Convert center-format normalised bbox to [x1, y1, x2, y2] pixels."""
    cx, cy, bh, bw = b.get("x", 0.5), b.get("y", 0.5), b.get("h", 0.0), b.get("w", 0.0)
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return [x1, y1, x2, y2]


def _build_masks(
    result: InferenceResult,
    image_width: int,
    image_height: int,
) -> list[MaskResult]:
    paired_bboxes = _pair_bbox_entries(result.bboxes_raw)
    masks: list[MaskResult] = []
    for idx, m in enumerate(result.masks_rle):
        if "counts" not in m or "size" not in m:
            continue
        height, width = m["size"][0], m["size"][1]
        if idx < len(paired_bboxes):
            bbox = _bbox_from_normalized(paired_bboxes[idx], image_width, image_height)
        else:
            bbox = mask_util.toBbox(m).tolist()  # [x, y, w, h] from RLE without full decode
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        masks.append(MaskResult(
            label=f"object {idx + 1}",
            bbox=bbox,
            rle=m,
            height=height,
            width=width,
            color=m.get("color", {})
        ))

    if not masks and paired_bboxes:
        for idx, b in enumerate(paired_bboxes):
            masks.append(MaskResult(
                label=f"object {idx + 1}",
                bbox=_bbox_from_normalized(b, image_width, image_height),
                rle={},
                height=image_height,
                width=image_width,
            ))

    return masks


def _build_response(
    result: InferenceResult,
    *,
    query: str = "",
    image_width: int = 0,
    image_height: int = 0,
    combined_mask: None=None
) -> Response:
    return Response(
        masks=_build_masks(result, image_width, image_height),
        text=result.text,
        query=query,
        image_width=image_width,
        image_height=image_height,
        inference_time_ms=result.inference_time_ms,
        queue_ms=result.queue_ms,
        total_time_ms=result.total_time_ms,
        tokenize_time_ms=result.tokenize_time_ms,
        prefill_time_ms=result.prefill_time_ms,
        decode_time_ms=result.decode_time_ms,
        finalize_time_ms=result.finalize_time_ms,
        num_decode_steps=result.num_decode_steps,
        avg_decode_batch_size=result.avg_decode_batch_size,
        prefill_batch_size=result.prefill_batch_size,
        prefill_tokens=result.prefill_tokens,
        num_preemptions=result.num_preemptions,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        layout_regions=result.layout_regions,
        combined_mask=combined_mask
    )


async def _submit_and_await(
    pool: WorkerPool,
    image_bytes: bytes,
    query: str,
    task: str,
    max_tokens: int,
    min_image_size: int,
    max_image_size: int,
) -> InferenceResult:
    """Submit pre-serialized image bytes to the least-loaded worker and await completion."""
    if not pool.any_ready:
        raise HTTPException(status_code=503, detail="No healthy engines available.")

    prompt = build_prompt_for_task(query, task)

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    worker = pool.select_worker()
    worker.submit(
        prompt=prompt,
        image_bytes=image_bytes,
        max_tokens=max_tokens,
        min_image_size=min_image_size,
        max_image_size=max_image_size,
        task=task,
        future=future,
        loop=loop,
        request_idx=next(_request_counter),
    )

    return await future


# ── App factory ────────────────────────────────────────────────────────


def create_app(config: ServerConfig) -> FastAPI:
    global IMAGES_DIR
    IMAGES_DIR = Path(config.images_dir)
    pool: WorkerPool | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal pool
        pool = WorkerPool(config)
        pool.start_all()
        logger.info("Waiting for %d engine(s) to initialize ...", len(pool.workers))
        pool.wait_ready(timeout=config.startup_timeout)
        n_healthy = len(pool.healthy_workers)
        logger.info(
            "%d / %d engine(s) ready.", n_healthy, len(pool.workers)
        )
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        app.state.pool = pool
        yield
        logger.info("Shutting down engine workers ...")
        pool.stop_all()

    app = FastAPI(
        title="Falcon Perception API",
        description="Multi-GPU inference server with continuous batching.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # ── Endpoints ──────────────────────────────────────────────────────

    @app.post(
        "/v1/predictions",
        response_model=Response,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def predict(req: PredictionRequest):
        """Run inference from a JSON body (image as URL or base64)."""
        if req.task == "segmentation" and pool and not any(w.do_segmentation for w in pool.healthy_workers):
            raise HTTPException(
                status_code=400,
                detail="Segmentation is not supported by the loaded model. Use task='detection' instead.",
            )
        pil_image = _load_image_from_request(req)
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        try:
            result = await _submit_and_await(
                pool, buf.getvalue(), req.query, req.task,
                req.max_tokens, req.min_image_size, req.max_image_size,
            )
        except HTTPException:
            raise
        except Exception:
            logger.exception("Inference failed for /v1/predictions")
            raise HTTPException(status_code=500, detail="Internal inference error.")
        return _build_response(
            result,
            query=req.query,
            image_width=pil_image.width,
            image_height=pil_image.height,
        )

    @app.post(
        "/v1/predictions/upload",
        response_model=Response,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def predict_upload(
        image: UploadFile = File(..., description="Image file (JPEG/PNG)"),
        query: str = Form(..., description="Text query, e.g. 'segment the dog'"),
        task: str = Form("segmentation", description="segmentation | detection | ocr | ocr_plain | ocr_layout"),
        max_tokens: int = Form(config.max_tokens),
        min_image_size: int = Form(config.min_image_size),
        max_image_size: int = Form(config.max_image_size),
    ):
        """Run inference from a multipart file upload."""
        if task == "segmentation" and pool and not any(w.do_segmentation for w in pool.healthy_workers):
            raise HTTPException(
                status_code=400,
                detail="Segmentation is not supported by the loaded model. Use task='detection' instead.",
            )
        data = await image.read()
        pil_image = Image.open(io.BytesIO(data)).convert("RGB")
        try:
            result = await _submit_and_await(
                pool, data, query, task,
                max_tokens, min_image_size, max_image_size,
            )
        except HTTPException:
            raise
        except Exception:
            logger.exception("Inference failed for /v1/predictions/upload")
            raise HTTPException(status_code=500, detail="Internal inference error.")
        return _build_response(
            result,
            query=query,
            image_width=pil_image.width,
            image_height=pil_image.height,
        )

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        """Readiness probe — returns GPU status."""
        if pool is None or not pool.any_ready:
            return HealthResponse(status="loading", num_gpus=0)

        gpus = []
        for w in pool.workers:
            s = w.get_status()
            gpus.append(
                GPUStatus(
                    gpu_id=s["gpu_id"],
                    device_name=torch.cuda.get_device_name(s["gpu_id"]),
                    waiting=s.get("waiting", 0),
                    running=s.get("running", 0),
                    vram_allocated_gib=torch.cuda.memory_allocated(s["gpu_id"]) / (1024 ** 3),
                    vram_reserved_gib=torch.cuda.memory_reserved(s["gpu_id"]) / (1024 ** 3),
                )
            )
        is_ocr = any(w.is_ocr for w in pool.healthy_workers)
        if is_ocr:
            supported_tasks = ["ocr_plain", "ocr_layout"]
        elif any(w.do_segmentation for w in pool.healthy_workers):
            supported_tasks = ["segmentation", "detection"]
        else:
            supported_tasks = ["detection"]

        return HealthResponse(
            status="ready",
            num_gpus=len(gpus),
            gpus=gpus,
            model_id=config.hf_model_id,
            supported_tasks=supported_tasks,
        )

    @app.get("/v1/status")
    async def status():
        """Per-GPU queue depths and memory usage."""
        if pool is None:
            return {"status": "loading", "workers": []}
        return {
            "status": "ready" if pool.any_ready else "loading",
            "workers": [w.get_status() for w in pool.workers],
        }

    @app.get("/v1/models")
    async def list_models():
        """OpenAI-compatible model listing."""
        is_ocr = pool is not None and any(w.is_ocr for w in pool.healthy_workers)
        has_segm = pool is not None and any(w.do_segmentation for w in pool.healthy_workers)
        if is_ocr:
            model_id = "falcon-ocr"
        elif has_segm:
            model_id = "falcon-perception"
        else:
            model_id = "falcon-perception-300m"
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "tii",
                }
            ],
        }

    # ── Image cache endpoints (frontend) ─────────────────────────────

    @app.get("/upload/check")
    async def check_image(image_id: str):
        """Check whether an image has already been uploaded to the server cache."""
        image_path = IMAGES_DIR / f"{image_id}.jpg"
        return {"exists": image_path.exists()}

    @app.post("/upload")
    async def upload_image(
        file: UploadFile = File(...),
        image_id: str = Form(...),
    ):
        """Upload an image to the server cache for later use by /segment."""
        image_path = IMAGES_DIR / f"{image_id}.jpg"

        if image_path.exists():
            return {"image_id": image_id, "already_existed": True}

        file_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        image.save(image_path, format="JPEG", quality=92)
        return {"image_id": image_id, "already_existed": False}

    @app.post("/segment", response_model=Response)
    async def segment(
        image_hash: str = Form(...),
        query: str = Form(...),
        min_size: int = Form(...),
        max_size: int = Form(...),
        backend_render: bool = Form(False),
    ):
        if pool and not any(w.do_segmentation for w in pool.healthy_workers):
            raise HTTPException(
                status_code=400,
                detail="Segmentation is not supported by the loaded model. Use /detect instead.",
            )
        logger.info("Segment: query='%s', file='%s', minSize=%d, maxSize=%d, backendRender=%d", 
                     query, image_hash, min_size, max_size, backend_render)

        image_path = IMAGES_DIR / f"{image_hash}.jpg"
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image '{image_hash}' not found. Upload it first via /upload.")

        image = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        try:
            result = await _submit_and_await(
                pool, buf.getvalue(), query, "segmentation",
                8192, min_size, max_size,
            )
        except HTTPException:
            raise
        except Exception:
            logger.exception("Inference failed for /segment")
            raise HTTPException(status_code=500, detail="Internal inference error.")
        rle_masks = result.masks_rle
        smooth_masks = []
        for rle in rle_masks:
            smooth_masks.append(smooth_mask_rle(rle))
        if backend_render:
            logger.info(" Rendering in backend")
            smooth_masks, combined_mask = render_masks(smooth_masks, img_w=image.width, img_h=image.height)
        else:
            combined_mask = None
        result.masks_rle = smooth_masks
        
        return _build_response(
            result,
            query=query,
            image_width=image.width,
            image_height=image.height,
            combined_mask= combined_mask
        )


    @app.post("/detect", response_model=Response)
    async def detect(
        image_hash: str = Form(...),
        query: str = Form(...),
        min_size: int = Form(...),
        max_size: int = Form(...),
    ):
        logger.info("detect: query='%s', file='%s', minSize=%d, maxSize=%d",
                     query, image_hash, min_size, max_size)

        image_path = IMAGES_DIR / f"{image_hash}.jpg"
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image '{image_hash}' not found. Upload it first via /upload.")

        image = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        result = await _submit_and_await(
            pool, buf.getvalue(), query, "detection",
            8192, min_size, max_size,
        )
        return _build_response(
            result,
            query=query,
            image_width=image.width,
            image_height=image.height,
        )


    @app.post("/ocr_plain", response_model=Response)
    async def ocr_plain(
        image_hash: str = Form(...),
        query: str = Form(...),
        min_size: int = Form(...),
        max_size: int = Form(...),
    ):
        logger.info("ocr_plain: query='%s', file='%s', minSize=%d, maxSize=%d",
                     query, image_hash, min_size, max_size)

        image_path = IMAGES_DIR / f"{image_hash}.jpg"
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image '{image_hash}' not found. Upload it first via /upload.")

        image = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        result = await _submit_and_await(
            pool, buf.getvalue(), query, "ocr_plain",
            8192, min_size, max_size,
        )
        return _build_response(
            result,
            query=query,
            image_width=image.width,
            image_height=image.height,
        )
        
    @app.post("/ocr_layout", response_model=Response)
    async def ocr_layout(
        image_hash: str = Form(...),
        query: str = Form(...),
        min_size: int = Form(...),
        max_size: int = Form(...),
    ):
        logger.info("ocr_layout: query='%s', file='%s', minSize=%d, maxSize=%d",
                     query, image_hash, min_size, max_size)

        image_path = IMAGES_DIR / f"{image_hash}.jpg"
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image '{image_hash}' not found. Upload it first via /upload.")

        image = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        result = await _submit_and_await(
            pool, buf.getvalue(), query, "ocr_layout",
            8192, min_size, max_size,
        )
        return _build_response(
            result,
            query=query,
            image_width=image.width,
            image_height=image.height,
        )
        
    return app









        

