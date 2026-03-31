# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Pydantic models for the Falcon Perception API."""

from __future__ import annotations

import time
import uuid
from typing import Literal
from typing import Optional
from pydantic import BaseModel, Field


# -- Request -----------------------------------------------------------------


class ImageInput(BaseModel):
    """Image source -- exactly one of *url* or *base64* must be provided."""

    url: str | None = None
    base64: str | None = None


class PredictionRequest(BaseModel):
    """POST /v1/predictions -- JSON body."""

    image: ImageInput
    query: str
    task: Literal["segmentation", "detection", "ocr_plain", "ocr_layout"] = "segmentation"
    max_tokens: int = 8192
    min_image_size: int = 256
    max_image_size: int = 1024


# -- Response ----------------------------------------------------------------


class MaskResult(BaseModel):
    label: str = "object"
    bbox: list[float] = Field(default_factory=list, description="[x1, y1, x2, y2]")
    rle: dict = Field(default_factory=dict, description="COCO RLE encoding {counts, size}")
    color: dict= None
    height: int
    width: int

class CombinedMask(BaseModel):
    data: str        # base64 raw RGBA bytes
    width: int
    height: int

class Response(BaseModel):
    id: str = Field(default_factory=lambda: f"pred_{uuid.uuid4().hex[:12]}")
    model: str = "falcon-perception"
    created: int = Field(default_factory=lambda: int(time.time()))

    masks: list[MaskResult] = []
    text: str = ""
    query: str = ""
    prompt_type: str = "text"
    image_width: int = 0
    image_height: int = 0

    inference_time_ms: float = 0.0
    queue_ms: float = 0.0
    total_time_ms: float = 0.0
    tokenize_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    finalize_time_ms: float = 0.0
    num_decode_steps: int = 0
    avg_decode_batch_size: float = 0.0
    prefill_batch_size: int = 0
    prefill_tokens: int = 0
    num_preemptions: int = 0

    input_tokens: int = 0
    output_tokens: int = 0

    layout_regions: list[dict] = Field(
        default_factory=list,
        description="For ocr_layout: [{category, bbox, score, text}]",
    )
    combined_mask: Optional[CombinedMask] = None


# -- Health / Status ---------------------------------------------------------


class GPUStatus(BaseModel):
    gpu_id: int
    device_name: str
    waiting: int
    running: int
    vram_allocated_gib: float = 0.0
    vram_reserved_gib: float = 0.0


class HealthResponse(BaseModel):
    status: Literal["ready", "loading"]
    num_gpus: int = 0
    gpus: list[GPUStatus] = []
    model_id: str = ""
    supported_tasks: list[str] = []


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
