# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""
Continuous-batching engine workers — one per GPU.

Each worker runs in a **separate process** with an isolated CUDA context,
enabling torch.compile + CUDA graph capture on every GPU independently.
The main process communicates with workers via ``multiprocessing.Queue``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import dataclass, field

import torch

from falcon_perception.server.config import ServerConfig

logger = logging.getLogger(__name__)


# ── IPC types (picklable, cross-process) ───────────────────────────────


@dataclass
class WorkerRequest:
    """Sent from the main process to a worker subprocess."""

    request_id: int
    prompt: str
    image_bytes: bytes
    max_tokens: int
    min_image_size: int
    max_image_size: int
    task: str = "segmentation"
    enqueue_time: float = field(default_factory=time.monotonic)


@dataclass
class InferenceResult:
    """Inference output — picklable, returned to the main process."""

    text: str
    masks_rle: list[dict]
    bboxes_raw: list[dict]
    image_size: tuple[int, int] | None
    input_tokens: int
    output_tokens: int

    # Timing: inference = sum of phases, queue = scheduling/IPC overhead,
    # total = inference + queue (end-to-end wall clock).
    inference_time_ms: float = 0.0
    queue_ms: float = 0.0
    total_time_ms: float = 0.0

    # Per-request timing breakdown (set by _harvest_done)
    tokenize_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    finalize_time_ms: float = 0.0
    num_decode_steps: int = 0
    avg_decode_batch_size: float = 0.0
    prefill_batch_size: int = 0
    prefill_tokens: int = 0
    num_preemptions: int = 0

    # OCR layout mode: [{category, bbox, score, text}]
    layout_regions: list[dict] = field(default_factory=list)


@dataclass
class WorkerResponse:
    """Sent from a worker subprocess back to the main process."""

    request_id: int
    gpu_id: int
    result: InferenceResult | None = None
    error: str | None = None


# Worker subprocess


def _worker_process_main(
    gpu_id: int,
    config: ServerConfig,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    ready_event,  # mp.Event
    failed_value,  # mp.Value('b', False)
    is_ocr_value,  # mp.Value('b', False)
):
    """Entry point for a GPU worker subprocess.

    Builds the engine (torch.compile + CUDA graphs) in full isolation,
    then runs the continuous-batching serve loop.
    """
    from falcon_perception import setup_torch_config

    setup_torch_config()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(f"falcon_perception.server.worker.gpu{gpu_id}")

    try:
        torch.cuda.set_device(gpu_id)
        engine = _build_engine(gpu_id, config, log)
        is_ocr_value.value = not engine.model.args.perception_heads
        _warmup_engine(engine, log)
        ready_event.set()

        with torch.inference_mode():
            while True:
                _drain_request_queue(
                    engine, request_queue, response_queue, gpu_id, log
                )

                has_running = bool(engine.running)
                has_waiting = bool(engine.waiting)

                if has_running or has_waiting:
                    can_prefill = has_waiting and bool(
                        engine.paged_kv_cache.free_batch_idx
                    )
                    if has_running or can_prefill:
                        try:
                            engine.run_one_step()
                        except Exception as exc:
                            import sys
                            import traceback

                            tb = traceback.format_exc()
                            log.error("Error in run_one_step:\n%s", tb)
                            sys.stderr.flush()
                            _fail_all_pending(
                                engine,
                                response_queue,
                                gpu_id,
                                f"Engine step failed: {type(exc).__name__}: {exc}",
                            )
                            continue

                    _harvest_done(engine, response_queue, gpu_id, log)

                    if not has_running and not can_prefill:
                        time.sleep(0.005)
                else:
                    time.sleep(0.001)

    except SystemExit:
        log.info("Shutting down.")
    except Exception:
        import sys

        log.exception("Fatal error in worker process")
        sys.stderr.flush()
        failed_value.value = True
        ready_event.set()


# ── Engine construction (inside worker subprocess) ─────────────────────


@torch.inference_mode()
def _build_engine(gpu_id, config, log):
    from falcon_perception.data import ImageProcessor
    from falcon_perception import load_from_hf_export

    log.info("Loading model from %s ...", config.hf_model_id)
    device = torch.device(f"cuda:{gpu_id}")
    dtype = getattr(torch, config.dtype, torch.float32)

    model, tokenizer, model_args = load_from_hf_export(
        hf_model_id=config.hf_model_id,
        hf_revision=config.hf_revision,
        hf_local_dir=config.hf_local_dir,
    )
    model = model.to(device=device, dtype=dtype)
    is_ocr = not model_args.perception_heads
    log.info("Model dtype: %s on %s (variant=%s)", dtype, device, "ocr" if is_ocr else "perception")
    if config.compile:
        torch._inductor.config.triton.cudagraphs = False
        model.compile(mode="default")

    image_processor = ImageProcessor(patch_size=16, merge_size=1)

    if is_ocr:
        from falcon_perception.paged_ocr_inference import OCRInferenceEngine

        engine = OCRInferenceEngine(
            model,
            tokenizer,
            image_processor,
            max_batch_size=config.max_batch_size,
            max_seq_length=config.max_seq_length,
            n_pages=config.n_pages,
            page_size=config.page_size,
            prefill_length_limit=config.prefill_length_limit,
            capture_cudagraph=config.cudagraph,
            max_decode_steps_between_prefills=config.max_decode_steps_between_prefills,
        )
    else:
        from falcon_perception.paged_inference import PagedInferenceEngine
        engine = PagedInferenceEngine(
            model,
            tokenizer,
            image_processor,
            max_batch_size=config.max_batch_size,
            max_seq_length=config.max_seq_length,
            n_pages=config.n_pages,
            page_size=config.page_size,
            prefill_length_limit=config.prefill_length_limit,
            capture_cudagraph=config.cudagraph,
            max_decode_steps_between_prefills=config.max_decode_steps_between_prefills,
        )

    engine.temperature = config.temperature
    engine.top_k = config.top_k

    # Compound-request state for ocr_layout (1 API request → N crop sequences)
    engine._compound_state = {}  # type: ignore[attr-defined]
    engine._layout_threshold = config.layout_threshold  # type: ignore[attr-defined]

    log.info("Engine ready (variant=%s).", "ocr" if is_ocr else "perception")
    return engine


@torch.inference_mode()
def _warmup_engine(engine, log):
    """Run a short dummy inference to trigger torch.compile JIT compilation.

    Without this, the first real request on each worker pays the full JIT
    cost (often 30-60 s), which is especially problematic with multiple GPU
    workers: only the worker that receives the first user request gets
    warmed up, leaving the rest cold until more concurrent traffic arrives.
    """
    from falcon_perception.paged_inference import SamplingParams, Sequence
    from PIL import Image

    dummy_img = Image.new("RGB", (1024, 1024), color=(128, 128, 128))
    is_ocr = not engine.model.args.perception_heads

    if is_ocr:
        prompt = "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>"
        task = "ocr"
    else:
        from falcon_perception import build_prompt_for_task
        prompt = build_prompt_for_task("all objects", "segmentation")
        task = "segmentation"

    seq = Sequence(
        text=prompt,
        image=dummy_img,
        min_image_size=256,
        max_image_size=1024,
        task=task,
    )
    sampling = SamplingParams(
        max_new_tokens=32,
        stop_token_ids=[
            engine.tokenizer.eos_token_id,
            engine.tokenizer.end_of_query_token_id,
        ],
    )

    log.info("Running warmup inference ...")
    t0 = time.monotonic()
    engine.generate([seq], sampling_params=sampling, use_tqdm=False, print_stats=False)
    log.info("Warmup: generate done in %.1fs", time.monotonic() - t0)

    if is_ocr and hasattr(engine, "load_layout_model"):
        t0 = time.monotonic()
        engine.load_layout_model()
        engine.run_layout_detection([dummy_img])
        log.info("Warmup: layout model done in %.1fs", time.monotonic() - t0)


# ── Serve-loop helpers (inside worker subprocess) ──────────────────────


def _drain_request_queue(engine, request_queue, response_queue, gpu_id, log):
    """Move new requests from the mp.Queue into engine.waiting."""
    from falcon_perception.paged_inference import SamplingParams, Sequence
    from PIL import Image

    while True:
        try:
            req = request_queue.get_nowait()
        except queue.Empty:
            break

        if req is None:
            raise SystemExit(0)

        try:
            if req.task == "ocr_layout":
                _enqueue_layout_request(
                    engine, req, response_queue, gpu_id, log,
                    layout_threshold=getattr(engine, "_layout_threshold", 0.3),
                )
                continue

            pil_image = Image.open(io.BytesIO(req.image_bytes)).convert("RGB")
            seq = Sequence(
                text=req.prompt,
                image=pil_image,
                min_image_size=req.min_image_size,
                max_image_size=req.max_image_size,
                request_idx=req.request_id,
                task=req.task,
            )
            stop_ids = [
                engine.tokenizer.eos_token_id,
                engine.tokenizer.end_of_query_token_id,
            ]
            seq.sampling_params = SamplingParams(
                max_new_tokens=req.max_tokens,
                stop_token_ids=stop_ids,
            )
            seq._api_request_id = req.request_id  # type: ignore[attr-defined]
            seq._api_enqueue_time = req.enqueue_time  # type: ignore[attr-defined]
            engine.waiting.append(seq)
        except Exception:
            log.exception("Error creating sequence for request %d", req.request_id)
            response_queue.put(
                WorkerResponse(
                    request_id=req.request_id,
                    gpu_id=gpu_id,
                    error="Failed to create sequence",
                )
            )


def _enqueue_layout_request(engine, req, response_queue, gpu_id, log, *, layout_threshold: float = 0.3):
    """Run layout detection and enqueue crop sequences for an ocr_layout request.

    Layout detection runs synchronously here (fast, ~20-50 ms per image).
    Each detected text region becomes a plain OCR sequence in engine.waiting.
    Results are assembled when all crops complete (see _harvest_done).
    """
    from falcon_perception.paged_ocr_inference import (
        OCRInferenceEngine,
        dedup_overlapping_detections,
        filter_nested_detections,
    )
    from falcon_perception.paged_inference import SamplingParams, Sequence
    from PIL import Image

    pil_image = Image.open(io.BytesIO(req.image_bytes)).convert("RGB")

    # Lazy-load the 3rd-party layout model on first use
    engine.load_layout_model()

    dets = engine.run_layout_detection([pil_image], threshold=layout_threshold)[0]
    dets = dedup_overlapping_detections(filter_nested_detections(dets))

    # Build crop sequences for text-bearing regions
    crop_det_indices: list[int] = []
    img_w, img_h = pil_image.size
    stop_ids = [engine.tokenizer.eos_token_id, engine.tokenizer.end_of_query_token_id]

    for seq, det_idx in OCRInferenceEngine.build_crop_sequences(
        pil_image, dets,
        min_image_size=req.min_image_size,
        max_image_size=req.max_image_size,
    ):
        seq.request_idx = req.request_id * 100_000 + det_idx
        seq.sampling_params = SamplingParams(max_new_tokens=req.max_tokens, stop_token_ids=stop_ids)
        seq._api_request_id = None  # type: ignore[attr-defined]  # skip normal harvest
        seq._compound_parent_id = req.request_id  # type: ignore[attr-defined]
        seq._compound_det_idx = det_idx  # type: ignore[attr-defined]
        engine.waiting.append(seq)
        crop_det_indices.append(det_idx)

    if not crop_det_indices:
        # No text-bearing regions found — return empty result immediately
        response_queue.put(WorkerResponse(
            request_id=req.request_id,
            gpu_id=gpu_id,
            result=InferenceResult(
                text="", masks_rle=[], bboxes_raw=[],
                image_size=(img_w, img_h),
                input_tokens=0, output_tokens=0,
                inference_time_ms=(time.monotonic() - req.enqueue_time) * 1000,
            ),
        ))
        return

    engine._compound_state[req.request_id] = {
        "total": len(crop_det_indices),
        "completed": 0,
        "dets": dets,
        "crop_texts": {},
        "crop_stats": {},
        "enqueue_time": req.enqueue_time,
        "image_size": (img_w, img_h),
    }


def _harvest_done(engine, response_queue, gpu_id, log):
    """Send results back for every completed sequence."""
    while engine.done:
        seq = engine.done.popleft()

        # ── Compound (layout) crop completion ──
        parent_id = getattr(seq, "_compound_parent_id", None)
        if parent_id is not None:
            _harvest_compound_crop(engine, seq, parent_id, response_queue, gpu_id, log)
            continue

        # ── Normal single-sequence completion ──
        request_id = getattr(seq, "_api_request_id", None)
        enqueue_time = getattr(seq, "_api_enqueue_time", 0.0)
        if request_id is None:
            continue

        try:
            text = engine.tokenizer.decode(seq.output_ids)
            s = seq.stats
            phase_ms = s.tokenize_ms + s.prefill_ms + s.decode_wall_ms + s.finalize_ms
            total_wall_ms = (time.monotonic() - enqueue_time) * 1000
            result = InferenceResult(
                text=text,
                masks_rle=list(seq.output_aux.masks_rle),
                bboxes_raw=list(seq.output_aux.bboxes_raw),
                image_size=seq.original_image_size,
                input_tokens=seq.input_length,
                output_tokens=seq.output_length,
                inference_time_ms=phase_ms,
                queue_ms=max(0.0, total_wall_ms - phase_ms),
                total_time_ms=total_wall_ms,
                tokenize_time_ms=s.tokenize_ms,
                prefill_time_ms=s.prefill_ms,
                decode_time_ms=s.decode_wall_ms,
                finalize_time_ms=s.finalize_ms,
                num_decode_steps=s.decode_steps,
                avg_decode_batch_size=(
                    s.decode_batch_sum / s.decode_steps
                    if s.decode_steps > 0 else 0.0
                ),
                prefill_batch_size=s.prefill_batch_size,
                prefill_tokens=s.prefill_tokens,
                num_preemptions=s.preemptions,
            )
            response_queue.put(
                WorkerResponse(
                    request_id=request_id, gpu_id=gpu_id, result=result
                )
            )
        except Exception:
            log.exception("Error building result for request %d", request_id)
            response_queue.put(
                WorkerResponse(
                    request_id=request_id,
                    gpu_id=gpu_id,
                    error="Failed to build inference result",
                )
            )


def _harvest_compound_crop(engine, seq, parent_id, response_queue, gpu_id, log):
    """Accumulate a completed layout-crop sequence.

    When all crops for a parent request are done, assemble and send the
    compound result back to the main process.
    """
    from falcon_perception.paged_ocr_inference import LAYOUT_TO_OCR_CATEGORY

    state = engine._compound_state.get(parent_id)
    if state is None:
        return

    try:
        det_idx = seq._compound_det_idx
        text = engine.tokenizer.decode(seq.output_ids)
        text = text.replace("<|end_of_query|>", "").replace("<|endoftext|>", "").strip()
        state["crop_texts"][det_idx] = text
        state["crop_stats"][det_idx] = {
            "input_tokens": seq.input_length,
            "output_tokens": seq.output_length,
            "tokenize_ms": seq.stats.tokenize_ms,
            "prefill_ms": seq.stats.prefill_ms,
            "decode_wall_ms": seq.stats.decode_wall_ms,
            "finalize_ms": seq.stats.finalize_ms,
            "decode_steps": seq.stats.decode_steps,
            "decode_batch_sum": seq.stats.decode_batch_sum,
            "prefill_batch_size": seq.stats.prefill_batch_size,
            "prefill_tokens": seq.stats.prefill_tokens,
            "preemptions": seq.stats.preemptions,
        }
        state["completed"] += 1
    except Exception:
        log.exception("Error processing layout crop for parent %d", parent_id)
        state["completed"] += 1

    if state["completed"] < state["total"]:
        return

    # All crops done — assemble the compound result
    try:
        regions = []
        for det_idx, det in enumerate(state["dets"]):
            cat_key = det["category"].strip().lower()
            if LAYOUT_TO_OCR_CATEGORY.get(cat_key) is None:
                continue
            regions.append({
                "category": det["category"],
                "bbox": det["bbox"],
                "score": det["score"],
                "text": state["crop_texts"].get(det_idx, ""),
            })

        # Aggregate timing across all crop sequences
        all_stats = state["crop_stats"].values()
        total_input = sum(s["input_tokens"] for s in all_stats)
        total_output = sum(s["output_tokens"] for s in all_stats)
        total_decode_steps = sum(s["decode_steps"] for s in all_stats)
        total_decode_batch_sum = sum(s["decode_batch_sum"] for s in all_stats)

        full_text = "\n\n".join(r["text"] for r in regions if r["text"])
        phase_ms = (sum(s["tokenize_ms"] for s in all_stats)
                    + sum(s["prefill_ms"] for s in all_stats)
                    + sum(s["decode_wall_ms"] for s in all_stats)
                    + sum(s["finalize_ms"] for s in all_stats))
        total_wall_ms = (time.monotonic() - state["enqueue_time"]) * 1000
        result = InferenceResult(
            text=full_text,
            masks_rle=[],
            bboxes_raw=[],
            image_size=state["image_size"],
            input_tokens=total_input,
            output_tokens=total_output,
            inference_time_ms=phase_ms,
            queue_ms=max(0.0, total_wall_ms - phase_ms),
            total_time_ms=total_wall_ms,
            tokenize_time_ms=sum(s["tokenize_ms"] for s in all_stats),
            prefill_time_ms=sum(s["prefill_ms"] for s in all_stats),
            decode_time_ms=sum(s["decode_wall_ms"] for s in all_stats),
            finalize_time_ms=sum(s["finalize_ms"] for s in all_stats),
            num_decode_steps=total_decode_steps,
            avg_decode_batch_size=(
                total_decode_batch_sum / total_decode_steps
                if total_decode_steps > 0 else 0.0
            ),
            prefill_tokens=sum(s["prefill_tokens"] for s in all_stats),
            num_preemptions=sum(s["preemptions"] for s in all_stats),
            layout_regions=regions,
        )
        response_queue.put(
            WorkerResponse(request_id=parent_id, gpu_id=gpu_id, result=result)
        )
    except Exception:
        log.exception("Error assembling layout result for request %d", parent_id)
        response_queue.put(
            WorkerResponse(
                request_id=parent_id, gpu_id=gpu_id,
                error="Failed to assemble layout result",
            )
        )
    finally:
        engine._compound_state.pop(parent_id, None)


def _fail_all_pending(engine, response_queue, gpu_id, error_msg):
    """Fail every in-flight sequence with an error response."""
    all_seqs = list(engine.waiting) + list(engine.running)
    failed_parents = set()
    for seq in all_seqs:
        if seq.batch_idx is not None:
            try:
                engine._input_pos[seq.batch_idx] = 0
                engine._rope_pos_t[seq.batch_idx] = 0
                engine.paged_kv_cache.erase(seq.batch_idx)
            except Exception:
                pass
        # Fail compound parent requests
        parent_id = getattr(seq, "_compound_parent_id", None)
        if parent_id is not None and parent_id not in failed_parents:
            failed_parents.add(parent_id)
            response_queue.put(
                WorkerResponse(request_id=parent_id, gpu_id=gpu_id, error=error_msg)
            )
            continue
        request_id = getattr(seq, "_api_request_id", None)
        if request_id is not None:
            response_queue.put(
                WorkerResponse(
                    request_id=request_id, gpu_id=gpu_id, error=error_msg
                )
            )
    engine.waiting.clear()
    engine.running.clear()
    engine._compound_state.clear()


# Main-process side


class WorkerProxy:
    """Main-process handle for a GPU worker subprocess."""

    def __init__(
        self, gpu_id: int, config: ServerConfig, response_queue: mp.Queue
    ):
        self.gpu_id = gpu_id
        self._config = config
        self._request_queue: mp.Queue = mp.Queue()
        self._response_queue = response_queue
        self._ready_event = mp.Event()
        self._failed_value = mp.Value("b", False)
        self._is_ocr_value = mp.Value("b", False)
        self._process: mp.Process | None = None

        self._pending: dict[
            int, tuple[asyncio.Future, asyncio.AbstractEventLoop]
        ] = {}
        self._pending_lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self):
        self._process = mp.Process(
            target=_worker_process_main,
            args=(
                self.gpu_id,
                self._config,
                self._request_queue,
                self._response_queue,
                self._ready_event,
                self._failed_value,
                self._is_ocr_value,
            ),
            name=f"engine-gpu{self.gpu_id}",
            daemon=True,
        )
        self._process.start()

    def stop(self):
        try:
            self._request_queue.put_nowait(None)
        except Exception:
            pass
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()

    def wait_ready(self, timeout: float = 600) -> bool:
        return self._ready_event.wait(timeout)

    # ── State queries ─────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._ready_event.is_set() and not self.failed

    @property
    def failed(self) -> bool:
        if self._failed_value.value:
            return True
        if (
            self._process is not None
            and not self._process.is_alive()
            and self._ready_event.is_set()
        ):
            return True
        return False

    @property
    def load(self) -> int:
        if not self.ready:
            return 999_999
        with self._pending_lock:
            return len(self._pending)

    @property
    def is_ocr(self) -> bool:
        return bool(self._is_ocr_value.value)

    def get_status(self) -> dict:
        if self.failed:
            return {"gpu_id": self.gpu_id, "status": "failed"}
        if not self.ready:
            return {"gpu_id": self.gpu_id, "status": "loading"}
        with self._pending_lock:
            n_pending = len(self._pending)
        return {
            "gpu_id": self.gpu_id,
            "status": "ready",
            "waiting": n_pending,
            "running": 0,
        }

    # ── Request submission ────────────────────────────────────────────

    def submit(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        max_tokens: int,
        min_image_size: int,
        max_image_size: int,
        task: str = "segmentation",
        future: asyncio.Future,
        loop: asyncio.AbstractEventLoop,
        request_idx: int,
    ):
        req = WorkerRequest(
            request_id=request_idx,
            prompt=prompt,
            image_bytes=image_bytes,
            max_tokens=max_tokens,
            min_image_size=min_image_size,
            max_image_size=max_image_size,
            task=task,
        )
        with self._pending_lock:
            self._pending[request_idx] = (future, loop)
        self._request_queue.put(req)

    # ── Response resolution (called by pool collector thread) ─────────

    def resolve(self, response: WorkerResponse):
        with self._pending_lock:
            entry = self._pending.pop(response.request_id, None)
        if entry is None:
            return
        future, loop = entry
        if response.error:
            try:
                loop.call_soon_threadsafe(
                    future.set_exception, RuntimeError(response.error)
                )
            except RuntimeError:
                pass
        else:
            try:
                loop.call_soon_threadsafe(future.set_result, response.result)
            except RuntimeError:
                pass


# ── Pool ───────────────────────────────────────────────────────────────


class WorkerPool:
    """Manages GPU worker subprocesses with least-load dispatch."""

    def __init__(self, config: ServerConfig):
        num_gpus = config.num_gpus
        if num_gpus <= 0:
            num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available")

        self._response_queue: mp.Queue = mp.Queue()
        self.workers: list[WorkerProxy] = [
            WorkerProxy(i, config, self._response_queue) for i in range(num_gpus)
        ]
        self._gpu_to_worker = {w.gpu_id: w for w in self.workers}
        self._collector_thread: threading.Thread | None = None
        self._running = False

    def start_all(self):
        self._running = True
        for w in self.workers:
            w.start()
        self._collector_thread = threading.Thread(
            target=self._collect_responses,
            name="result-collector",
            daemon=True,
        )
        self._collector_thread.start()

    def _collect_responses(self):
        """Drain the shared response queue and resolve asyncio futures."""
        while self._running:
            try:
                resp: WorkerResponse = self._response_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            proxy = self._gpu_to_worker.get(resp.gpu_id)
            if proxy is not None:
                proxy.resolve(resp)

    def wait_ready(self, timeout: float = 600):
        for w in self.workers:
            if not w.wait_ready(timeout):
                logger.warning(
                    "GPU %d did not initialize within %ds", w.gpu_id, timeout
                )

        healthy = self.healthy_workers
        if not healthy:
            raise RuntimeError("All GPU workers failed during initialization")

        failed = [w for w in self.workers if w.failed]
        if failed:
            ids = ", ".join(str(w.gpu_id) for w in failed)
            logger.warning(
                "Workers on GPU(s) %s failed; continuing with %d healthy worker(s)",
                ids,
                len(healthy),
            )

    def stop_all(self):
        self._running = False
        for w in self.workers:
            w.stop()
        if self._collector_thread:
            self._collector_thread.join(timeout=2)

    @property
    def healthy_workers(self) -> list[WorkerProxy]:
        return [w for w in self.workers if w.ready and not w.failed]

    def select_worker(self) -> WorkerProxy:
        """Return the healthy worker with the lowest current load."""
        candidates = self.healthy_workers
        if not candidates:
            raise RuntimeError("No healthy GPU workers available")
        return min(candidates, key=lambda w: w.load)

    @property
    def any_ready(self) -> bool:
        return any(w.ready and not w.failed for w in self.workers)

    @property
    def all_ready(self) -> bool:
        return all(w.ready and not w.failed for w in self.workers)
