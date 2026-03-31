# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

import hashlib
import logging
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from tqdm import tqdm

from falcon_perception.aux_output import AuxOutput
from falcon_perception.data import (
    get_pos_thw_single,
    load_image,
    resize_image_if_necessary,
    tokenize_inputs,
)
from falcon_perception.model import FalconPerception, ImgScatterEntry
from falcon_perception.paged_attention import PagedKVCache
from falcon_perception.sampling import sample_token

logger = logging.getLogger(__name__)


# ── Timing / OOM helpers ─────────────────────────────────────────────

class _Timer:
    """Context-manager that records wall-clock milliseconds in ``.ms``."""
    __slots__ = ("ms", "_t0")

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._t0) * 1000.0
        return False


@contextmanager
def _oom_guard(device, stage: str, **ctx):
    """Catch ``torch.cuda.OutOfMemoryError`` and re-raise with GPU memory diagnostics."""
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        alloc_gb = torch.cuda.memory_allocated(device) / 1e9
        resv_gb = torch.cuda.memory_reserved(device) / 1e9
        detail = ", ".join(f"{k}={v}" for k, v in ctx.items())
        raise RuntimeError(
            f"CUDA OOM during {stage}: {detail}, "
            f"alloc={alloc_gb:.2f}GB, reserved={resv_gb:.2f}GB"
        ) from e


@dataclass
class SamplingParams:
    max_new_tokens: int = -1
    stop_token_ids: list[int] | None = None
    segmentation_threshold: float = 0.3
    coord_dedup_threshold: float = 0.00
    hr_upsample_ratio: int = 8


@dataclass
class SequenceStats:
    """Per-request timing and scheduling stats, populated during inference."""
    tokenize_ms: float = 0.0
    prefill_ms: float = 0.0
    prefill_batch_size: int = 0
    prefill_tokens: int = 0    # total tokens fed into prefill (> input_length if preempted)
    decode_wall_ms: float = 0.0
    decode_steps: int = 0      # also == decode tokens generated (1 per step)
    decode_batch_sum: int = 0
    finalize_ms: float = 0.0
    preemptions: int = 0


class Sequence:
    def __init__(
        self,
        text: str,
        image: Image.Image | str | None,
        min_image_size: int = 256,
        max_image_size: int = 1024,
        request_idx: int = 0,
        task: str = "segmentation",
    ):
        # ── Request identity ──
        self.request_idx = request_idx
        self.task = task
        self.finished = False
        self.sampling_params: SamplingParams | None = None
        self.stats = SequenceStats()

        # ── Input: text / tokens (set during tokenization) ──
        self.text = text
        self.input_ids: Tensor = None
        self.rope_pos_t: Tensor | None = None
        self.rope_pos_hw: Tensor | None = None
        # ── Input: image ──
        self._image_raw = image
        self._image_pil = image if isinstance(image, Image.Image) else None
        self.original_image_size: tuple[int, int] | None = None
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.image_tensor: Tensor | None = None
        self.image_hash: int | None = None

        # ── Runtime state (set by scheduler / decode) ──
        self.batch_idx: int = None
        self.pos_t: Tensor = None
        self.current_bbox_xy: Tensor | None = None
        self.current_bbox_hw: Tensor | None = None
        self.hr_image_features: Tensor | None = None
        self._hr_cache_hit: bool = False

        # ── Output ──
        self._output_ids = []
        self._output_logits = []
        self._output_probs = []
        self.output_aux = AuxOutput()

    def add_next_token(
        self,
        token_id: Tensor,   # () int64 GPU scalar
        logits: Tensor,      # () float32 GPU scalar
        probs: Tensor,       # () float32 GPU scalar
        xy: Tensor,          # (2,) float32 GPU
        hw: Tensor,          # (2,) float32 GPU
        is_coord: Tensor,    # () bool GPU
        is_size: Tensor,     # () bool GPU
        segm: Tensor,        # (segm_dim,) model-dtype GPU
        is_segm: Tensor,     # () bool GPU
        pos_t: Tensor | None,
    ):
        """Record one generated token.  All GPU — no host-device sync."""
        self._output_ids.append(token_id)
        self._output_logits.append(logits)
        self._output_probs.append(probs)
        if pos_t is not None:
            self.pos_t = pos_t + 1
        else:
            assert self.pos_t is not None
            self.pos_t += 1

        self.current_bbox_xy = xy
        self.current_bbox_hw = hw
        self.output_aux.append_bbox(xy, hw, is_coord, is_size)

        if segm.shape[0] > 0:
            self.output_aux.append_segm(segm, is_segm)

    def copy(self):
        return Sequence(self.text, self._image_raw, self.min_image_size, self.max_image_size, task=self.task)

    @property
    def output_ids(self):
        return torch.tensor(self._output_ids, dtype=torch.int64)

    @property
    def output_logits(self):
        return torch.tensor(self._output_logits, dtype=torch.float32)

    @property
    def output_probs(self):
        return torch.tensor(self._output_probs, dtype=torch.float32)

    @property
    def input_length(self):
        assert self.input_ids is not None, "input_ids has not been initialized"
        return len(self.input_ids)

    @property
    def output_length(self):
        return len(self._output_ids)

    @property
    def total_length(self):
        return self.input_length + self.output_length

    @property
    def total_token_ids(self):
        assert self.input_ids is not None, "input text has not been tokenized to input_ids"
        if self.output_length > 0:
            return torch.cat([self.input_ids, self.output_ids], dim=0)
        return self.input_ids

    @property
    def last_token_id(self):
        return self._output_ids[-1]

    @property
    def pil_image(self) -> Image.Image | None:
        """Get PIL image, loading from path if necessary (cached)."""
        if self._image_pil is not None:
            return self._image_pil
        if self._image_raw:
            self._image_pil = load_image(self._image_raw)
            return self._image_pil
        return None


def process_sampling_params(
    sequences: list[Sequence],
    sampling_params: SamplingParams | list[SamplingParams] | None,
    tokenizer,
):
    if sampling_params is None:
        sampling_params = SamplingParams(stop_token_ids=[tokenizer.eos_token_id])
    if isinstance(sampling_params, SamplingParams):
        sampling_params = [sampling_params] * len(sequences)

    assert len(sampling_params) == len(sequences), (
        "sampling_params must be a list of the same length as sequences"
    )

    for seq, param in zip(sequences, sampling_params):
        seq.sampling_params = param


# Preset engine configs keyed by VRAM tier (GiB).
# Pick the first tier whose threshold <= your GPU.
# Tiers are searched large-to-small so the biggest match wins.
_GPU_PRESETS: list[tuple[int, dict]] = [
    (79, dict(n_pages=1024,  page_size=128, max_batch_size=64, prefill_length_limit=32768, max_hr_cache_entries=256)),
    (47, dict(n_pages=512,  page_size=128, max_batch_size=32, prefill_length_limit=16384, max_hr_cache_entries=128)),
    (23, dict(n_pages=256,  page_size=128, max_batch_size=16,  prefill_length_limit=8192,  max_hr_cache_entries=64)),
    (15, dict(n_pages=128,  page_size=128, max_batch_size=8,  prefill_length_limit=8192,  max_hr_cache_entries=32)),
]


def engine_config_for_gpu(
    max_image_size: int = 1024,
    device: torch.device | str = "cuda",
    gpu_memory_gb: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Return a preset ``PagedInferenceEngine`` config for the current GPU.

    Looks up the GPU's total VRAM (or *gpu_memory_gb* if given) and returns
    the recommended preset from ``_GPU_PRESETS``.  Presets are calibrated for
    float32 KV cache; for half-precision dtypes (bfloat16 / float16) the KV
    cache is 2x smaller, so ``n_pages`` and ``max_batch_size`` are doubled.

    Returns
    -------
    dict
        Kwargs suitable for ``PagedInferenceEngine(**cfg, ...)``.
    """
    dev = torch.device(device)
    if gpu_memory_gb is None:
        props = torch.cuda.get_device_properties(dev)
        gpu_memory_gb = props.total_memory / 1024**3

    kv_scale = 2 if dtype in (torch.bfloat16, torch.float16) else 1

    for threshold_gb, preset in _GPU_PRESETS:
        if gpu_memory_gb >= threshold_gb:
            cfg = {**preset, "max_image_size": max_image_size}
            cfg["n_pages"] *= kv_scale
            cfg["max_batch_size"] *= kv_scale
            logger.info(
                "engine_config_for_gpu: %.1f GB VRAM → %d GB tier, dtype=%s "
                "(n_pages=%d, max_batch=%d, prefill_limit=%d)",
                gpu_memory_gb, threshold_gb, dtype,
                cfg["n_pages"], cfg["max_batch_size"], cfg["prefill_length_limit"],
            )
            return cfg

    # Fallback: smallest preset
    cfg = {**_GPU_PRESETS[-1][1], "max_image_size": max_image_size}
    cfg["n_pages"] *= kv_scale
    cfg["max_batch_size"] *= kv_scale
    logger.warning(
        "engine_config_for_gpu: %.1f GB VRAM below all presets, using smallest.",
        gpu_memory_gb,
    )
    return cfg


class PagedInferenceEngine:
    def __init__(
        self,
        model: FalconPerception,
        tokenizer,
        image_processor,
        max_batch_size,
        max_seq_length,
        n_pages,
        page_size=128,
        prefill_length_limit=-1,  # No limit
        kernel_options=None,
        seed: int | None = None,
        enable_hr_cache: bool = True,
        max_hr_cache_entries: int = 100,
        max_image_size: int = 1024,
        capture_cudagraph: bool = True,
        max_decode_steps_between_prefills: int = 16,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.image_processor = image_processor
        assert max_seq_length % page_size == 0, "max_seq_length must be divisible by page_size"
        self.max_seq_length = max_seq_length
        assert max_batch_size > 1, "max_batch_size must be greater than 1"
        self.max_batch_size = max_batch_size
        self.kernel_options = kernel_options or {}
        self.prefill_length_limit = prefill_length_limit  # NOTE: control the peak memory usage of prefill

        self.device = model.device
        self.paged_kv_cache = PagedKVCache(
            n_pages=n_pages,
            page_size=page_size,
            max_batch_size=max_batch_size,
            n_heads=self.model.args.n_heads,
            head_dim=self.model.args.head_dim,
            num_layers=self.model.args.n_layers,
            dtype=model.dtype,
            device=self.device,
        )
        _bytes_per_elem = 2 if model.dtype in (torch.bfloat16, torch.float16) else 4
        _kv_gb = (
            self.model.args.n_layers * 2 * self.model.args.n_heads
            * n_pages * page_size * self.model.args.head_dim
            * _bytes_per_elem
        ) / 1024**3
        logger.info("KV cache: %d pages × %d = %d tokens, %.2f GiB",
                     n_pages, page_size, n_pages * page_size, _kv_gb)
        # Causal decoding block mask in the logical space, hence the max logical seq length
        self.block_mask = self.paged_kv_cache.create_causal_blockmask(B=self.max_batch_size, L=self.max_seq_length)
        self.rng = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        # --- Persistent position buffers (indexed by batch slot, like flex-nano-vllm) ---
        # Updated in-place by prefill_sequences and decode_step so that
        # CUDA-graph'd decode can read them 
        self._input_pos = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )
        # RoPE 1-D position (may differ from input_pos when images are present)
        self._rope_pos_t = torch.zeros(
            self.max_batch_size, dtype=torch.int64, device=self.device
        )

        # Pinned staging buffers for the decode-setup copies.  Writing values
        # here is a pure CPU memset (no CUDA calls) and the subsequent bulk
        # copy_(..., non_blocking=True) from pinned memory is a single
        # cudaMemcpyAsync that never triggers stream synchronisation.
        self._decode_bidx_pinned = torch.zeros(max_batch_size, dtype=torch.int64, pin_memory=True)
        self._decode_pos_pinned = torch.zeros(max_batch_size, dtype=torch.int32, pin_memory=True)
        self._decode_gv_bidx_pinned = torch.zeros(max_batch_size, dtype=torch.int64, pin_memory=True)
        # Matching GPU scratch for scatter_ indices/values
        self._decode_bidx_gpu = torch.zeros(max_batch_size, dtype=torch.int64, device=self.device)
        self._decode_pos_gpu = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

        # --- CUDA graph state (initialised by capture_decode_cudagraph) ---
        self.cudagraph_captured = False
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_bs: list[int] = []
        self._graph_pool: int | None = None
        self._graph_vars: dict[str, torch.Tensor] = {}

        # Background tokenization thread for overlapping CPU image/text
        # preprocessing with GPU compute.  All heavy ops in _tokenize_single
        # (PIL, numpy, Rust tokenizer) release the GIL, giving true parallelism.
        self._prefetch_workers = 8
        self._tokenize_executor = ThreadPoolExecutor(max_workers=self._prefetch_workers)
        self._prefetch_futures: dict[int, Future] = {}  # id(seq) -> Future

        # --- HR image features cache (LRU, CPU pinned memory pool) ---
        # Each cache entry lives in its own pre-allocated pinned buffer, so
        # both D2H (miss) and H2D (hit) are truly async with non_blocking=True
        # on _hr_transfer_stream — zero runtime cudaHostAlloc, zero pageable
        # memcpy, and DMA fully overlaps with GPU compute on the default stream.
        self.enable_hr_cache = enable_hr_cache
        self._max_hr_cache_entries = max_hr_cache_entries
        # cache: image_hash → (pool_buffer, D, h, w)
        self._hr_features_cache: OrderedDict[int, tuple[Tensor, int, int, int]] = OrderedDict()
        self._hr_transfer_stream = torch.cuda.Stream(self.device)

        # Pinned buffer for async D2H of token IDs.  The copy is enqueued
        # on the default stream right after sampling, and an event is recorded
        # immediately after.  _check_done syncs only that event — not the full
        # stream — so it doesn't wait for the upsampler that runs later.
        self._token_ids_pinned = torch.empty(
            max_batch_size, dtype=torch.long, pin_memory=True
        )
        self._token_ids_ready: torch.cuda.Event | None = None
        self._token_ids_count: int = 0
        if enable_hr_cache and hasattr(model, "itok_upsampler"):
            _feat_dim = model.args.segm_out_dim
            _max_hw = max_image_size
            _bytes_per = model.dtype.itemsize
            _buf_bytes = _feat_dim * _max_hw * _max_hw * _bytes_per
            _pool_mb = max_hr_cache_entries * _buf_bytes / (1024 ** 2)
            logger.info("Pre-allocating %d pinned buffers (%d×%d×%d %s, %.0f MiB total)",
                        max_hr_cache_entries, _feat_dim, _max_hw, _max_hw, model.dtype, _pool_mb)
            self._hr_pinned_pool: list[Tensor] = [
                torch.empty(_feat_dim, _max_hw, _max_hw, dtype=model.dtype, pin_memory=True)
                for _ in range(max_hr_cache_entries)
            ]
            logger.info("Pinned pool ready.")
        else:
            self._hr_pinned_pool = []

        # Scheduling state — initialised here so the engine is usable without
        # calling generate() (e.g. the server's continuous-batching loop pushes
        # sequences directly into self.waiting and calls run_one_step).
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.done: deque[Sequence] = deque()
        self.temperature: float = 0.0
        self.top_k: int | None = None

        # Scheduler: how many consecutive decode steps before allowing a prefill.
        # 0 = always prefill first (legacy behaviour).
        # 16 (default) = stable decode runs, periodic prefill admission.
        # Large = decode-first, only prefill when running queue is empty.
        self.max_decode_steps_between_prefills = max_decode_steps_between_prefills
        self._steps_since_prefill: int = 0
        self._decode_run_lengths: list[int] = []
        self._prefill_batch_sizes: list[int] = []
        self._prefill_token_counts: list[int] = []

        if capture_cudagraph:
            self.capture_decode_cudagraph()

    # ── HR image features cache helpers ──────────────────────────────────

    def _hr_cache_start_loads(
        self, sequences: list[Sequence]
    ) -> dict | None:
        """Check HR feature cache and start async H2D for cache hits.

        Called BEFORE the transformer forward pass so the DMA overlaps with
        GPU compute on the default stream.

        Returns a context dict consumed by ``_hr_cache_resolve``, or None
        when segmentation is disabled / no segmentation sequences exist.
        """
        if not self.model.args.perception_heads:
            return None

        seg_seqs = [
            s for s in sequences
            if s.image_tensor is not None and s.task == "segmentation"
        ]
        if not seg_seqs:
            return None

        gpu_futures: dict[int, Tensor] = {}  # id(seq) → in-flight GPU tensor
        if self.enable_hr_cache:
            hits: list[tuple[Sequence, Tensor, int, int, int]] = []
            for seq in seg_seqs:
                if seq.image_hash is not None and seq.image_hash in self._hr_features_cache:
                    self._hr_features_cache.move_to_end(seq.image_hash)
                    buf, D, h, w = self._hr_features_cache[seq.image_hash]
                    hits.append((seq, buf, D, h, w))
                    seq._hr_cache_hit = True

            if hits:
                # One contiguous GPU allocation for all hits, then async
                # pinned→GPU copies on the transfer stream.
                sizes = [D * h * w for _, _, D, h, w in hits]
                staging = torch.empty(sum(sizes), device=self.device, dtype=hits[0][1].dtype)
                off = 0
                for (seq, buf, D, h, w), n in zip(hits, sizes):
                    gpu_futures[id(seq)] = staging[off:off + n].view(D, h, w)
                    off += n
                with torch.cuda.stream(self._hr_transfer_stream):
                    for seq, buf, D, h, w in hits:
                        pinned_src = buf.view(-1)[:D * h * w]
                        gpu_futures[id(seq)].view(-1).copy_(pinned_src, non_blocking=True)

        all_cached = bool(seg_seqs) and all(s._hr_cache_hit for s in seg_seqs)

        h2d_event: torch.cuda.Event | None = None
        if gpu_futures:
            h2d_event = self._hr_transfer_stream.record_event()

        return {
            "seg_seqs": seg_seqs,
            "gpu_futures": gpu_futures,
            "h2d_event": h2d_event,
            "all_cached": all_cached,
        }

    def _hr_cache_store(self, seq: Sequence) -> None:
        """Copy HR features GPU→pinned on ``_hr_transfer_stream``."""
        if not self.enable_hr_cache or seq.image_hash is None:
            return

        if len(self._hr_features_cache) >= self._max_hr_cache_entries:
            _, (evicted_buf, _, _, _) = self._hr_features_cache.popitem(last=False)
            self._hr_pinned_pool.append(evicted_buf)
        if seq.image_hash in self._hr_features_cache:
            old_buf, _, _, _ = self._hr_features_cache.pop(seq.image_hash)
            self._hr_pinned_pool.append(old_buf)

        if not self._hr_pinned_pool:
            return

        D, h, w = seq.hr_image_features.shape
        buf = self._hr_pinned_pool.pop()
        numel = D * h * w
        pinned_flat = buf.view(-1)[:numel]
        gpu_flat = seq.hr_image_features.contiguous().view(-1)

        d2h_ready = torch.cuda.current_stream().record_event()
        with torch.cuda.stream(self._hr_transfer_stream):
            self._hr_transfer_stream.wait_event(d2h_ready)
            pinned_flat.copy_(gpu_flat, non_blocking=True)
            gpu_flat.record_stream(self._hr_transfer_stream)

        self._hr_features_cache[seq.image_hash] = (buf, D, h, w)

    def _stage_token_ids_d2h(self, next_token: Tensor) -> None:
        """Copy token IDs to pinned memory and record an event on the default stream.

        Must be called right after sampling, BEFORE any expensive work
        (upsampler) so that ``_check_done`` can sync only the event — not
        the full stream — and avoid waiting for the upsampler.
        """
        B = next_token.shape[0]
        self._token_ids_pinned[:B].copy_(next_token[:B], non_blocking=True)
        self._token_ids_ready = torch.cuda.current_stream().record_event()
        self._token_ids_count = B

    def _hr_cache_resolve(
        self,
        hr_ctx: dict | None,
        sequences: list[Sequence],
        h_BSD: Tensor,
        pixel_values_list: list[Tensor],
        seq_offsets: dict[int, tuple[int, int]],
    ) -> None:
        """Run upsampler for cache misses, assign cache hits, store new entries.

        Called AFTER the transformer forward pass.
        """
        if hr_ctx is None:
            return

        seg_seqs: list[Sequence] = hr_ctx["seg_seqs"]
        gpu_futures: dict[int, Tensor] = hr_ctx["gpu_futures"]
        h2d_event: torch.cuda.Event | None = hr_ctx["h2d_event"]
        all_cached: bool = hr_ctx["all_cached"]

        # ── Upsampler for cache misses (one image at a time to bound memory ) ──
        if not all_cached:
            ps = self.model.args.spatial_patch_size
            all_seq_w_img = [s for s in sequences if s.image_tensor is not None]
            for img_idx, seq in enumerate(all_seq_w_img):
                if seq.task != "segmentation" or seq._hr_cache_hit:
                    continue

                ratio = seq.sampling_params.hr_upsample_ratio
                start, end = seq_offsets[id(seq)]
                seq_h = h_BSD[0, start:end]  # (S, D)

                img_positions = (seq.input_ids == self.model.args.img_id).nonzero(as_tuple=True)[0]
                img_token_start = int(img_positions[0])
                h_valid = seq.image_tensor.shape[1] // ps
                w_valid = seq.image_tensor.shape[2] // ps

                # Re-pad to square max_image_size for AnyUp training consistency.
                pv = pixel_values_list[img_idx]  # (T, H_native, W_native, C)
                target = ((seq.max_image_size + ps - 1) // ps) * ps
                _, h_cur, w_cur, _ = pv.shape
                if h_cur < target or w_cur < target:
                    pv = F.pad(pv, (0, 0, 0, target - w_cur, 0, target - h_cur))

                target_patches = target // ps
                output_size = (target_patches * ratio, target_patches * ratio)

                hr_feat = self.model.upsample_single_img_features(
                    seq_h,
                    pv,
                    img_token_start=img_token_start,
                    h_valid=h_valid,
                    w_valid=w_valid,
                    output_size=output_size,
                )

                # Crop to original (unpadded) pixel dimensions.
                h_px = (seq.image_tensor.shape[1] // ps) * ratio
                w_px = (seq.image_tensor.shape[2] // ps) * ratio
                seq.hr_image_features = hr_feat[:, :h_px, :w_px].contiguous()
                self._hr_cache_store(seq)

        # ── Assign cache hits & return pool slots ──
        if gpu_futures:
            if h2d_event is not None:
                torch.cuda.current_stream().wait_event(h2d_event)
            for seq in seg_seqs:
                if seq._hr_cache_hit:
                    seq.hr_image_features = gpu_futures[id(seq)]

    def prefill_sequences(self, sequences: list[Sequence]):
        # NOTE: there are two kind of positional indices
        # 1. `input_pos` which count the token indices in the sequence so far
        # 2. `input_thw` which is the one used for 1+2D golden gate rope
        BLOCK = 128
        pad_id = int(self.tokenizer.pad_token_id or 0)

        # Per-image: pad to patch-size-aligned and transfer to GPU individually.
        # No batch_images_with_mask — every consumer loops per-image anyway,
        # so batching only wasted memory/compute by padding to max_image_size.
        ps = self.model.args.spatial_patch_size
        pixel_values_list: list[Tensor] = []
        for seq in sequences:
            if seq.image_tensor is not None:
                img = seq.image_tensor  # numpy or Tensor, (T, H, W, C)
                if not isinstance(img, Tensor):
                    img = torch.from_numpy(img)
                _, h, w, _ = img.shape
                h_pad = (ps - h % ps) % ps
                w_pad = (ps - w % ps) % ps
                if h_pad or w_pad:
                    img = F.pad(img, (0, 0, 0, w_pad, 0, h_pad))
                pixel_values_list.append(
                    img.pin_memory().to(self.device, non_blocking=True)
                )

        # IMPORTANT: Build per-sequence tensors, each padded to FlexAttention BLOCK
        # boundary so no block ever spans two documents.  This eliminates
        # bfloat16 non-determinism from mixed-document softmax blocks.
        padded_ids, padded_pos, padded_bidx = [], [], []
        padded_rt, padded_rhw = [], []
        aligned_offsets: list[int] = []
        offset = 0
        for seq in sequences:
            L = seq.total_length
            gap = (BLOCK - L % BLOCK) % BLOCK

            ids = seq.total_token_ids
            pos = torch.arange(L, dtype=torch.long)
            bidx = torch.full((L,), seq.batch_idx, dtype=torch.int64)

            assert seq.rope_pos_t is not None and seq.rope_pos_hw is not None
            if seq.output_length == 0:
                rt = seq.rope_pos_t
                rhw = seq.rope_pos_hw
            else:
                last_t = seq.rope_pos_t[-1]
                extra_t = last_t + torch.arange(1, seq.output_length + 1, dtype=torch.long)
                rt = torch.cat([seq.rope_pos_t, extra_t])
                rhw = torch.cat([seq.rope_pos_hw, torch.zeros(seq.output_length, 2)])

            if gap:
                ids = F.pad(ids, (0, gap), value=pad_id)
                pos = F.pad(pos, (0, gap), value=0)
                bidx = F.pad(bidx, (0, gap), value=0)       # batch_idx=0 is reserved (no-op)
                rt = F.pad(rt, (0, gap), value=0)
                rhw = F.pad(rhw, (0, 0, 0, gap), value=0.0)

            padded_ids.append(ids)
            padded_pos.append(pos)
            padded_bidx.append(bidx)
            padded_rt.append(rt)
            padded_rhw.append(rhw)
            aligned_offsets.append(offset)
            offset += L + gap

        input_ids = torch.cat(padded_ids).unsqueeze(0)
        input_pos = torch.cat(padded_pos).unsqueeze(0)
        batch_idx = torch.cat(padded_bidx).unsqueeze(0)
        rope_pos_t = torch.cat(padded_rt).unsqueeze(0)
        rope_pos_hw = torch.cat(padded_rhw).unsqueeze(0)

        input_ids = input_ids.to(self.device, non_blocking=True)  # B=1,S
        input_pos = input_pos.to(self.device, non_blocking=True)  # B=1,S
        batch_idx = batch_idx.to(self.device, non_blocking=True)  # B=1,S
        rope_pos_t = rope_pos_t.to(self.device, non_blocking=True)
        rope_pos_hw = rope_pos_hw.to(self.device, non_blocking=True)
        last_pos_idx = torch.tensor(
            [aligned_offsets[i] + sequences[i].total_length - 1 for i in range(len(sequences))],
            dtype=torch.int32, device=self.device,
        )
        bboxes = [seq.output_aux.materialize_bboxes() for seq in sequences]
        all_xy, all_hw = self.model._extract_coords(bboxes)
        all_xy = all_xy.to(device=self.device, dtype=self.model.dtype)
        all_hw = all_hw.to(device=self.device, dtype=self.model.dtype)

        mask = self.paged_kv_cache.create_prefill_blockmask(
            batch_idx,
            input_ids,
            self.tokenizer.image_cls_token_id,
            self.tokenizer.end_of_image_token_id,
        )

        # ── HR cache: kick off async H2D for cache hits (overlaps with forward) ──
        hr_ctx = self._hr_cache_start_loads(sequences)

        # Pre-compute image scatter metadata on CPU so
        # _scatter_img_tokens_with_projector can use slice indexing only
        # (no boolean index / masked_scatter → no nonzero → no GPU sync).
        # Valid dims come from the original (unpadded) image tensor shape,
        # avoiding expensive reductions on pinned-memory masks.
        img_scatter_info: list[ImgScatterEntry] = []
        for i, seq in enumerate(sequences):
            if seq.image_tensor is not None:
                img_positions = (seq.input_ids == self.model.args.img_id).nonzero(as_tuple=True)[0]
                tok_start = aligned_offsets[i] + int(img_positions[0])
                n_tok = len(img_positions)
                h_valid = seq.image_tensor.shape[1] // ps
                w_valid = seq.image_tensor.shape[2] // ps
                img_scatter_info.append(ImgScatterEntry(0, tok_start, n_tok, h_valid, w_valid))

        logits_BSV, h_BSD = self.model.forward(
            tokens=input_ids,
            attention_mask=mask,
            kv_cache=self.paged_kv_cache,
            rope_pos_t=rope_pos_t,
            rope_pos_hw=rope_pos_hw,
            pixel_values=pixel_values_list or None,
            coord_xy=all_xy,
            size_hw=all_hw,
            input_pos=input_pos,
            batch_idx=batch_idx,
            flex_attn_kernel_options=self.kernel_options
            | {"FORCE_USE_FLEX_ATTENTION": True},
            img_scatter_info=img_scatter_info or None,
        )

        last_logits_BV = logits_BSV[0, last_pos_idx, :]
        last_hidden_BD = h_BSD[0, last_pos_idx, :]
        last_rope_pos_t = rope_pos_t[0, last_pos_idx]  # Store pos_t at 0th dim of last idx
        next_token, logits, probs = sample_token(
            last_logits_BV,
            rng=self.rng,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        next_token = next_token.squeeze(-1)   # (B,) int64 GPU
        logits = logits.squeeze(-1)           # (B,) float32 GPU
        probs = probs.squeeze(-1)             # (B,) float32 GPU

        xy_B2, hw_B2, is_coord_B, is_size_B, _coord_logits = self.model.sample_bbox(last_hidden_BD, next_token)
        segm_BD, is_segm_B = self.model.get_segm_tokens(last_hidden_BD, next_token)
        for i in range(len(sequences)):
            sequences[i].add_next_token(
                token_id=next_token[i],
                logits=logits[i],
                probs=probs[i],
                xy=xy_B2[i],
                hw=hw_B2[i],
                is_coord=is_coord_B[i],
                is_size=is_size_B[i],
                segm=segm_BD[i],
                is_segm=is_segm_B[i],
                pos_t=last_rope_pos_t[i],
            )

        # Stage token-ID D2H + event BEFORE the upsampler so
        # _check_done can sync only the event, not the full stream.
        self._stage_token_ids_d2h(next_token)

        # ── HR cache: upsampler for misses + assign hits ──
        # Per-sequence token offsets from the block-aligned packed layout
        # so _hr_cache_resolve can use slice indexing (zero-copy views).
        seq_offsets = {
            id(seq): (aligned_offsets[i], aligned_offsets[i] + seq.total_length)
            for i, seq in enumerate(sequences)
        }

        self._hr_cache_resolve(
            hr_ctx, sequences, h_BSD,
            pixel_values_list,
            seq_offsets,
        )


    def get_decoding_block_mask(self, batch_idx: Tensor, input_pos: Tensor):
        """
        Args:
            batch_idx: [B]
            input_pos: [B]
        Returns:
            block_mask: [B, H, ROWS=1, MAX_BLOCKS_IN_COL]

        This function slices the
            full block mask self.block_mask:  [max_batch_size, H, MAX_BLOCKS_IN_ROW, MAX_BLOCKS_IN_COL]
            using input_pos and batch_id
        """

        # NOTE: this function is entirely in logical space
        def causal_offset(off: Tensor):
            def offset(b, h, q_idx, kv_idx):
                return q_idx + off[b] >= kv_idx

            return offset

        block_mask = self.block_mask
        # batch_idx: [B], input_pos: [B]
        assert batch_idx.ndim == 1, "batch_idx must be 1D"
        assert input_pos.ndim == 1, "input_pos must be 1D"
        (B,) = batch_idx.shape
        input_block_idx = input_pos // block_mask.BLOCK_SIZE[0]  # [B]
        kv_num_blocks = block_mask.kv_num_blocks[batch_idx, :, input_block_idx].view(B, 1, 1)
        kv_indices = block_mask.kv_indices[batch_idx, :, input_block_idx].view(B, 1, 1, -1)
        full_kv_num_blocks, full_kv_indices = None, None
        if block_mask.full_kv_num_blocks is not None:
            assert block_mask.full_kv_indices is not None
            full_kv_num_blocks = block_mask.full_kv_num_blocks[batch_idx, :, input_block_idx].view(B, 1, 1)  # noqa
            full_kv_indices = block_mask.full_kv_indices[batch_idx, :, input_block_idx].view(B, 1, 1, -1)  # noqa
        seq_length = (1, block_mask.seq_lengths[1])
        mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=block_mask.BLOCK_SIZE,
            mask_mod=causal_offset(input_pos),
            seq_lengths=seq_length,
        )
        return mask

    # ------------------------------------------------------------------
    # CUDA graph helpers
    # ------------------------------------------------------------------

    def _decode_forward_impl(
        self,
        batch_idx: torch.Tensor,
        input_ids: torch.Tensor,
        coord_xy: torch.Tensor,
        size_hw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode forward pass used for both eager and CUDA-graph capture.

        The engine handles mask construction and position lookup from
        persistent buffers, then delegates to ``model.forward``.
        """
        B = input_ids.shape[0]

        # Build attention mask from persistent position buffers
        mask = self.get_decoding_block_mask(batch_idx, self._input_pos[batch_idx])
        mask = self.paged_kv_cache.convert_logical_block_mask(mask, batch_idx)

        # RoPE positions from persistent buffer
        rope_pos_t = self._rope_pos_t[batch_idx]  # [B]

        return self.model.forward(
            tokens=input_ids.view(B, 1),
            attention_mask=mask,
            kv_cache=self.paged_kv_cache,
            rope_pos_t=rope_pos_t.view(B, 1),
            coord_xy=coord_xy,
            size_hw=size_hw,
            input_pos=self._input_pos[batch_idx].view(B, 1),
            batch_idx=batch_idx.view(-1),
            flex_attn_kernel_options=self.kernel_options,
        )

    def capture_decode_cudagraph(self):
        """Capture CUDA graphs for decode at various batch sizes.

        Call this *once* before the generation loop.  After capture, decode_step
        will replay the graph instead of re-launching every kernel individually.
        """
        max_bs = self.max_batch_size
        D = self.model.args.dim
        V = self.model.args.vocab_size

        # Static buffers that persist across replays
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=self.device)
        batch_idx = torch.arange(max_bs, dtype=torch.int64, device=self.device)
        coord_xy = torch.zeros((max_bs, 2), dtype=self.model.dtype, device=self.device)
        size_hw = torch.zeros((max_bs, 2), dtype=self.model.dtype, device=self.device)
        logits_out = torch.zeros((max_bs, 1, V), dtype=self.model.dtype, device=self.device)
        h_out = torch.zeros((max_bs, 1, D), dtype=self.model.dtype, device=self.device)

        # ---- Temporarily allocate KV cache pages for all batch slots ----
        # During capture, FlexAttention kernels will access KV cache memory via
        # the page table.  If no pages are allocated, the page table indices are
        # uninitialised (-1) and accessing them causes illegal memory access.
        # We reserve 1 page per slot, run warmup+capture, then free everything.
        temp_slots: list[int] = []
        page_size = self.paged_kv_cache.page_size
        for i in range(max_bs):
            if not self.paged_kv_cache.can_reserve(page_size):
                break
            slot = self.paged_kv_cache.allocate()
            slot_tensor = torch.tensor([slot], device=self.device, dtype=torch.long)
            self.paged_kv_cache.reserve(
                batch_idx_int=slot,
                batch_idx=slot_tensor,
                seq_len=page_size,
            )
            temp_slots.append(slot)

        # Batch size buckets: [1, 2, 4, 8] then multiples of 16 up to max_bs
        # Note: CUDA graphs require fixed tensor shapes. But the actual batch size
        # varies during generation (sequences finish at different times).
        # Solution: pre-capture a graph for each "bucket" size. At runtime, we pick
        # the smallest bucket >= actual batch size and pad with zeros.
        self._graph_bs = sorted(
            set([1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) + [max_bs])
        )
        self._graph_bs = [bs for bs in self._graph_bs if bs <= max_bs]
        self._graphs = {}
        self._graph_pool = None

        # The loop runs largest-to-smallest (reversed), so the first capture (bs=max)
        # establishes the pool, and smaller captures reuse it.
        for bs in reversed(self._graph_bs):
            logger.info("Capturing decode graph for batch_size=%d", bs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()

            # Warmup (populates caches inside FlexAttention / compiled kernels)
            logits_out[:bs], h_out[:bs] = self._decode_forward_impl(
                batch_idx[:bs], input_ids[:bs], coord_xy[:bs], size_hw[:bs]
            )

            # Capture
            with torch.cuda.graph(graph, pool=self._graph_pool):
                logits_out[:bs], h_out[:bs] = self._decode_forward_impl(
                    batch_idx[:bs], input_ids[:bs], coord_xy[:bs], size_hw[:bs]
                )
            logger.info("  bs=%d: captured", bs)
            if self._graph_pool is None:
                self._graph_pool = graph.pool()
            self._graphs[bs] = graph
            torch.cuda.synchronize()

        # ---- Free temporary KV cache pages ----
        for slot in temp_slots:
            self._input_pos[slot] = 0
            self._rope_pos_t[slot] = 0
            self.paged_kv_cache.erase(slot)

        self._graph_vars = dict(
            input_ids=input_ids,
            batch_idx=batch_idx,
            coord_xy=coord_xy,
            size_hw=size_hw,
            logits_out=logits_out,
            h_out=h_out,
        )
        self.cudagraph_captured = True
        logger.info("Captured %d graphs for bs=%s", len(self._graphs), self._graph_bs)

    def _decode_forward(self, sequences: list[Sequence]):
        """Run one decode step.

        When a CUDA graph has been captured, replays the graph (zero CPU
        dispatch overhead).  Otherwise calls ``_decode_forward_impl``
        directly (eager).  Both paths use the same graph-safe forward
        function, avoiding any dict materialisation for coords.
        """
        B = len(sequences)

        if self.cudagraph_captured:
            graph_bs = next(x for x in self._graph_bs if x >= B)
            graph = self._graphs[graph_bs]
            gv = self._graph_vars

            # ── CPU-only: fill pinned staging buffers (zero CUDA calls) ──
            self._decode_gv_bidx_pinned.zero_()
            for i, seq in enumerate(sequences):
                bidx = seq.batch_idx
                self._decode_bidx_pinned[i] = bidx
                self._decode_pos_pinned[i] = seq.total_length - 1
                self._decode_gv_bidx_pinned[i] = bidx

            # ── Setup on default stream (pinned→GPU, scatter, etc.) ──
            self._decode_bidx_gpu[:B].copy_(
                self._decode_bidx_pinned[:B], non_blocking=True)
            self._decode_pos_gpu[:B].copy_(
                self._decode_pos_pinned[:B], non_blocking=True)
            self._input_pos.scatter_(
                0, self._decode_bidx_gpu[:B], self._decode_pos_gpu[:B])

            for i, seq in enumerate(sequences):
                self._rope_pos_t[seq.batch_idx].copy_(seq.pos_t)
                gv["input_ids"][i].copy_(seq.last_token_id)

            gv["batch_idx"].copy_(
                self._decode_gv_bidx_pinned, non_blocking=True)

            gv["input_ids"][B:].zero_()

            gv["coord_xy"].zero_()
            gv["size_hw"].zero_()
            for i, seq in enumerate(sequences):
                if seq.current_bbox_xy is not None:
                    gv["coord_xy"][i].copy_(seq.current_bbox_xy)
                    gv["size_hw"][i].copy_(seq.current_bbox_hw)  # pyright: ignore[reportArgumentType]

            graph.replay()
            logits_BSV = gv["logits_out"][:B]
            h_BSD = gv["h_out"][:B]
        else:
            # --- Eager path: build tensors and call directly ---
            # Pin the CPU tensor so .to(device, non_blocking=True) is truly
            # async and doesn't trigger cudaStreamSynchronize (pageable H2D).
            batch_idx = torch.tensor(
                [seq.batch_idx for seq in sequences],
                dtype=torch.int64,
            ).pin_memory().to(self.device, non_blocking=True)

            for seq in sequences:
                self._input_pos[seq.batch_idx].fill_(seq.total_length - 1)
                self._rope_pos_t[seq.batch_idx].copy_(seq.pos_t)

            input_ids = torch.stack(
                [seq.last_token_id for seq in sequences]
            ).to(self.device, non_blocking=True)
            coord_xy = torch.zeros((B, 2), dtype=self.model.dtype, device=self.device)
            size_hw = torch.zeros((B, 2), dtype=self.model.dtype, device=self.device)
            for i, seq in enumerate(sequences):
                if seq.current_bbox_xy is not None:
                    coord_xy[i] = seq.current_bbox_xy
                    size_hw[i] = seq.current_bbox_hw  # pyright: ignore[reportArgumentType]

            logits_BSV, h_BSD = self._decode_forward_impl(
                batch_idx, input_ids, coord_xy, size_hw,
            )

        return logits_BSV, h_BSD


    def decode_step(self, sequences: list[Sequence]):
        """Sync-free decode step — no GPU→CPU reads on the hot path."""
        logits_BSV, h_BSD = self._decode_forward(sequences)

        next_token, logits, probs = sample_token(
            logits_BSV[:, -1, :],
            rng=self.rng,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        next_token = next_token.squeeze(-1)   # (B,) int64 GPU
        logits = logits.squeeze(-1)           # (B,) float32 GPU
        probs = probs.squeeze(-1)             # (B,) float32 GPU

        h_last = h_BSD[:, -1, :]   # (B, D)
        xy_B2, hw_B2, is_coord_B, is_size_B, coord_logits = self.model.sample_bbox(h_last, next_token)
        segm_BD, is_segm_B = self.model.get_segm_tokens(h_last, next_token)

        for b, seq in enumerate(sequences):
            thresh = seq.sampling_params.coord_dedup_threshold
            if thresh > 0:
                raw = seq.output_aux.coord_history_raw()
                if raw is not None:
                    self.model.dedup_single_coord(
                        xy_B2[b], is_coord_B[b],
                        raw[0], raw[1], coord_logits[b],
                        threshold=thresh,
                    )
            seq.add_next_token(
                token_id=next_token[b],
                logits=logits[b],
                probs=probs[b],
                xy=xy_B2[b],
                hw=hw_B2[b],
                is_coord=is_coord_B[b],
                is_size=is_size_B[b],
                segm=segm_BD[b],
                is_segm=is_segm_B[b],
                pos_t=None,
            )

        self._stage_token_ids_d2h(next_token)

    def _check_done(self, sequences: list[Sequence]):
        # Token IDs were staged to pinned memory (decode_step inlines the
        # staging; prefill_sequences calls _stage_token_ids_d2h).
        assert self._token_ids_ready is not None, (
            "Token IDs were not staged before _check_done"
        )
        self._token_ids_ready.synchronize()
        last_ids_cpu: list[int] = self._token_ids_pinned[:self._token_ids_count].tolist()
        self._token_ids_ready = None

        # Identify done sequences.
        done_seqs: list[Sequence] = []
        for seq, last_id_int in zip(sequences, last_ids_cpu):
            assert seq.sampling_params is not None
            assert seq.batch_idx is not None
            should_stop = last_id_int in seq.sampling_params.stop_token_ids
            is_max_len = seq.input_length + seq.output_length >= self.max_seq_length
            is_max_new = seq.output_length == seq.sampling_params.max_new_tokens
            if should_stop or is_max_len or is_max_new:
                done_seqs.append(seq)

        if done_seqs:
            self._finalize_done(done_seqs)

        return [seq for seq in sequences if not seq.finished]

    def _finalize_done(self, done_seqs: list[Sequence]) -> None:
        """Materialize bboxes and masks, then release GPU resources."""
        for seq in done_seqs:
            with _oom_guard(
                self.device, "FINALIZE",
                n_segm_tokens=len(seq.output_aux.segm_embeds),
                image_size=seq.original_image_size,
            ), _Timer() as t:
                seq.output_aux.finalize(
                    hr_image_features=seq.hr_image_features,
                    threshold=seq.sampling_params.segmentation_threshold,
                    task=seq.task,
                    original_image_size=seq.original_image_size,
                )
            seq.stats.finalize_ms = t.ms
            seq.hr_image_features = None
            seq.image_tensor = None
            seq.finished = True
            self.done.append(seq)
            self._input_pos[seq.batch_idx].fill_(0)
            self._rope_pos_t[seq.batch_idx].fill_(0)
            self.paged_kv_cache.erase(seq.batch_idx)

    def _prefetch_next_tokenize(self):
        """Submit tokenization for the next un-tokenized sequences in the waiting queue.

        Fills up to ``_prefetch_workers`` slots so that burst slot openings
        (e.g. many sequences finishing in the same decode step) find their
        sequences already tokenized.
        """
        # Reap completed futures
        done_ids = [sid for sid, f in self._prefetch_futures.items() if f.done()]
        for sid in done_ids:
            self._prefetch_futures.pop(sid)

        if not self.waiting:
            return

        budget = self._prefetch_workers - len(self._prefetch_futures)
        for seq in self.waiting:
            if budget <= 0:
                break
            if seq.input_ids is not None or id(seq) in self._prefetch_futures:
                continue
            self._prefetch_futures[id(seq)] = self._tokenize_executor.submit(
                self._tokenize_single, seq,
            )
            budget -= 1

    def _await_prefetch(self, seq: Sequence):
        """Block until background tokenization completes for *seq*, if one is running."""
        future = self._prefetch_futures.pop(id(seq), None)
        if future is not None:
            future.result()

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def _should_prefill(self) -> bool:
        """Decide whether the next step should be a prefill."""
        if not self.waiting or not self.paged_kv_cache.free_batch_idx:
            return False  #  nothing to prefill or no space to prefill
        if not self.running:
            return True  # no running sequences, so prefill
        return self._steps_since_prefill >= self.max_decode_steps_between_prefills

    def _build_prefill_batch(self) -> list[Sequence]:
        """Pop sequences from waiting, tokenize, allocate KV pages.

        Only allocates enough pages for the current sequence length (not
        future tokens) so that we maximise the decode batch size.  If a
        sequence was preempted mid-decode we re-prefill the full length
        generated so far.
        """
        batch: list[Sequence] = []
        prefill_length_sum = 0
        while self.waiting and len(self.paged_kv_cache.free_batch_idx) > 0:
            seq = self.waiting[0]
            if seq.input_ids is None:
                _tok_t0 = time.perf_counter()
                self._await_prefetch(seq)
                if seq.input_ids is None:
                    self._tokenize_single(seq)
                    seq.stats.tokenize_ms = (time.perf_counter() - _tok_t0) * 1000.0

            if not self.paged_kv_cache.can_reserve(seq.total_length) or (
                batch
                and self.prefill_length_limit != -1
                and prefill_length_sum + seq.total_length >= self.prefill_length_limit
            ):
                break

            prefill_length_sum += seq.total_length
            seq = self.waiting.popleft()
            batch_idx_int = self.paged_kv_cache.allocate()
            batch_idx_tensor = torch.tensor([batch_idx_int], device=self.device, dtype=torch.long)
            self.paged_kv_cache.reserve(
                batch_idx_int=batch_idx_int,
                batch_idx=batch_idx_tensor,
                seq_len=seq.total_length,
            )
            seq.batch_idx = batch_idx_int
            batch.append(seq)
        return batch

    def _build_decode_batch(self) -> list[Sequence]:
        """Reserve pages for running sequences; preempt newest if OOM.

        running is ordered oldest-first so preemption evicts the
        most-recently-admitted sequence (least work lost).
        """
        batch: list[Sequence] = []
        while self.running:
            seq = self.running.popleft()
            assert seq.batch_idx is not None, "Sequence batch_idx missing; scheduling bug."
            if self.paged_kv_cache.capacity[seq.batch_idx] >= seq.total_length:
                batch.append(seq)
            elif self.paged_kv_cache.can_reserve(seq.total_length, batch_idx_int=seq.batch_idx):
                bidx_t = torch.tensor([seq.batch_idx], dtype=torch.long).pin_memory().to(
                    self.device, non_blocking=True
                )
                self.paged_kv_cache.reserve(
                    batch_idx_int=seq.batch_idx,
                    batch_idx=bidx_t,
                    seq_len=seq.total_length,
                )
                batch.append(seq)
            else:
                self.running.appendleft(seq)
                newest = self.running.pop()
                assert newest.batch_idx is not None, "Sequence batch_idx missing; scheduling bug."
                self.waiting.appendleft(newest)
                self.paged_kv_cache.erase(newest.batch_idx)
                newest.stats.preemptions += 1
        return batch

    def run_one_step(self):
        self._prefetch_next_tokenize()

        # ── Try prefill if the scheduler says so ──────────────────
        if self._should_prefill():
            batch = self._build_prefill_batch()
            if batch:
                self._prefetch_next_tokenize()
                with _Timer() as t, _oom_guard(
                    self.device, "PREFILL",
                    batch_size=len(batch),
                    total_tokens=sum(s.total_length for s in batch),
                    seq_lengths=[s.total_length for s in batch],
                ):
                    self.prefill_sequences(batch)
                prefill_n_tokens = sum(s.total_length for s in batch)
                self._prefill_batch_sizes.append(len(batch))
                self._prefill_token_counts.append(prefill_n_tokens)
                for seq in batch:
                    seq.stats.prefill_ms += t.ms
                    seq.stats.prefill_batch_size = len(batch)
                    seq.stats.prefill_tokens += seq.total_length
                self.running.extend(self._check_done(batch))
                if self._steps_since_prefill > 0:
                    self._decode_run_lengths.append(self._steps_since_prefill)
                self._steps_since_prefill = 0
                return "prefill+upsampler"

        # ── Decode step ───────────────────────────────────────────
        if self.running:
            batch = self._build_decode_batch()
            if batch:
                self._prefetch_next_tokenize()
                with _Timer() as t, _oom_guard(
                    self.device, "DECODE", batch_size=len(batch),
                ):
                    self.decode_step(batch)
                _bs = len(batch)
                for seq in batch:
                    seq.stats.decode_wall_ms += t.ms
                    seq.stats.decode_steps += 1
                    seq.stats.decode_batch_sum += _bs
                self.running = deque(self._check_done(batch))
                self._steps_since_prefill += 1
                return "decode"

        return "idle"

    def _tokenize_single(self, seq: Sequence):
        """
        Tokenize a single sequence on-demand.
        This is called lazily when the sequence is about to be prefilled.
        """
        # Get image (PIL or from path)
        pil_img = seq.pil_image
        if pil_img is not None:
            # Store original size before any processing
            if seq.original_image_size is None:
                seq.original_image_size = (pil_img.height, pil_img.width)
            # Resize if necessary
            img = resize_image_if_necessary(pil_img, seq.min_image_size, seq.max_image_size)
            seq.image_hash = int.from_bytes(hashlib.sha256(img.tobytes()).digest()[:8])
            images = [img]
        else:
            images = []

        # Preprocess images
        if images:
            images = self.image_processor.preprocess(images=images)

        input_ids_np, img_tensor_list = tokenize_inputs(
            seq.text,
            images,
            self.tokenizer,
            self.model.args.spatial_patch_size,
            merge_size=1,
            max_length=self.max_seq_length,
        )
        img_tensor = img_tensor_list[0]
        pixel_mask_THW = np.ones(img_tensor.shape[:3], dtype=bool)
        tpos_np, hw_np = get_pos_thw_single(
            input_ids_np, pixel_mask_THW, self.tokenizer,
            self.model.args.spatial_patch_size,
        )
        input_ids_1d = torch.from_numpy(input_ids_np)
        seq.rope_pos_t = torch.from_numpy(tpos_np)
        seq.rope_pos_hw = torch.from_numpy(hw_np)

        seq.image_tensor = img_tensor
        seq.input_ids = input_ids_1d  # set last: acts as "ready" flag for prefetch

    @torch.inference_mode()
    def generate(
        self,
        sequences: list[Sequence],
        use_tqdm=False,
        profiler=None,
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        print_stats=False,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.waiting = deque(sequences)
        self.running.clear()
        self.done.clear()
        self._steps_since_prefill = 0
        self._decode_run_lengths = []
        self._prefill_batch_sizes = []
        self._prefill_token_counts = []
        process_sampling_params(sequences, sampling_params, self.tokenizer)

        self._prefetch_next_tokenize()

        total_sequences = len(self.waiting)
        times = []
        with tqdm(total=total_sequences, disable=not use_tqdm, desc="Generating") as pbar:
            prev_done = 0
            while self.waiting or self.running:
                time_start = time.perf_counter()
                step_type = self.run_one_step()
                time_end = time.perf_counter()

                times.append({"step_type": step_type, "time": time_end - time_start})
                if profiler:
                    profiler.step()
                curr_done = len(self.done)
                if curr_done > prev_done:
                    pbar.update(curr_done - prev_done)
                    prev_done = curr_done
        if print_stats:
            self.print_time_stats(times)

        return sorted(self.done, key=lambda s: s.request_idx)

    def print_time_stats(self, times):
        stats = {}
        for step in ["decode", "prefill+upsampler"]:
            step_times = [t["time"] for t in times if t["step_type"] == step]
            stats[step] = {
                "count": len(step_times),
                "total": sum(step_times),
                "mean": sum(step_times) / len(step_times) if step_times else 0,
                "min": min(step_times) if step_times else 0,
                "max": max(step_times) if step_times else 0,
            }

        print("\nScheduler:")
        print(f"  max_decode_steps_between_prefills: {self.max_decode_steps_between_prefills}")

        print("\nTime statistics by step type:")
        for step, metrics in stats.items():
            print(f"\n{step}:")
            print(f"  Count: {metrics['count']}")
            print(f"  Total: {metrics['total']:.4f}s")
            print(f"  Mean:  {metrics['mean']:.4f}s")
            print(f"  Min:   {metrics['min']:.4f}s")
            print(f"  Max:   {metrics['max']:.4f}s")
        print(f"\nTotal time: {sum(t['time'] for t in times):.4f}s")

        # Scheduling metrics
        done_seqs = list(self.done)
        total_decode_steps = sum(s.stats.decode_steps for s in done_seqs)
        total_decode_batch_sum = sum(s.stats.decode_batch_sum for s in done_seqs)
        avg_decode_bs = total_decode_batch_sum / total_decode_steps if total_decode_steps else 0
        total_preemptions = sum(s.stats.preemptions for s in done_seqs)

        runs = self._decode_run_lengths
        if self._steps_since_prefill > 0:
            runs = runs + [self._steps_since_prefill]

        print(f"\nScheduling metrics:")
        print(f"  Avg decode batch size: {avg_decode_bs:.1f}")
        print(f"  Total preemptions:     {total_preemptions}")

        pbs = self._prefill_batch_sizes
        ptc = self._prefill_token_counts
        if pbs:
            print(f"  Prefill steps:         {len(pbs)}")
            print(f"  Prefill samples/step:  min={min(pbs)}, mean={sum(pbs)/len(pbs):.1f}, max={max(pbs)}")
            print(f"  Prefill tokens/step:   min={min(ptc)}, mean={sum(ptc)/len(ptc):.0f}, max={max(ptc)}")
        if runs:
            print(f"  Decode run lengths:    min={min(runs)}, mean={sum(runs)/len(runs):.1f}, max={max(runs)}  (n={len(runs)} runs)")

        if torch.cuda.is_available():
            print("\nCUDA memory (peak):")
            print(f"  allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")
            print(f"  reserved:  {torch.cuda.max_memory_reserved()/1024**3:.2f} GiB")

