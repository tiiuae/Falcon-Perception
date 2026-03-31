# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""
OCR Inference Engine with optional 3rd-party layout detection.

Supports two generation modes:

- **plain**:  Full-page OCR on each image -> list of text strings.
- **layout**: 3rd-party layout model detects regions -> crop per region ->
              OCR per crop -> reassembled per-image list of dicts with
              category, bbox, score, text.

The layout model (PP-DocLayoutV3 by default) is loaded lazily on first use.

Both modes delegate the actual autoregressive generation to
:class:`PagedInferenceEngine`, reusing its scheduling, prefill/decode loop,
paged KV cache management, preemption, and optional CUDA-graph replay.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from falcon_perception.data import load_image
from falcon_perception.paged_inference import (
    PagedInferenceEngine,
    SamplingParams,
    Sequence,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout category -> OCR prompt mapping
# ---------------------------------------------------------------------------

CATEGORY_PROMPTS = {
    "plain": "Extract the text content from this image.",
    "formula": "Extract the formula content from this image.",
    "table": "Extract the table content from this image.",
    "text": "Extract the text content from this image.",
    "caption": "Extract the caption content from this image.",
    "footnote": "Extract the footnote content from this image.",
    "list-item": "Extract the list-item content from this image.",
    "page-footer": "Extract the page-footer content from this image.",
    "page-header": "Extract the page-header content from this image.",
    "section-header": "Extract the section-header content from this image.",
    "title": "Extract the title content from this image.",
}

LAYOUT_TO_OCR_CATEGORY: dict[str, str | None] = {
    "text": "text",
    "table": "table",
    "formula": "formula",
    "caption": "caption",
    "footnote": "footnote",
    "list-item": "list-item",
    "title": "title",
    "header": "text",
    "footer": "page-footer",
    "number": "text",
    "figure_title": "caption",
    "paragraph_title": "section-header",
    "doc_title": "title",
    "reference_content": "text",
    "reference": "text",
    "abstract": "text",
    "aside_text": "text",
    "content": "text",
    "formula_number": "text",
    "vision_footnote": "footnote",
    "algorithm": "text",
    "page-footer": "page-footer",
    "page-header": "page-header",
    "section-header": "section-header",
    # Categories with no text to extract
    "image": None,
    "picture": None,
    "figure": None,
    "chart": None,
    "seal": None,
}

_LAYOUT_TARGET_H, _LAYOUT_TARGET_W = 800, 800
_MIN_CROP_DIM = 16

# ---------------------------------------------------------------------------
# Geometry helpers for nested-detection filtering
# ---------------------------------------------------------------------------


def _box_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _intersection_area(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _containment_ratio(small, large):
    area = _box_area(small)
    if area <= 0:
        return 0.0
    return _intersection_area(small, large) / area


def _iou(a, b):
    inter = _intersection_area(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def dedup_overlapping_detections(
    detections: list[dict], iou_threshold: float = 0.8, area_ratio_threshold: float = 0.9,
) -> list[dict]:
    """Remove near-duplicate boxes (high IoU), keeping the larger one.

    When two boxes overlap above *iou_threshold*, the one with the larger area
    is retained.  If the areas are similar (min/max > *area_ratio_threshold*),
    one is chosen at random.
    """
    if len(detections) <= 1:
        return detections

    suppressed: set[int] = set()
    for i in range(len(detections)):
        if i in suppressed:
            continue
        for j in range(i + 1, len(detections)):
            if j in suppressed:
                continue
            if _iou(detections[i]["bbox"], detections[j]["bbox"]) > iou_threshold:
                area_i = _box_area(detections[i]["bbox"])
                area_j = _box_area(detections[j]["bbox"])
                ratio = min(area_i, area_j) / max(area_i, area_j) if max(area_i, area_j) > 0 else 1.0
                if ratio > area_ratio_threshold:
                    loser = random.choice([i, j])
                    suppressed.add(loser)
                    if loser == i:
                        break
                elif area_i >= area_j:
                    suppressed.add(j)
                else:
                    suppressed.add(i)
                    break
    return [d for k, d in enumerate(detections) if k not in suppressed]


def filter_nested_detections(
    detections: list[dict], containment_threshold: float = 0.8
) -> list[dict]:
    """Remove any box that is mostly contained within a strictly larger box."""
    areas = [_box_area(d["bbox"]) for d in detections]
    keep = []
    for i, det in enumerate(detections):
        is_nested = False
        for j, other in enumerate(detections):
            if i == j:
                continue
            if areas[j] <= areas[i]:
                continue
            if _containment_ratio(det["bbox"], other["bbox"]) > containment_threshold:
                is_nested = True
                break
        if not is_nested:
            keep.append(det)
    return keep


# ---------------------------------------------------------------------------
# OCR Inference Engine
# ---------------------------------------------------------------------------


class OCRInferenceEngine(PagedInferenceEngine):
    """OCR engine supporting plain and layout modes.

    Extends :class:`PagedInferenceEngine`, reusing its scheduling,
    prefill/decode, CUDA-graph support, and paged KV-cache management.

    Additional capabilities:

    * Lazy-loaded 3rd-party layout model (PP-DocLayoutV3 via
      ``AutoModelForObjectDetection``).
    * Batch layout detection as a preprocessing step (separate from the
      paged engine loop).
    * Per-category OCR prompt construction.
    * Result reassembly: layout bboxes + OCR text per region.
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        max_batch_size: int = 64,
        max_seq_length: int = 4096,
        n_pages: int = 1536,
        page_size: int = 128,
        prefill_length_limit: int = 16384,
        kernel_options: dict | None = None,
        seed: int | None = None,
        capture_cudagraph: bool = True,
        max_decode_steps_between_prefills: int = 16
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
            n_pages=n_pages,
            page_size=page_size,
            prefill_length_limit=prefill_length_limit,
            kernel_options=kernel_options,
            seed=seed,
            enable_hr_cache=False,
            capture_cudagraph=capture_cudagraph,
            max_decode_steps_between_prefills=max_decode_steps_between_prefills
        )
        self._layout_det_model: torch.nn.Module | None = None
        self._layout_processor: object | None = None
        self._layout_id2label: dict[int, str] | None = None
        self._tvF: object | None = None

    # ── Layout model management ──────────────────────────────────────

    def load_layout_model(
        self,
        layout_model: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
    ):
        """Lazy-load 3rd-party layout detection model from HuggingFace.

        Called automatically on the first ``generate_with_layout`` invocation.
        Can also be called explicitly to pre-warm the model.
        """
        if self._layout_det_model is not None:
            return
        import torchvision.transforms.functional as tvF
        from transformers import (
            AutoModelForObjectDetection,
            PPDocLayoutV3ImageProcessorFast,
        )

        self._layout_processor = PPDocLayoutV3ImageProcessorFast.from_pretrained(
            layout_model
        )
        self._layout_det_model = (
            AutoModelForObjectDetection.from_pretrained(
                layout_model, dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )
        self._layout_id2label = self._layout_det_model.config.id2label  # type: ignore[union-attr]
        self._tvF = tvF
        logger.info("Layout model loaded: %s", layout_model)

    @torch.inference_mode()
    def run_layout_detection(
        self,
        images: list[Image.Image],
        threshold: float = 0.5,
    ) -> list[list[dict]]:
        """Run layout detection on a batch of PIL images.

        Returns per-image list of ``{category, bbox [x1,y1,x2,y2], score}``,
        sorted by reading order.
        """
        assert self._layout_det_model is not None, "call load_layout_model() first"
        assert self._layout_processor is not None
        assert self._layout_id2label is not None
        assert self._tvF is not None
        device = self.device
        tvF = self._tvF  # torchvision.transforms.functional

        target_sizes = torch.tensor([img.size[::-1] for img in images])
        tensors = [tvF.pil_to_tensor(img) for img in images]  # type: ignore[attr-defined]

        # GPU-accelerated resize + normalize
        pixel_batch = torch.empty(
            len(tensors), 3, _LAYOUT_TARGET_H, _LAYOUT_TARGET_W,
            dtype=torch.float16, device=device,
        )
        size_groups: dict[tuple[int, int], list[int]] = {}
        for i, t in enumerate(tensors):
            size_groups.setdefault((t.shape[1], t.shape[2]), []).append(i)

        for _shape, indices in size_groups.items():
            batch = torch.stack([tensors[i] for i in indices])
            batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
            batch = F.interpolate(
                batch,
                size=(_LAYOUT_TARGET_H, _LAYOUT_TARGET_W),
                mode="bicubic",
                align_corners=False,
                antialias=False,
            )
            batch = (batch.clamp_(0, 255) / 255.0).to(torch.float16)
            for j, idx in enumerate(indices):
                pixel_batch[idx] = batch[j]
            del batch

        outputs = self._layout_det_model(pixel_values=pixel_batch)
        del pixel_batch

        # Post-process on GPU
        logits = outputs.logits
        boxes = outputs.pred_boxes
        order_logits = outputs.order_logits

        box_centers, box_dims = boxes.split(2, dim=-1)
        boxes_xyxy = torch.cat(
            [box_centers - 0.5 * box_dims, box_centers + 0.5 * box_dims], dim=-1
        )

        img_h, img_w = target_sizes.unbind(1)
        scale = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(
            device, dtype=boxes_xyxy.dtype
        )
        boxes_xyxy = boxes_xyxy * scale[:, None, :]

        num_queries = logits.shape[1]
        num_classes = logits.shape[2]
        scores = logits.sigmoid()
        scores_flat, index = scores.flatten(1).topk(num_queries, dim=-1)
        labels = index % num_classes
        box_indices = index // num_classes
        boxes_xyxy = boxes_xyxy.gather(
            dim=1, index=box_indices.unsqueeze(-1).expand(-1, -1, 4)
        )

        order_seqs = self._layout_processor._get_order_seqs(order_logits)  # type: ignore[union-attr]
        order_seqs = order_seqs.gather(dim=1, index=box_indices)

        batch_results = []
        for s, l, b, o in zip(scores_flat, labels, boxes_xyxy, order_seqs):
            mask = s >= threshold
            o_valid = o[mask]
            _, indices_sorted = o_valid.sort()

            detections = []
            for si, li, bi in zip(
                s[mask][indices_sorted],
                l[mask][indices_sorted],
                b[mask][indices_sorted],
            ):
                detections.append({
                    "category": self._layout_id2label[li.item()],
                    "bbox": [round(x, 2) for x in bi.tolist()],
                    "score": round(si.item(), 4),
                })
            batch_results.append(detections)

        return batch_results

    # ── Helpers ───────────────────────────────────────────────────────

    def _decode_seq_text(self, seq: Sequence) -> str:
        """Decode generated token ids to a clean text string."""
        text = self.tokenizer.decode(seq.output_ids)
        return (
            text.replace("<|end_of_query|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )

    @staticmethod
    def _make_ocr_prompt(category: str = "plain") -> str:
        instruction = CATEGORY_PROMPTS.get(
            category.strip().lower(), CATEGORY_PROMPTS["plain"]
        )
        return f"<|image|>{instruction}\n<|OCR_PLAIN|>"

    def _stop_token_ids(self) -> list[int]:
        return [
            self.tokenizer.end_of_query_token_id,
            self.tokenizer.eos_token_id,
        ]

    # ── Generate: plain OCR ──────────────────────────────────────────

    @torch.inference_mode()
    def generate_plain(
        self,
        images: list[Image.Image | str],
        *,
        category: str | list[str] = "plain",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_k: int | None = None,
        min_image_size: int = 64,
        max_image_size: int = 1024,
        use_tqdm: bool = True,
        print_stats: bool = False,
        profiler=None,
    ) -> list[str]:
        """Full-page OCR on each image.

        Args:
            images: PIL images (or paths/URLs).
            category: OCR category (applied to all if a single string, or
                one per image).  See :data:`CATEGORY_PROMPTS` for valid keys.
            max_new_tokens: Max generation steps per sequence.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k sampling (``None`` = disabled).
            min_image_size: Min side after resize.
            max_image_size: Max side after resize.
            print_stats: Print scheduling / throughput statistics.
            profiler: Optional ``torch.profiler`` instance (stepped each engine step).

        Returns:
            One text string per image.
        """
        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        if isinstance(category, str):
            category = [category] * len(images)

        sequences = [
            Sequence(
                text=self._make_ocr_prompt(cat),
                image=img,
                min_image_size=min_image_size,
                max_image_size=max_image_size,
                request_idx=idx,
                task="ocr",
            )
            for idx, (img, cat) in enumerate(zip(images, category))
        ]

        done = super().generate(
            sequences,
            sampling_params=SamplingParams(
                max_new_tokens=max_new_tokens,
                stop_token_ids=self._stop_token_ids(),
            ),
            temperature=temperature,
            top_k=top_k,
            use_tqdm=use_tqdm,
            print_stats=print_stats,
            profiler=profiler,
        )
        return [self._decode_seq_text(seq) for seq in done]

    # ── Layout crop helpers ─────────────────────────────────────────

    @staticmethod
    def build_crop_sequences(
        pil_img: Image.Image,
        detections: list[dict],
        *,
        min_image_size: int = 64,
        max_image_size: int = 1024,
        min_crop_dim: int = _MIN_CROP_DIM,
    ) -> list[tuple[Sequence, int]]:
        """Build OCR sequences from layout detections on a single image.

        Returns a list of ``(Sequence, det_idx)`` pairs.  Crops that are too
        small or have an unrecognised layout category are skipped.
        """
        img_w, img_h = pil_img.size
        results: list[tuple[Sequence, int]] = []

        for det_idx, det in enumerate(detections):
            cat_key = det["category"].strip().lower()
            ocr_cat = LAYOUT_TO_OCR_CATEGORY.get(cat_key)
            if ocr_cat is None:
                continue

            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(img_w, int(x2 + 0.5)), min(img_h, int(y2 + 0.5))
            cw, ch = x2 - x1, y2 - y1
            if cw < min_crop_dim or ch < min_crop_dim:
                continue
            short, long = sorted((cw, ch))
            if long > max_image_size and short * (max_image_size / long) < min_crop_dim:
                continue

            crop = pil_img.crop((x1, y1, x2, y2))
            seq = Sequence(
                text=OCRInferenceEngine._make_ocr_prompt(ocr_cat),
                image=crop,
                min_image_size=min_image_size,
                max_image_size=max_image_size,
                request_idx=0,
                task="ocr",
            )
            results.append((seq, det_idx))

        return results

    # ── Generate: layout + OCR ───────────────────────────────────────

    @torch.inference_mode()
    def generate_with_layout(
        self,
        images: list[Image.Image | str],
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_k: int | None = None,
        min_image_size: int = 64,
        max_image_size: int = 1024,
        use_tqdm: bool = True,
        print_stats: bool = False,
        profiler=None,
        layout_threshold: float = 0.3,
        layout_batch_size: int = 16,
        containment_threshold: float = 0.8,
        layout_model: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
    ) -> list[list[dict]]:
        """Run layout detection then OCR on each detected region.

        1. Batch layout detection via a 3rd-party model.
        2. Filter nested / duplicate detections.
        3. For each text-bearing detection, crop the region and create a
           plain-OCR sequence with the category-appropriate prompt.
        4. Run all crop sequences through the paged engine.
        5. Reassemble per-image results.

        Args:
            images: PIL images (or paths/URLs).
            max_new_tokens: Max generation steps per crop.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k sampling (``None`` = disabled).
            min_image_size: Min crop side after resize for OCR.
            max_image_size: Max crop side after resize for OCR.
            print_stats: Print scheduling / throughput statistics.
            profiler: Optional ``torch.profiler`` instance (stepped each engine step).
            layout_threshold: Confidence threshold for layout detections.
            layout_batch_size: Batch size for layout detection forward pass.
            containment_threshold: Drop boxes mostly contained in larger ones.
            layout_model: HuggingFace model ID for layout detection.

        Returns:
            Per-image list of dicts with keys ``category``, ``bbox``
            ``[x1, y1, x2, y2]``, ``score``, ``text``.
        """
        self.load_layout_model(layout_model)

        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        pil_images: list[Image.Image] = []
        for img in images:
            pil = img if isinstance(img, Image.Image) else load_image(img)
            pil_images.append(pil.convert("RGB"))  # type: ignore[union-attr]

        # ── 1. Batch layout detection ─────────────────────────────────
        all_dets: list[list[dict]] = []
        for i in range(0, len(pil_images), layout_batch_size):
            chunk = pil_images[i : i + layout_batch_size]
            all_dets.extend(
                self.run_layout_detection(chunk, threshold=layout_threshold)
            )

        # ── 2. Filter nested boxes, then dedup overlapping ones ───────
        all_dets = [
            dedup_overlapping_detections(filter_nested_detections(d, containment_threshold))
            for d in all_dets
        ]

        # ── 2b. Fallback: images with no useful layout -> plain OCR ──
        fallback_indices: set[int] = set()
        for img_idx, dets in enumerate(all_dets):
            if not dets or (
                len(dets) == 1
                and dets[0]["category"].strip().lower() == "image"
            ):
                fallback_indices.add(img_idx)
                logger.debug("Image %d: layout empty/trivial, falling back to plain OCR", img_idx)

        # ── 3. Build crop sequences and track origins ─────────────────
        sequences: list[Sequence] = []
        crop_origins: list[tuple[int, int]] = []  # (image_idx, det_idx)

        for img_idx, (pil_img, dets) in enumerate(zip(pil_images, all_dets)):
            if img_idx in fallback_indices:
                seq = Sequence(
                    text=self._make_ocr_prompt("plain"),
                    image=pil_img,
                    min_image_size=min_image_size,
                    max_image_size=max_image_size,
                    request_idx=len(sequences),
                    task="ocr",
                )
                sequences.append(seq)
                crop_origins.append((img_idx, -1))
                continue
            for seq, det_idx in self.build_crop_sequences(
                pil_img, dets,
                min_image_size=min_image_size,
                max_image_size=max_image_size,
            ):
                seq.request_idx = len(sequences)
                sequences.append(seq)
                crop_origins.append((img_idx, det_idx))

        # ── 4. OCR all crops via the paged engine ─────────────────────
        crop_texts: list[str] = []
        if sequences:
            done = super().generate(
                sequences,
                sampling_params=SamplingParams(
                    max_new_tokens=max_new_tokens,
                    stop_token_ids=self._stop_token_ids(),
                ),
                temperature=temperature,
                top_k=top_k,
                use_tqdm=use_tqdm,
                print_stats=print_stats,
                profiler=profiler,
            )
            crop_texts = [self._decode_seq_text(seq) for seq in done]

        # ── 5. Reassemble per-image results ───────────────────────────
        results: list[list[dict]] = [[] for _ in range(len(pil_images))]
        for (img_idx, det_idx), text in zip(crop_origins, crop_texts):
            if det_idx == -1:
                w, h = pil_images[img_idx].size
                results[img_idx].append({
                    "category": "plain",
                    "bbox": [0, 0, w, h],
                    "score": 1.0,
                    "text": text,
                })
            else:
                det = all_dets[img_idx][det_idx]
                results[img_idx].append({
                    "category": det["category"],
                    "bbox": det["bbox"],
                    "score": det["score"],
                    "text": text,
                })

        return results
