# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Core agent loop for the Falcon Perception Agent.

The agent alternates between:
  1. Sending the current message history to the VLM (GPT-4o or any
     OpenAI-compatible multimodal model).
  2. Parsing the ``<tool>`` call from the response.
  3. Executing the tool (FP inference, crop extraction, or relation computation).
  4. Appending the tool result back to the message history.

The loop terminates when the VLM calls the ``answer`` tool.

Context management
------------------
To keep the context window compact (and API costs low), at most one
Set-of-Marks image from a ``ground_expression`` call is kept in the message
history at a time — earlier ones are pruned when a new one arrives.
Additionally, the last *max_tail* non-image messages (get_crop, compute_relations)
are retained for in-round context.  The system message and original user image
are always preserved.

Usage::

    from demo.agent import run_agent, VLMClient

    client = VLMClient(api_key="sk-...", model="gpt-5-mini")
    result  = run_agent(image, "Which person is closer to the camera?",
                        fp_engine, tokenizer, client)
    print(result.answer)
    result.final_image.show()
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .fp_tools import compute_relations, masks_to_vlm_json, run_ground_expression
from .viz import get_crop, pil_to_base64_url, render_final, render_som

# ---------------------------------------------------------------------------
# System prompt loader
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.txt"


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Output of a completed agent run.

    Attributes
    ----------
    answer:
        The VLM's free-text response to the user query.
    supporting_mask_ids:
        Mask IDs from the last ``ground_expression`` call that the VLM
        selected as visual evidence for its answer.
    final_image:
        The original image with only the supporting masks rendered on top.
        ``None`` if no masks were selected.
    history:
        Full message history for inspection / debugging.
    n_fp_calls:
        Number of times Falcon Perception was invoked.
    n_vlm_calls:
        Number of times the VLM was called.
    """
    answer: str
    supporting_mask_ids: list[int] = field(default_factory=list)
    final_image: Image.Image | None = None
    history: list[dict] = field(default_factory=list)
    n_fp_calls: int = 0
    n_vlm_calls: int = 0


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------

_TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)


def _parse_tool_call(text: str) -> dict | None:
    """Extract and parse the JSON inside the first ``<tool>…</tool>`` block."""
    m = _TOOL_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    # Models occasionally emit an extra closing brace (e.g. `}}}` instead of `}}`).
    raw = raw.replace("}}}", "}}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

def _count_images(messages: list[dict]) -> int:
    """Count ``pil_image`` and ``image_url`` content items across all messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("pil_image", "image_url"):
                total += 1
    return total


def _has_som_image(msg: dict) -> bool:
    """Return True if *msg* is a user message that contains a SoM rendered image."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content", [])
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and item.get("type") == "pil_image"
        for item in content
    )


def _prune_context(messages: list[dict]) -> list[dict]:
    """Keep the message history compact.

    Strategy:
      - Always keep ``messages[0]`` (system) and ``messages[1]`` (original
        user message with the input image).
      - Keep the last ``ground_expression`` assistant + user pair.
      - Keep all subsequent ``get_crop`` / ``compute_relations`` messages.
      - Discard everything in between.
    """
    if len(messages) <= 4:
        return messages

    head = messages[:2]        # [system, original_user]
    tail = messages[2:]        # everything after

    # Find the last user message that contains a SoM image (ground_expression result)
    last_som_idx = -1
    for i, msg in enumerate(tail):
        if _has_som_image(msg):
            last_som_idx = i

    if last_som_idx == -1:
        # No SoM image yet; keep last 10 tail messages to avoid unbounded growth
        return head + tail[-10:]

    # Keep the assistant message right before the SoM (the ground_expression call)
    # + the SoM user message itself + everything after it.
    som_start = max(0, last_som_idx - 1)
    pruned_tail = tail[som_start:]

    # Safety: cap total image count at 2 (original + 1 SoM)
    # If get_crop introduced additional images beyond that, remove them.
    images_in_tail = sum(
        1 for msg in pruned_tail
        if _has_som_image(msg) or (
            isinstance(msg.get("content"), list)
            and any(
                isinstance(item, dict) and item.get("type") == "pil_image"
                and msg.get("role") == "user"
                for item in msg.get("content", [])
            )
        )
    )
    # Trim oldest crop images if needed (keep at most 1 extra image in tail)
    if images_in_tail > 1:
        trimmed: list[dict] = []
        extra_count = 0
        for msg in pruned_tail:
            if (
                msg.get("role") == "user"
                and isinstance(msg.get("content"), list)
                and any(
                    isinstance(it, dict) and it.get("type") == "pil_image"
                    for it in msg["content"]
                )
                and not _has_som_image(msg)   # not the SoM — it's a crop
            ):
                extra_count += 1
                if extra_count > 1:
                    continue   # discard older crops
            trimmed.append(msg)
        pruned_tail = trimmed

    return head + pruned_tail


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_baseline(
    image: Image.Image,
    query: str,
    vlm_client,
) -> str | None:
    """Direct VLM answer with no Falcon Perception grounding.

    Sends the image and query straight to the VLM (no tools, no system prompt
    about segmentation).  Use this alongside :func:`run_agent` to measure how
    much grounded reasoning improves over a plain visual QA baseline.

    Returns the VLM's response string, or ``None`` on API failure.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful visual assistant. "
                "Answer the user's question about the image concisely and accurately."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "pil_image", "image": image.convert("RGB")},
                {"type": "text", "text": query},
            ],
        },
    ]
    return vlm_client.send(messages)


def run_agent(
    image: Image.Image,
    query: str,
    fp_engine,
    tokenizer,
    vlm_client,
    *,
    max_generations: int = 20,
    fp_max_new_tokens: int = 2048,
    fp_max_dimension: int = 1024,
    fp_hr_upsample_ratio: int = 8,
    verbose: bool = True,
) -> AgentResult:
    """Run the Falcon Perception Agent on *image* with *query*.

    Parameters
    ----------
    image:
        Input PIL image.
    query:
        Natural-language query / task for the agent.
    fp_engine:
        Initialised ``PagedInferenceEngine`` (created once by the caller and
        reused across calls).
    tokenizer:
        Matching tokenizer for *fp_engine*.
    vlm_client:
        :class:`~demo.agent.vlm_client.VLMClient` instance (wraps GPT-4o or
        any OpenAI-compatible VLM endpoint).
    max_generations:
        Hard cap on VLM calls before the loop raises (default: 20).
    fp_max_new_tokens:
        Token budget per FP inference call.
    fp_max_dimension:
        Longest-edge cap for images fed to FP (pixels).
    fp_hr_upsample_ratio:
        HR upsampling ratio for FP segmentation heads.
    verbose:
        Print per-step summaries to stdout.

    Returns
    -------
    :class:`AgentResult`
    """
    system_prompt = _load_system_prompt()
    pil_image = image.convert("RGB")

    # ── Initialise message history ─────────────────────────────────────────
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "pil_image", "image": pil_image},
                {"type": "text", "text": f"User query: {query}"},
            ],
        },
    ]

    current_masks: dict[int, dict] = {}   # mask_id → metadata (includes rle)
    n_fp_calls = 0
    n_vlm_calls = 0

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Falcon Perception Agent")
        print(f"  Query: {query!r}")
        print(f"{'─' * 60}")

    for step in range(max_generations):

        # ── VLM call ──────────────────────────────────────────────────────
        if verbose:
            print(f"\n[Step {step + 1}] Calling VLM ...")
        response_text = vlm_client.send(messages)
        n_vlm_calls += 1

        if response_text is None:
            raise RuntimeError(
                f"VLM returned None at step {step + 1}. "
                "Check your API key and endpoint configuration."
            )

        if verbose:
            # Show only the <think> preview (first 300 chars) and the tool call
            think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
            if think_match:
                think_preview = think_match.group(1).strip()[:300].replace("\n", " ")
                print(f"  [think] {think_preview}{'...' if len(think_match.group(1)) > 300 else ''}")
            tool_match = _TOOL_RE.search(response_text)
            if tool_match:
                print(f"  [tool]  {tool_match.group(1).strip()[:200]}")

        # ── Parse tool call ───────────────────────────────────────────────
        tool_call = _parse_tool_call(response_text)
        if tool_call is None:
            raise ValueError(
                f"Could not parse a <tool> tag from VLM response at step {step + 1}.\n"
                f"Response: {response_text[:500]}"
            )

        tool_name = tool_call.get("name", "")
        params = tool_call.get("parameters", {})

        # ── Execute tool ──────────────────────────────────────────────────

        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
        )

        if tool_name == "ground_expression":
            expression = params.get("expression", "")
            if verbose:
                print(f"  → ground_expression({expression!r})")

            current_masks = run_ground_expression(
                fp_engine,
                tokenizer,
                pil_image,
                expression,
                max_new_tokens=fp_max_new_tokens,
                hr_upsample_ratio=fp_hr_upsample_ratio,
                max_dimension=fp_max_dimension,
            )
            n_fp_calls += 1

            n_masks = len(current_masks)
            if verbose:
                print(f"     → {n_masks} mask(s) returned")

            if n_masks == 0:
                tool_result_content: list[dict] = [
                    {
                        "type": "text",
                        "text": (
                            f"ground_expression({expression!r}) returned 0 masks. "
                            "Try a different, more general expression."
                        ),
                    }
                ]
            else:
                som_image = render_som(pil_image, current_masks)
                meta_json = json.dumps(
                    {"n_masks": n_masks, "masks": masks_to_vlm_json(current_masks)},
                    indent=2,
                )
                tool_result_content = [
                    {"type": "pil_image", "image": som_image},
                    {
                        "type": "text",
                        "text": (
                            f"ground_expression returned {n_masks} mask(s). "
                            f"The Set-of-Marks image is shown above.\n\n"
                            f"Mask metadata:\n{meta_json}"
                        ),
                    },
                ]

            messages.append({"role": "user", "content": tool_result_content})

        elif tool_name == "get_crop":
            mask_id = int(params.get("mask_id", -1))
            if verbose:
                print(f"  → get_crop(mask_id={mask_id})")

            if mask_id not in current_masks:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": (
                        f"get_crop failed: mask_id={mask_id} does not exist. "
                        f"Available IDs: {sorted(current_masks.keys())}"
                    )}],
                })
            else:
                crop_img = get_crop(pil_image, current_masks[mask_id])
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "pil_image", "image": crop_img},
                        {"type": "text", "text": f"Zoomed crop of mask {mask_id}."},
                    ],
                })

        elif tool_name == "compute_relations":
            mask_ids = params.get("mask_ids", [])
            if verbose:
                print(f"  → compute_relations(mask_ids={mask_ids})")

            relations = compute_relations(current_masks, mask_ids)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"compute_relations result:\n"
                            + json.dumps(relations, indent=2)
                        ),
                    }
                ],
            })

        elif tool_name == "answer":
            response_text_final = params.get("response", "")
            selected_ids = [int(i) for i in params.get("supporting_mask_ids", [])]

            if verbose:
                print(f"\n{'─' * 60}")
                print(f"  Answer: {response_text_final}")
                print(f"  Supporting masks: {selected_ids}")
                print(f"  FP calls: {n_fp_calls}  |  VLM calls: {n_vlm_calls}")
                print(f"{'─' * 60}\n")

            final_image = (
                render_final(pil_image, current_masks, selected_ids)
                if selected_ids and current_masks
                else pil_image.copy()
            )

            return AgentResult(
                answer=response_text_final,
                supporting_mask_ids=selected_ids,
                final_image=final_image,
                history=messages,
                n_fp_calls=n_fp_calls,
                n_vlm_calls=n_vlm_calls,
            )

        else:
            raise ValueError(
                f"Unknown tool '{tool_name}' at step {step + 1}. "
                "Expected one of: ground_expression, get_crop, compute_relations, answer."
            )

        # ── Context pruning ───────────────────────────────────────────────
        messages = _prune_context(messages)

    raise RuntimeError(
        f"Agent exceeded max_generations={max_generations} without calling 'answer'."
    )
