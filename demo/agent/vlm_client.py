# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""OpenAI-compatible VLM client for the Falcon Perception Agent.

Handles:
- Converting PIL images embedded in messages to base64 data-URLs
- Sending multi-turn multimodal chat completion requests
- Returning the generated text string (or None on failure)

Works with any OpenAI-compatible endpoint: OpenAI, Azure, vLLM, Together AI,
or a self-hosted model served via the OpenAI API format.

Usage::

    client = VLMClient(api_key="sk-...", model="gpt-4o")

    # Messages follow the standard OpenAI format with one extension:
    # content items of type "pil_image" are auto-converted to base64.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "pil_image", "image": my_pil_image},
            {"type": "text", "text": "What is in this image?"},
        ]},
    ]
    response = client.send(messages)
    print(response)
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from .viz import pil_to_base64_url


class VLMClient:
    """Thin wrapper around the OpenAI chat-completions API.

    Parameters
    ----------
    api_key:
        OpenAI API key (or any API key accepted by the endpoint).
    model:
        Model name (e.g. ``"gpt-5-mini"``, ``"gpt-5-mini-2025-08-07"``).
    base_url:
        Override the API base URL for self-hosted / proxy endpoints.
        ``None`` uses the default OpenAI endpoint.
    max_tokens:
        Maximum tokens to generate per call (default: 2048).
    max_image_side:
        Images embedded in messages are downscaled so their longest side does
        not exceed this value before base64 encoding (reduces API cost).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        base_url: str | None = None,
        max_tokens: int = 2048,
        max_image_side: int = 1536,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install openai"
            ) from exc

        self.model = model
        self.max_tokens = max_tokens
        self.max_image_side = max_image_side
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, messages: list[dict[str, Any]]) -> str | None:
        """Send *messages* to the VLM and return the response text.

        Handles ``{"type": "pil_image", "image": PIL.Image}`` content items
        by converting them to ``image_url`` base64 data-URLs.

        Returns ``None`` on API failure (with a printed warning).
        """
        processed = self._prepare_messages(messages)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=processed,
                max_completion_tokens=self.max_tokens,
                n=1,
            )
            choices = response.choices
            if choices:
                return choices[0].message.content
            print("[VLMClient] Warning: empty choices in API response.")
            return None
        except Exception as exc:
            print(f"[VLMClient] API call failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert PIL images in messages to base64 image_url content items."""
        processed = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                processed.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        new_content.append(item)
                        continue

                    if item.get("type") == "pil_image":
                        img: Image.Image = item["image"]
                        url = pil_to_base64_url(img, max_side=self.max_image_side)
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": url, "detail": "high"},
                        })
                    else:
                        new_content.append(item)

                processed.append({"role": role, "content": new_content})
            else:
                processed.append(msg)

        return processed
