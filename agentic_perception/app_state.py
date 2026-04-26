"""Shared application state for the agentic perception system."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AppState:
    bboxes: list[dict] = field(default_factory=list)
    # Each bbox: {"label": str, "bbox": [x1, y1, x2, y2]}

    zoom_region: tuple[int, int, int, int] | None = None
    # (x1, y1, x2, y2) crop region when zoomed

    last_frame: np.ndarray | None = None
    # Most recent camera frame (for sending to FP when user queries)

    chat_history: list[dict] = field(default_factory=list)
    # Ollama message format: [{"role": "user/assistant/tool", "content": ...}]

    def clear(self):
        """Reset all state to defaults."""
        self.bboxes = []
        self.zoom_region = None
        self.chat_history = []
        # Intentionally keep last_frame — camera is still running
