# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.

"""Falcon Perception Agent — grounded visual reasoning via tool-use.

The agent uses GPT-4o (or any OpenAI-compatible VLM) as an orchestrator and
Falcon Perception as a grounded segmentation tool.  At each step the VLM
calls one of four tools:

    ground_expression  — run Falcon Perception on a referring expression
    get_crop           — zoom into a specific mask for verification
    compute_relations  — compute pairwise spatial metrics between masks
    answer             — return the final answer and terminate

Usage::

    from demo.agent import run_agent, VLMClient

    client = VLMClient(api_key="sk-...", model="gpt-5-mini")
    result = run_agent(image, "Which person is closer to the camera?", engine, tokenizer, client)
    print(result.answer)
    result.final_image.show()
"""

from .agent_loop import AgentResult, run_agent, run_baseline
from .vlm_client import VLMClient

__all__ = ["run_agent", "run_baseline", "AgentResult", "VLMClient"]
