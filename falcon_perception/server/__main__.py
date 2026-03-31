#!/usr/bin/env python3
# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.
"""
Launch the Falcon Perception inference server.

Usage
-----
    # Defaults: auto-detect GPUs, port 7860
    python -m falcon_perception.server

    # Explicit config
    python -m falcon_perception.server --num-gpus 4 --port 8000 --compile --cudagraph

    # From a local model export
    python -m falcon_perception.server --hf-local-dir ./my_export/ --num-gpus 1
"""

import logging
import os
import signal

import tyro

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Workers run in separate processes (spawn).  Must be set before any CUDA
# or multiprocessing usage so that child processes get isolated CUDA contexts.
import multiprocessing as _mp

_mp.set_start_method("spawn", force=True)

# Ensure Ctrl-C works even when GPU worker threads are blocking in CUDA ops.
# First SIGINT: raise KeyboardInterrupt (uvicorn graceful shutdown).
# Second SIGINT: force-kill immediately.
_sigint_received = False


def _handle_sigint(signum, frame):
    global _sigint_received
    if _sigint_received:
        os._exit(1)
    _sigint_received = True
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, lambda s, f: os._exit(0))

from falcon_perception import setup_torch_config  # noqa: E402
from falcon_perception.server.config import ServerConfig  # noqa: E402

setup_torch_config()


def main(config: ServerConfig):
    import uvicorn

    from falcon_perception.server.app import create_app

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app = create_app(config)
    logger = logging.getLogger("falcon_perception.server")
    logger.info("Starting Falcon Perception server on %s:%d (hf_model=%s)", config.host, config.port, config.hf_model_id)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


tyro.cli(main)
