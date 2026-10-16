"""Where the local Nonescape ONNX model lives.

The runtime (`image_api.py`) and the evaluation harness (`eval/image_eval.py`)
resolve the model path through this one function so a locally trained
checkpoint can be swapped in without editing code.

Set ``NONESCAPE_MODEL`` to point at a different ONNX file, e.g. the head
re-trained by ``bin/probe-nonescape.py``:

    set NONESCAPE_MODEL=nonescape-mini-v1.onnx        (Windows PowerShell)

Default stays `nonescape-mini-v0.onnx` (the upstream model) because it is the
only checkpoint available to a fresh clone — see ``docs/eval-image.md`` for the
measured trade-off between v0 and a locally trained head.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "nonescape-mini-v0.onnx"
ENV_VAR = "NONESCAPE_MODEL"


def resolve_model_path(env: dict | None = None) -> str:
    """Return the ONNX path to load (env override wins, else the default)."""
    source = os.environ if env is None else env
    return source.get(ENV_VAR) or DEFAULT_MODEL
