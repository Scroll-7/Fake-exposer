"""Convert the Nonescape mini safetensors checkpoint to ONNX — offline, zero cost.

Why this exists
---------------
`bin/download-models.ps1` used to fall back to the HuggingFace *safetensors*
file but saved it under the name `nonescape-mini-v0.onnx`. onnxruntime then
fails with ``INVALID_PROTOBUF: Protobuf parsing failed`` and the local image
detector silently does nothing.

There is no prebuilt ONNX available: the original CDN now returns 403 and the
HuggingFace repo only ships ``.safetensors``. So we reconstruct the exact
architecture defined by the upstream Python SDK (``python/nonescape/__init__.py``)
and export ONNX ourselves.

Architecture (NonescapeClassifierMini):
    backbone = torchvision efficientnet_v2_s(num_classes=1024)
    head     = nn.Linear(1024, 2)
    forward  = head(backbone(x)) -> raw logits [B, 2]

The exported graph returns **raw logits** (no softmax) because both consumers
(`image_api.py` and `eval/image_eval.py`) apply softmax themselves and take
index 1 as P(AI).

Usage:
    python bin/convert-nonescape.py
    python bin/convert-nonescape.py --input nonescape-mini-v0.safetensors
"""

from __future__ import annotations

import argparse
import os
import sys


def build_model(embedding_size: int = 1024, num_classes: int = 2, dropout: float = 0.2):
    import torch
    import torch.nn as nn
    import torchvision.models as models

    class NonescapeClassifierMini(nn.Module):
        """Mirrors upstream NonescapeClassifierMini (logits, not softmax)."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = models.efficientnet_v2_s(
                weights=None, num_classes=embedding_size, dropout=dropout
            )
            self.head = nn.Linear(embedding_size, num_classes)

        def forward(self, x):  # noqa: D102
            return self.head(self.backbone(x))

    return NonescapeClassifierMini()


def convert(input_path: str, output_path: str, opset: int = 17) -> None:
    import torch
    from safetensors.torch import load_file

    if not os.path.exists(input_path):
        raise SystemExit(f"[!] Input checkpoint not found: {input_path}")

    model = build_model()
    state_dict = load_file(input_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"[!] Checkpoint does not match architecture.\n"
            f"    missing={missing[:5]}... ({len(missing)})\n"
            f"    unexpected={unexpected[:5]}... ({len(unexpected)})"
        )
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
    )
    print(f"[+] Exported ONNX -> {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


def verify(onnx_path: str) -> bool:
    try:
        import numpy as np
        import onnxruntime as ort
        import torch  # noqa: F401  (only used to keep the same dummy convention)
    except ImportError:
        return False

    session = ort.InferenceSession(onnx_path)
    shape = session.get_inputs()[0].shape
    noise = np.random.default_rng(0).random((1, 3, 224, 224), dtype=np.float32)
    out = session.run(None, {session.get_inputs()[0].name: noise})[0]
    print(f"[+] onnxruntime loaded OK — input={shape} output_shape={out.shape}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Nonescape mini -> ONNX converter.")
    parser.add_argument("--input", default="nonescape-mini-v0.safetensors")
    parser.add_argument("--output", default="nonescape-mini-v0.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)

    convert(args.input, args.output, opset=args.opset)
    if not args.no_verify:
        verify(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
