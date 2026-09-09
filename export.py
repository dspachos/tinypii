#!/usr/bin/env python3
"""Phase 3 — quantize the ONNX export to INT8 and assemble web_llm_model/.

Run inside .venv-export (optimum 1.x + transformers 4.x stack):
    .venv-export/bin/python export.py

Quantizes encoder_model and decoder_model_merged (the only two graphs
Transformers.js v3 loads for text2text) and copies the tokenizer/config
files needed client-side.
"""
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

SRC = Path("onnx_generative")
DST = Path("web_llm_model")
GRAPHS = ("encoder_model", "decoder_model_merged")

# Dynamic INT8, QDQ format with per-channel weights, embeddings excluded —
# the same recipe the xenova/* community models use for ORT-Web.
qconfig = AutoQuantizationConfig.avx2(is_static=False)

for name in GRAPHS:
    quantizer = ORTQuantizer(SRC / f"{name}.onnx")
    quantizer.quantize(save_dir=str(DST / "onnx"), quantization_config=qconfig)
    print(f"quantized {name}")

for fname in ("config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json"):
    src = SRC / fname
    if src.exists():
        shutil.copy(src, DST / fname)
        print(f"copied {fname}")

for f in sorted((DST / "onnx").glob("*.onnx")):
    print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")
