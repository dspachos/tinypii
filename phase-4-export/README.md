# Phase 4 — Export for the browser

Converts the fine-tuned checkpoint into the INT8 ONNX files that
Transformers.js v3 loads, and assembles `web/model/`.

This phase runs in a separate pinned virtualenv (`.venv-export`) because
optimum 1.x needs the transformers 4.x export stack. `make install` creates
both virtualenvs from `requirements.txt` and `requirements-export.txt`.

## Commands

Run from the repo root.

```
make onnx      # step 1: fp32 ONNX export -> models/onnx
make export    # step 2: INT8 quantize + assemble -> web/model (~93 MB total)
```

Then `make serve` and open http://127.0.0.1:8123.

## The two export traps

1. `tie_word_embeddings`. The phase 2 checkpoint saves it as `false` — the
   fine-tune leaves `shared` and `lm_head` diverged. If a checkpoint reaches
   this phase with the flag still `true`, optimum silently re-ties the
   weights and the exported decoder emits garbage. If you export a re-tied
   checkpoint manually, you lose the trained `lm_head` the same way.

2. Tokenizer files. Transformers.js v3 needs tokenizer files written by
   the transformers-4 stack. Files written by transformers 5.x decode
   without spaces in the browser (`paidwith[CREDIT_CARD]`) while Node still
   looks fine, so only a browser test catches it. `export.py` regenerates
   them from `google/flan-t5-small` inside `.venv-export` (the fine-tune
   does not change the vocab).

## Output

`web/model/`:

| File | Size |
|---|---|
| `onnx/encoder_model_quantized.onnx` | ~34 MB |
| `onnx/decoder_model_merged_quantized.onnx` | ~57 MB |
| tokenizer and config files | <1 MB |

Only the encoder and merged decoder graphs are quantized and shipped — those
are the only two Transformers.js v3 loads for text2text generation.
