# tinypii

Client-side pure-neural PII redaction in the browser.

A fine-tuned [flan-t5-small](https://huggingface.co/google/flan-t5-small) reads free text and
returns it with personally identifiable information replaced by tags — entirely through neural
inference. The model ships as INT8 ONNX and runs with
[Transformers.js v3](https://huggingface.co/docs/transformers.js) on WebGPU, with a WASM
fallback. No regex, no Luhn check, no string rules in the deployed artifact. No network calls
after the model loads.

```
redact: Hi, I am Marcus Vance. Charge 4532-1182-9920-1092 and reach me at contact@acme.org.

→ Hi, I am [NAME]. Charge [CREDIT_CARD] and reach me at [EMAIL].
```

Tag set: `[NAME]` `[CREDIT_CARD]` `[EMAIL]` `[PHONE]` `[ADDRESS]`

## Why

PII detection tools usually ship pattern matchers to the client. Pattern matchers are easy to
audit but blind to anything the patterns did not anticipate. A neural model generalizes —
typos, unusual card formats, international names — but neural output cannot be trusted
blindly either. This project tests a middle path:

1. A small generative model learns to **copy** the input and replace PII spans with tags.
2. Every training pair passes a programmatic QA gate before it is used.
3. A **verification guard** in the browser checks that the output reproduces the input
   verbatim outside the tag spans. If it does not, the UI flags the output as unverified
   instead of showing it as safe.

The guard is pure string comparison. It detects corruption of non-PII text, not every missed
redaction. This is a probabilistic system, not a compliance guarantee (see
[Limitations](#limitations)).

## Results

All numbers measured, none estimated. Full details in [RESULTS.md](RESULTS.md).

| Metric | Python fp32 (n=877) | q8 ONNX Node (n=200) | Browser q8 |
|---|---|---|---|
| Tag recall | 0.996 | 0.992 | spot-check OK |
| Tag precision | ~0.99 | 1.000 | spot-check OK |
| Copy fidelity | 0.995 | 0.9997 | guard OK |
| Exact match | 0.97 | 0.96 | — |

Footprint and speed:

| Metric | Value |
|---|---|
| Download | ~93 MB (INT8), cached after first visit |
| End-to-end latency, WASM, one sentence | 319 ms |
| Latency, Node/CPU per example | ~65 ms |
| Cold model load in browser | 20–40 s first visit |

Training data: 8,647 teacher-LLM pairs (gemini-2.5-flash, generated through a QA gate) plus
200 template-based seed pairs.

## Quickstart

You need the exported model directory, which is not committed (see
[Rebuild from scratch](#rebuild-from-scratch)). With `web_llm_model/` present:

```bash
python3 -m http.server 8123
# open http://127.0.0.1:8123
```

The page needs HTTP; `file://` cannot fetch the model files.

## Repo layout

| Path | Role |
|---|---|
| `generate_data.py` | Teacher-LLM pair generation with programmatic QA gate |
| `train_seq2seq.py` | Fine-tune flan-t5-small with recall/fidelity metrics |
| `evaluate.py` | Acceptance-gate evaluation on held-out and adversarial sets |
| `export.py` | ONNX export + INT8 quantization into `web_llm_model/` |
| `index.html` | Browser client: streaming output, verification guard, chunking |
| `eval/adversarial.json` | Frozen hand-written edge-case set |
| `test/smoke.mjs` | Node smoke test for the ONNX model |
| `test/parity.mjs` | Checks that Python and Node ONNX outputs agree |
| `test/browser-test.mjs` | Headless Chromium run of `index.html` |

Data (`data/`), model checkpoints, and ONNX artifacts are gitignored — they are regenerable.

## Rebuild from scratch

### 1. Data

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# free template-based seed pairs (smoke-test the training loop first)
.venv/bin/python generate_data.py seed --target 200

# teacher-generated pairs; costs API tokens
.venv/bin/python generate_data.py teacher --target 8647
# weak-spot regeneration mode:
.venv/bin/python generate_data.py teacher --focus
```

Every pair passes a QA gate: tags stripped from the target must reconstruct the input
exactly outside tag spans, and the tag inventory must match validators run on the raw input
(validators are training-time tooling — they never ship). Failed pairs are logged and
rejected.

### 2. Train

```bash
.venv/bin/python train_seq2seq.py --smoke        # seed data, minutes
.venv/bin/python train_seq2seq.py --epochs 5 --batch 8 --grad-accum 2
```

### 3. Evaluate the acceptance gate

```bash
.venv/bin/python evaluate.py            # optional: --beams 5 for beam search
```

Gate: tag recall ≥ 0.98 and copy fidelity ≥ 0.99 on the held-out split and the adversarial
set.

### 4. Export and quantize

```bash
python3.13 -m venv .venv-export
.venv-export/bin/pip install 'torch==2.7.1' 'transformers==4.51.3' 'optimum[exporters]<2' 'onnx<1.18' 'onnxruntime<1.22'
.venv-export/bin/optimum-cli export onnx --model ./saved_generative_pii_model --task text2text-generation-with-past ./onnx_generative
.venv-export/bin/python export.py
```

Then serve as in the quickstart.

### Export trap

The fine-tune (transformers 5.x) leaves `shared` and `lm_head` weights diverged while
`config.tie_word_embeddings` still says `true`. Exporting with that config silently re-ties
them and the decoder emits garbage. Fix: set `tie_word_embeddings: false` in the saved
model's config before the ONNX export.

## Limitations

Measured on a frozen adversarial set (fp32 recall 0.973 / q8 recall 0.919 — below the 0.98
gate, accepted with a documented tail):

- Names in short ALL-CAPS imperatives are sometimes missed (`CALL TOM REED`).
- The word immediately before a tag is occasionally eaten (`ph [PHONE]` → `[PHONE]`).
- Dotted 10-digit numbers oscillate between PHONE and CREDIT_CARD.
- Newlines are flattened — the T5 SentencePiece tokenizer cannot emit `\n`. The browser
  client pre-flattens input.
- Alphanumeric and spaced tracking numbers get redacted as cards (known-hard by design).
- Spaces around tags occasionally collapse (`Dr.[NAME]`) — cosmetic.

Non-goals: compliance certification, a regex fallback in the browser, and PII categories
beyond the five tags.
