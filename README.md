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

All numbers measured, none estimated. Full details in [docs/results.md](docs/results.md).

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

## Repository layout

```
phase-1-generate-data/   training pairs from templates or a teacher LLM, QA-gated
phase-2-train/           flan-t5-small fine-tune with recall/fidelity metrics
phase-3-evaluate/        acceptance gate: held-out split + frozen adversarial set
phase-4-export/          ONNX export + INT8 quantization into web/model/
web/                     browser client (index.html + generated model in web/model/)
eval/                    frozen hand-written adversarial set
test/                    node smoke, parity, and headless browser tests
docs/                    historical build plan and measured results
```

`data/` and `models/` are generated outputs, gitignored. Each phase folder has its own README
with prerequisites, commands, and cost notes.

## Pipeline at a glance

| Phase | Folder | Make target | Output |
|---|---|---|---|
| 1 · data | `phase-1-generate-data/` | `make seed`, `make data` | `data/*.jsonl` |
| 2 · train | `phase-2-train/` | `make smoke`, `make train` | `models/flan-t5-small-pii/` |
| 3 · evaluate | `phase-3-evaluate/` | `make evaluate` | go/no-go verdict |
| 4 · export | `phase-4-export/` | `make onnx`, `make export` | `web/model/` |
| serve | `web/` | `make serve` | http://127.0.0.1:8123 |

## Quickstart

The repo does not ship model weights or training data — both are regenerable through the
pipeline. After phases 1–4, or with your own `web/model/` in place:

```bash
make install   # once: both python virtualenvs + node dev dependencies
make serve     # http://127.0.0.1:8123
```

The page needs HTTP; `file://` cannot fetch the model files.

## Rebuild from scratch

```bash
make install
make seed          # free template pairs; enough for `make smoke`
make data          # teacher-LLM pairs — needs .env.local, costs API tokens
make smoke         # validates the training loop in minutes
make train         # full fine-tune
make evaluate      # acceptance gate
make onnx          # fp32 ONNX export (runs in .venv-export)
make export        # INT8 quantization -> web/model/
make serve
```

Node/browser tests: `make test` and `make test-browser` (needs `make serve` running).

Details, prerequisites, and expected outputs per phase: the README in each phase folder.

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

## License

MIT — see [LICENSE](LICENSE).
