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

| Metric | Python fp32, held-out (n=1,059) | q8 ONNX Node (n=200) | Browser q8 |
|---|---|---|---|
| Tag recall | 0.9966 | 0.9946 | spot-check OK |
| Tag precision | 0.9941 | 0.9973 | spot-check OK |
| Copy fidelity | 0.9873 | 1.000 | guard OK |
| Exact match | 0.89 | 0.905 | — |

Adversarial set (30 hand-written edge cases): recall 1.000, and 1.000 / 1.000
recall / fidelity on every case except the three documented known-hard ones
(UPS tracking numbers, `[urgent]` brackets, multi-line text).

Footprint and speed:

| Metric | Value |
|---|---|
| Download | ~96 MB (INT8), cached after first visit |
| Latency, Node/CPU per example | ~52 ms |
| End-to-end latency, browser WASM, one sentence | ~300 ms |
| Cold model load in browser | 20–40 s first visit |

Training data: 10,383 QA-gated teacher pairs plus 200 template seed pairs,
shipped in [`data/`](data/README.md).

## Repository layout

```
phase-1-generate-data/   training pairs from templates or a teacher LLM, QA-gated
phase-2-train/           flan-t5-small fine-tune with recall/fidelity metrics
phase-3-evaluate/        acceptance gate: held-out split + frozen adversarial set
phase-4-export/          ONNX export + INT8 quantization into web/model/
web/                     browser demo (index.html + shipped INT8 model in web/model/)
examples/                minimal embed example for your own page
eval/                    frozen hand-written adversarial set
test/                    node smoke, parity, and headless browser tests
docs/                    historical build plan and measured results
data/                    shipped training pairs
```

`models/` is a generated output, gitignored. The QA-gated training pairs ship in
[`data/`](data/README.md), so retraining needs no API access. Each phase folder has its own
README with prerequisites, commands, and cost notes.

## Pipeline at a glance

| Phase | Folder | Make target | Output |
|---|---|---|---|
| 1 · data | `phase-1-generate-data/` | `make seed`, `make data` | `data/*.jsonl` |
| 2 · train | `phase-2-train/` | `make smoke`, `make train` | `models/flan-t5-small-pii/` |
| 3 · evaluate | `phase-3-evaluate/` | `make evaluate` | go/no-go verdict |
| 4 · export | `phase-4-export/` | `make onnx`, `make export` | `web/model/` |
| serve | `web/` | `make serve` | http://127.0.0.1:8123/web/index.html |

## Quickstart

The repo ships the trained INT8 model and the training data. To try it:

```bash
make serve   # then open http://127.0.0.1:8123/web/index.html
```

The page needs HTTP; `file://` cannot fetch the model files. The minimal embed
example lives at `http://127.0.0.1:8123/examples/minimal.html`.

## Use it in your own page

A minimal integration — model load plus one redaction call — lives in
[examples/minimal.html](examples/minimal.html). It needs `web/model/` from
phase 4, served over HTTP.

For real use, also copy the verification guard and the chunking from
[web/index.html](web/index.html). The guard flags outputs where the model
altered text outside the redaction tags.

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

Measured behavior of the shipped model:

- Newlines are flattened — the T5 SentencePiece tokenizer cannot emit `\n`. The browser
  client pre-flattens input.
- Tracking numbers (UPS-style) and bracketed words like `[urgent]` are sometimes
  redacted as PII (known-hard by design; outside the training distribution).
- Words with diacritics may come back normalized (`NÓW` → `NOW`).
- Held-out copy fidelity is 0.9873, below the 0.99 gate — an accepted, documented
  deviation. The residual loss sits in rare teacher-data noise, not systematic word
  dropping: the 971 zero-PII copy pairs in the dataset are exact, and the q8 sample
  scores 1.000 fidelity.
- Spaces around tags occasionally collapse (`Dr.[NAME]`) — cosmetic.

Non-goals: compliance certification, a regex fallback in the browser, and PII categories
beyond the five tags.

## License

MIT — see [LICENSE](LICENSE).
