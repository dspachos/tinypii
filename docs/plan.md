# Client-Side Pure-Neural PII Redaction — Build Plan

Status: phases 1–4 complete — quality/footprint/limitations in docs/results.md

Note: this is the historical build plan. Script names below map to the current
layout as `generate_data.py` → `phase-1-generate-data/generate.py`,
`train_seq2seq.py` → `phase-2-train/finetune.py`, `evaluate.py` →
`phase-3-evaluate/evaluate.py`, `export.py` → `phase-4-export/export.py`, and
`index.html` → `web/index.html`.

## 1. Goal

Fine-tune a small generative seq2seq model (flan-t5-small) to redact PII from free
text entirely through neural inference, exported to ONNX INT8 and running 100%
client-side in the browser via Transformers.js v3.

**Hard constraint:** no regex / Luhn / string rules in the *deployed artifact*.
Training-time and data-QA tooling may use anything — the purity claim only covers
what ships to the browser.

## 2. Corrected baseline numbers (vs. original proposal)

| Metric | Claimed in proposal | Actual expectation |
|---|---|---|
| Params (flan-t5-small) | ~60M | **~77M** (60M is t5-small) |
| INT8 download | 25–30 MB | **~75–90 MB** (1 byte/param floor) |
| WASM latency | 150–300 ms | **~1–5 s** typical (128 tokens, CPU) |
| WebGPU latency | — | likely sub-second; must be **measured**, not assumed |

Decision rule: build with flan-t5-small first (supported, known-good in
Transformers.js). Only shrink to a custom ~25M-param T5 config **if** measured
size/latency fails acceptance criteria (§7). No speculative small-model work.

## 3. Tag set (fixed across all phases)

`[NAME]` `[CREDIT_CARD]` `[EMAIL]` `[PHONE]` `[ADDRESS]`

Task prefix `redact: ` used identically in training and inference.

## 4. Phases

### Phase 1 — Data generation (`generate_data.py`)

- Teacher LLM via structured outputs (instructor/JSON mode). Provider is a
  config value: OpenAI `gpt-4o-mini` default, local Ollama fallback if no key.
- Target **≥ 4,000 validated pairs** after cleaning (batches of 5–10, resumable,
  saved incrementally to JSONL).
- Diversity axes (rotate explicitly per batch so coverage is forced, not hoped
  for): scenario domain, card format (dashes/spaces/continuous/Amex 15-digit),
  international names, typos/slang, multiple entities per message, 0-PII
  messages (model must learn to copy untouched).
- **Programmatic QA gate on every pair** (this is the fix for trusting the
  teacher): strip tags from target → must reconstruct input exactly outside
  tag spans; tag inventory must match PII actually present (spot-check via
  validators/regex — allowed here, not shipped); dedupe near-identical inputs.
  Reject & log failures; keep a rejection-rate stat.
- Small template-based seed set (~200 pairs, same QA gate) for smoke-testing
  the training loop before spending on teacher tokens.
- Split: 90/10 train/test, **plus a frozen hand-written adversarial eval set**
  (~100 examples: odd separators, 15-digit cards, no-PII, very long text,
  adjacent entities like "Alice4111...").

### Phase 2 — Fine-tuning (`train_seq2seq.py`)

- flan-t5-small, Seq2SeqTrainer, `max_length=256` input / `128` target,
  lr ~3e-4–5e-4, 3–5 epochs, fp16 on GPU / off on CPU.
- **`compute_metrics` is mandatory**, on generated text:
  - exact match rate,
  - per-tag recall/precision (a missed tag = leak; recall is the headline number),
  - copy fidelity: % of non-PII characters preserved verbatim.
- Acceptance gate to proceed: tag recall ≥ 0.98 and copy fidelity ≥ 0.99 on the
  held-out test split **and** the frozen adversarial set. If not met: more data,
  more epochs, or more 0-PII copy examples — in that order.

### Phase 3 — Export & quantization (`export.py`)

- `optimum-cli export onnx --task text2text-generation-with-past`
- Dynamic INT8 on encoder + merged decoder (`ORTQuantizer`,
  `AutoQuantizationConfig` dynamic variant; record final per-file sizes).
- Verify the output filenames match what Transformers.js `dtype: 'q8'` expects
  (`*_quantized.onnx`); rename if optimum emits different names.
- Record total download size in the eval report (§7).

### Phase 4 — Browser client (`index.html`)

- Transformers.js **v3**: `pipeline('text2text-generation', './model', { dtype: 'q8', device: 'webgpu' })`
  with WASM fallback if WebGPU init fails (report which path is active in the UI).
- No `temperature` option — greedy is default and sufficient.
- **Post-generation verification guard** (pure string comparison, keeps the
  no-regex constraint): output with tags stripped must match input outside tag
  spans. Mismatch → visibly flag "unverified output", never silently show.
- **Overflow handling:** if input > ~200 tokens (headroom under the 256 cap),
  chunk by sentence and process sequentially, or warn + truncate visibly.
  Never present a truncated result as complete.
- Stream tokens as they generate (progress feedback during multi-second runs).
- Serve via `python -m http.server` (documented — `file://` cannot fetch models).

### Phase 5 — Evaluation report (`RESULTS.md`)

Measured, not estimated: download size, cold-load time, tokens/sec and
end-to-end latency on WASM and WebGPU (if available), tag recall on the
adversarial set, copy fidelity, verification-guard trip rate.

## 5. Risk register

| Risk | Mitigation |
|---|---|
| Silent missed redaction = PII leak | recall metric gate (Phase 2), verification guard (Phase 4), adversarial eval set |
| Model paraphrases/corrupts non-PII text | copy-fidelity metric + guard; 0-PII training examples |
| Teacher drifts from word-for-word rule | programmatic QA gate rejects bad pairs before training |
| 80 MB download too heavy | phase gate: measure first; custom small T5 only if needed |
| WASM latency unacceptable | WebGPU path; streaming UX; chunking |
| Input truncation drops unredacted tail | overflow handling with visible warning |

## 6. Non-goals

- Compliance certification (GDPR/HIPAA-grade guarantees) — this is a
  probabilistic system; the guard detects corruption, not all leaks.
- Regex fallback in the browser (violates the purity constraint).
- PII categories beyond the five tags (extendable later by regenerating data).

## 7. Go/no-go criteria (end of Phase 5)

Proceed to "done" if: total download ≤ ~90 MB, WASM end-to-end ≤ ~5 s or WebGPU
≤ ~1.5 s on a mid-range laptop, adversarial tag recall ≥ 0.98, copy fidelity
≥ 0.99. Otherwise decide: shrink model, raise data volume, or relax latency.
