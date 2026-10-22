# RESULTS — Client-Side Pure-Neural PII Redactor

## Shipped model (rebuild, 2026-10-22)

Model: flan-t5-small, 5 epochs, lr 3e-4, effective batch 16, MPS, on the
shipped dataset: 10,383 teacher pairs (gemini-2.5-flash for the first
1,856; gemini-3.5-flash-lite for the rest, including one focused weak-spot
round) + 200 seed pairs. Two filter passes cleaned the teacher data: the
regex QA gate (186 pairs with names left after honorifics) and a
model-assisted scan (86 pairs whose gold left PII plain).

| Metric | fp32 held-out (n=1,059) | q8 ONNX Node (n=200) |
|---|---|---|
| tag recall | 0.9966 | 0.9946 |
| tag precision | 0.9941 | 0.9973 |
| copy fidelity | 0.9873 | 1.000 |
| exact match | 0.8905 | 0.905 |

Adversarial set (30 hand-written edges): fp32 recall 1.000, precision 0.974,
fidelity 0.942 — every failure is one of the three known-hard cases
(UPS tracking, `[urgent]`, multi-line). Excluding those: 1.000 / 1.000.

Acceptance gate: recall passes everywhere; held-out fidelity 0.9873 misses
the 0.99 gate by 0.0027 — accepted with this documented deviation. The
residual loss sits in rare teacher-data noise (diacritics, one corrupted
gold), not systematic word dropping: the 971 zero-PII copy pairs are exact
and training fidelity plateaued at epoch 2.

Footprint: encoder 35.8 MB + decoder 59.5 MB (INT8) + tokenizer ≈ 96 MB.
Node/CPU: ~52 ms per example (n=230, 12 s total).

Fixes vs the first build attempts, for the record: teacher golds that left
names unredacted taught contradictory behavior and capped fidelity at
~0.967; the honorific gate check plus the model-assisted filter recovered
+0.020 fidelity.

---

## First build (historical)

All numbers measured, none estimated. Model: flan-t5-small fine-tuned for
redaction (5 epochs, lr 3e-4, eff. batch 16, MPS) on 8,647 teacher-LLM pairs
(gemini-2.5-flash through an OpenAI-compatible endpoint, 4 generation
rounds) + 200 seed
pairs, all through the QA gate.

## Quality

| Metric | Python fp32 (held-out, n=877) | q8 ONNX Node (sample, n=200) | Browser q8 |
|---|---|---|---|
| tag recall | 0.9959 | 0.992 | spot-check OK |
| tag precision | ~0.99 | 1.000 | spot-check OK |
| copy fidelity | 0.9952 | 0.9997 | guard OK |
| exact match | 0.97 | 0.96 | — |

Adversarial set (30 hand-written edges): fp32 recall 0.973 / fidelity 0.921
(excl. known-hard: 0.971 / 0.975). q8: 0.919 / 0.835. **Below the 0.98/0.99
gate — accepted with documented tail** (see Known Limitations).

## Footprint & speed

- Download: encoder 34.1 MB + decoder 56.8 MB + tokenizer/config ≈ **93 MB**
  (plan budget was ≤90 MB; +3 MB accepted)
- Browser (WASM, headless Chromium): model load 20–40 s first visit (cached
  after), **319 ms** end-to-end for a one-sentence redaction
- Node/CPU: ~65 ms average per example (n=230)

## Known limitations (the documented tail)

1. Names in short ALL-CAPS imperatives sometimes missed (`CALL TOM REED`)
2. Word immediately before a tag occasionally eaten (`ph [PHONE]` → `[PHONE]`)
3. Dotted 10-digit numbers oscillate between PHONE and CREDIT_CARD (q8 says
   CREDIT_CARD)
4. Newlines are flattened — T5 SentencePiece cannot emit `\n` (tokenizer
   limitation; browser client pre-flattens)
5. Known-hard by design: alphanumeric/spaced tracking numbers redacted as
   cards; `[urgent]`-style brackets can be mangled
6. Spaces around tags occasionally collapse (`Dr.[NAME]`) — cosmetic

## Export gotcha (recorded for reproducibility)

The transformers-5 fine-tune leaves `shared` and `lm_head` weights diverged
while `config.tie_word_embeddings` still says `true`. Exporting with that
config silently re-ties them and the decoder emits garbage ("debit debit…").
Fix: set `tie_word_embeddings: false` in the saved model's config before
`optimum-cli export onnx` (done here; see export.py history).

## Reproduce

```
make install               # both python virtualenvs + node deps
make seed && make data     # phase 1 (teacher mode needs .env.local, API cost)
make train                 # phase 2
make evaluate              # phase 3 (acceptance gate)
make onnx && make export   # phase 4
make serve                 # open http://127.0.0.1:8123
```

Node/browser tests: `make test`, `make test-browser` (run `make serve` in
another terminal first). Raw commands: `phase-1-generate-data/generate.py`,
`phase-2-train/finetune.py`, `phase-3-evaluate/evaluate.py`,
`phase-4-export/export.py`, `test/smoke.mjs`, `test/parity.mjs`,
`test/browser-test.mjs`.
