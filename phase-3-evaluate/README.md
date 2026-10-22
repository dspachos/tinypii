# Phase 3 — Acceptance gate

Scores the saved model on two sets and prints a go/no-go verdict:

- the held-out test split, rebuilt with the same seed as training
- the frozen adversarial set, `eval/adversarial.json` — 30 hand-written edge
  cases (odd separators, 15-digit cards, no-PII messages, adjacent entities)

## Prerequisites

- A trained model at `models/flan-t5-small-pii` (phase 2)
- `data/synthetic.jsonl` and `data/seed.jsonl` (phase 1), to rebuild the split

## Command

Run from the repo root.

```
make evaluate
# or directly:
.venv/bin/python phase-3-evaluate/evaluate.py [--model models/flan-t5-small-pii] [--beams 5]
```

## The gate

GO requires, on both sets: tag recall at 0.98 or higher and copy fidelity at
0.99 or higher.

Known-hard cases (UPS tracking numbers, `[urgent]` brackets, multi-line
text) are scored but excluded from the gate math, because they sit outside
the training distribution on purpose.

## Reading the verdict

The published model passes the held-out gate. On the adversarial set it
scored below the gate and was accepted with a documented tail. If you get
NO-GO: add more data, more epochs, or more zero-PII copy examples, in that
order. See `docs/results.md` for the measured numbers.
