# Phase 2 — Fine-tune

Fine-tunes `google/flan-t5-small` on the phase 1 data. Every example uses
the `redact: ` task prefix. Input is truncated at 256 tokens, targets at 128.

## Prerequisites

- `make install`
- `data/seed.jsonl` (`make seed`) for the smoke run
- `data/synthetic.jsonl` + `data/seed.jsonl` (`make data`, `make seed`) for
  the full run

## Commands

Run from the repo root.

| Command | What it does |
|---|---|
| `make smoke` | tiny run on 64 seed pairs, minutes; validates the loop |
| `make train` | full run, 5 epochs, effective batch 16 → `models/flan-t5-small-pii` |

Direct flags on `phase-2-train/finetune.py`: `--epochs`, `--batch`,
`--grad-accum`, `--output-dir`.

## Metrics during training

`compute_metrics` runs on generated text after each epoch:

| Metric | Meaning |
|---|---|
| tag_recall | gold tags the model also produced (a miss is a leak) |
| tag_precision | predicted tags backed by gold (under 1 is over-redaction) |
| copy_fidelity | share of non-PII tokens preserved verbatim |
| exact_match | whole output identical to the target |
| hallucination | generated tokens with no gold counterpart |

The formal acceptance gate runs in phase 3.

## Output

`models/flan-t5-small-pii` — saved with `tie_word_embeddings: false`. The
fine-tune leaves `shared` and `lm_head` diverged while the config still says
tied. Saving untied keeps the ONNX export in phase 4 correct. Do not
re-tie these weights.
