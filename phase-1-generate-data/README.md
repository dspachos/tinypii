# Phase 1 — Generate training data

Produces (raw input, redacted target) pairs for fine-tuning. Two modes:

- `seed` — free, template-based pairs. Enough to validate the training loop.
- `teacher` — pairs from a teacher LLM through an OpenAI-compatible endpoint.
  This is what the published dataset was built from.

Every pair passes a programmatic QA gate before it is written.

## Prerequisites

- `make install` (creates `.venv`)
- For `teacher` mode only: copy `.env.example` to `.env.local` and fill in
  `OPENAI_API_ENDPOINT` and `OPENAI_KEY`. The default teacher model is
  `gemini-2.5-flash`; pass `--model` to use another OpenAI-compatible model.

## Commands

Run from the repo root.

| Command | What it does |
|---|---|
| `make seed` | 200 template pairs → `data/seed.jsonl` |
| `make data` | teacher pairs → `data/synthetic.jsonl` (target 8,647) |
| `.venv/bin/python phase-1-generate-data/generate.py teacher --focus` | regenerate toward known weak spots |
| `.venv/bin/python phase-1-generate-data/generate.py teacher --target N` | custom count |

Both modes are resumable. Rerun the same command and it appends until the
target count is reached.

## Output

`data/seed.jsonl` and `data/synthetic.jsonl`, one JSON record per line:

```json
{"raw_input": "...", "redacted_target": "...", "source": "seed|teacher"}
```

The `redact: ` task prefix is applied in phase 2, not stored here.

## The QA gate

A pair is rejected unless all of these hold:

- The tag-stripped target reconstructs the input at word level, in order.
- The tag inventory matches the PII actually present in the input.
- No email or digit run is left unredacted in the target.
- No name left unredacted after an honorific (`Ms. D'Angelo`).
- The input is not a near-duplicate of one already accepted.

Rejections are counted by reason and printed at the end. Regex is allowed
here on purpose: the no-regex constraint covers only the deployed browser
artifact.

## Filter an existing file

Two cleanup passes run over a finished dataset:

| Command | What it drops |
|---|---|
| `make filter` | Pairs that fail the current QA gate, incl. names after honorifics |
| `make filter-model` | Pairs whose target leaves PII plain that a trained model redacts (leaky golds) |

Leaky golds teach contradictory behavior: the model gets punished for
tagging a name the teacher left unredacted, which drags the copy-fidelity
metric even when the model is right.

## Cost

`teacher` mode spends API tokens. With batches of 8, the published dataset
(10,383 pairs after filtering) took one broad gemini-2.5-flash run, one
broad gemini-3.5-flash-lite run, and one focused weak-spot round. At flash
tier the total API cost stayed under $15.
