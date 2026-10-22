# Dataset

QA-gated training pairs for the PII redaction model. Phase 2 consumes these
files as-is, so retraining needs no API access.

| File | Pairs | Source |
|---|---|---|
| `synthetic.jsonl` | 10,383 | teacher LLMs via an OpenAI-compatible endpoint: gemini-2.5-flash (first 1,856 pairs) and gemini-3.5-flash-lite (rest, incl. one focused weak-spot round) |
| `seed.jsonl` | 200 | templates, no API cost |

Record format, one JSON object per line:

```json
{"raw_input": "Hi, I am Marcus Vance. Charge 4111 2222 3333 4444.",
 "redacted_target": "Hi, I am [NAME]. Charge [CREDIT_CARD].",
 "source": "teacher|seed"}
```

All PII in these pairs is synthetic. The teacher prompts instructed fake
names, cards, emails, phones, and addresses; the seed templates generate
random values.

## QA gate

Every pair passed the checks in `phase-1-generate-data/generate.py` before
it was accepted:

- tag-stripped target reconstructs the input at word level, in order
- tag inventory matches the PII actually present in the input
- no unredacted email or digit run in the target
- no near-duplicate input

In the final generation round the gate rejected about 2% of candidates
(mostly duplicates), with zero API failures.

After generation, two filter passes cleaned the file
(`generate.py filter`): a stricter regex gate that drops names left
unredacted after honorifics (186 pairs), and a model-assisted pass
(`generate.py filter --model models/flan-t5-small-pii`) that drops pairs
whose target leaves PII plain while a trained model redacts it (86 pairs).

## Regenerate

`make seed` and `make data` (phase 1). `make data` needs `.env.local` with
teacher credentials and spends API tokens; see
[phase-1-generate-data/README.md](../phase-1-generate-data/README.md).
