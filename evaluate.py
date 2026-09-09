#!/usr/bin/env python3
"""Phase 2 acceptance gate. See PLAN.md §4 Phase 2 / §7.

Evaluates the saved model on the held-out test split (rebuilt with the same
seed as training) and the frozen adversarial set, then prints the go/no-go
verdict: tag recall >= 0.98 AND copy fidelity >= 0.99 on both.

Usage: python evaluate.py [--model saved_generative_pii_model]
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from train_seq2seq import load_pairs, redaction_metrics

# Outside the training distribution on purpose; excluded from the gate math,
# reported separately. See PLAN.md Phase 1 notes.
KNOWN_HARD = ("UPS tracking", "[urgent]")


@torch.no_grad()
def generate_batch(model, tokenizer, texts: list[str], device: str) -> list[str]:
    enc = tokenizer(
        ["redact: " + t for t in texts],
        return_tensors="pt", padding=True, truncation=True, max_length=256,
    ).to(device)
    out = model.generate(**enc, max_new_tokens=128)
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def evaluate_set(model, tokenizer, rows: list[dict], device: str, label: str) -> tuple[dict, dict | None]:
    preds = generate_batch(model, tokenizer, [r["raw_input"] for r in rows], device)
    golds = [r["redacted_target"] for r in rows]
    metrics = redaction_metrics(preds, golds)
    print(f"\n== {label} ({len(rows)} examples) ==")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    hard_idx = [i for i, r in enumerate(rows)
                if any(h in r["raw_input"] for h in KNOWN_HARD)]
    metrics_easy = None
    if hard_idx:
        easy = [i for i in range(len(rows)) if i not in hard_idx]
        metrics_easy = redaction_metrics([preds[i] for i in easy], [golds[i] for i in easy])
        print(f"  (excluding {len(hard_idx)} known-hard: "
              f"recall {metrics_easy['tag_recall']:.4f}, "
              f"fidelity {metrics_easy['copy_fidelity']:.4f})")

    shown = 0
    for pred, gold, row in zip(preds, golds, rows):
        if shown >= 8 or pred.strip() == gold.strip():
            continue
        shown += 1
        print(f"  MISMATCH: {row['raw_input'][:70]!r}\n    pred: {pred[:90]!r}\n    gold: {gold[:90]!r}")
    return metrics, metrics_easy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saved_generative_pii_model")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()

    test_rows = held_out_rows()
    adv_rows = json.load(open("eval/adversarial.json", encoding="utf-8"))

    m_test, _ = evaluate_set(model, tokenizer, test_rows, device, "held-out test split")
    m_adv, m_adv_easy = evaluate_set(model, tokenizer, adv_rows, device, "adversarial set")

    def gate(m):
        return m["tag_recall"] >= 0.98 and m["copy_fidelity"] >= 0.99

    print("\n== acceptance gate ==")
    print(f"  held-out:           {'PASS' if gate(m_test) else 'FAIL'}")
    print(f"  adversarial (easy): {'PASS' if m_adv_easy and gate(m_adv_easy) else 'FAIL'}")
    ok = gate(m_test) and bool(m_adv_easy) and gate(m_adv_easy)
    print(f"\n  VERDICT: {'GO' if ok else 'NO-GO — iterate: more data / epochs / zero-PII examples'}")


def held_out_rows() -> list[dict]:
    """Rebuild the same 90/10 split as training (seed 42); return test rows."""
    from datasets import Dataset
    rows = load_pairs(["data/synthetic.jsonl", "data/seed.jsonl"])
    ds = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=42)
    return [{
        "raw_input": ex["input_text"][len("redact: "):],
        "redacted_target": ex["target_text"],
    } for ex in ds["test"]]


if __name__ == "__main__":
    main()
