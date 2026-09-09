#!/usr/bin/env python3
"""Phase 2 — fine-tune flan-t5-small for PII redaction. See PLAN.md §4 Phase 2.

Usage:
    python train_seq2seq.py             # full run -> saved_generative_pii_model/
    python train_seq2seq.py --smoke     # tiny run on seed data, validates the loop

The acceptance gate (tag recall >= 0.98, copy fidelity >= 0.99) is evaluated
afterwards by evaluate.py on the held-out split and the adversarial set.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

MODEL_NAME = "google/flan-t5-small"
TAGS = ("NAME", "CREDIT_CARD", "EMAIL", "PHONE", "ADDRESS")
TAG_RE = re.compile(r"\[(NAME|CREDIT_CARD|EMAIL|PHONE|ADDRESS)\]")


def load_pairs(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    obj = json.loads(line)
                    rows.append({
                        "input_text": "redact: " + obj["raw_input"],
                        "target_text": obj["redacted_target"],
                    })
    return rows


# ------------------------------------------------------------------ metrics ---

def tag_counts(text: str) -> Counter:
    return Counter(TAG_RE.findall(text))


def residue_tokens(text: str) -> list[str]:
    """Non-PII word content: tags stripped, punctuation-insensitive."""
    return re.sub(r"[^0-9A-Za-z]+", " ", TAG_RE.sub(" ", text)).split()


def redaction_metrics(preds: list[str], golds: list[str]) -> dict:
    """Micro-averaged over the dataset.

    - tag_recall:    gold tags the prediction also produced (leaks when < 1)
    - tag_precision: predicted tags that are backed by gold (over-redaction when < 1)
    - copy_fidelity: fraction of gold non-PII tokens preserved, in order
    - hallucination: pred tokens with no gold counterpart / pred tokens
    """
    matched_tags = gold_tags = pred_tags = 0
    matched_tok = gold_tok = extra_tok = 0
    exact = 0
    for pred, gold in zip(preds, golds):
        pc, gc = tag_counts(pred), tag_counts(gold)
        matched_tags += sum(min(pc[t], gc[t]) for t in TAGS)
        gold_tags += sum(gc.values())
        pred_tags += sum(pc.values())
        ptoks, gtoks = residue_tokens(pred), residue_tokens(gold)
        it = iter(ptoks)
        m = sum(1 for w in gtoks if w in it)
        matched_tok += m
        gold_tok += len(gtoks)
        extra_tok += len(ptoks) - m
        exact += int(pred.strip() == gold.strip())
    n = max(len(golds), 1)
    return {
        "tag_recall": matched_tags / max(gold_tags, 1),
        "tag_precision": matched_tags / max(pred_tags, 1),
        "copy_fidelity": matched_tok / max(gold_tok, 1),
        "hallucination": extra_tok / max(matched_tok + extra_tok, 1),
        "exact_match": exact / n,
    }


# --------------------------------------------------------------- training ----

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run on seed data")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--output-dir", default="saved_generative_pii_model")
    args = ap.parse_args()

    set_seed(42)
    if args.smoke:
        rows = load_pairs(["data/seed.jsonl"])[:64]
        epochs, steps, log = 1, 8, 4
        out_dir = "saved_smoke_model"
    else:
        rows = load_pairs(["data/synthetic.jsonl", "data/seed.jsonl"])
        epochs, steps, log = args.epochs, -1, 25
        out_dir = args.output_dir

    dataset = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        model_inputs = tokenizer(examples["input_text"], max_length=256, truncation=True)
        labels = tokenizer(text_target=examples["target_text"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.asarray(preds[0] if isinstance(preds, tuple) else preds)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(np.asarray(labels) != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_golds = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return redaction_metrics(decoded_preds, decoded_golds)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./trainer_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        num_train_epochs=epochs,
        max_steps=steps,
        logging_steps=log,
        save_total_limit=1,
        predict_with_generate=True,
        generation_max_length=128,
        fp16=torch.cuda.is_available(),
        report_to="none",
        use_cpu=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print({k: round(v, 4) for k, v in metrics.items()})

    if not args.smoke:
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
