#!/usr/bin/env python3
"""Phase 1 — synthetic PII redaction pairs. See PLAN.md §4 Phase 1.

Modes:
    python generate_data.py teacher --target 4000   # LLM pairs -> data/synthetic.jsonl
    python generate_data.py seed                    # template pairs -> data/seed.jsonl

Records: {"raw_input", "redacted_target", "source"}. The "redact: " task
prefix is applied in Phase 2, not stored here.

Credentials come from .env.local (AMAZEEAI_BASE_URL / AMAZEEAI_API_KEY).

The QA gate uses regex on purpose: PLAN.md's no-regex constraint covers the
deployed browser artifact, not training-data validation.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

TAGS = ("NAME", "CREDIT_CARD", "EMAIL", "PHONE", "ADDRESS")
TAG_RE = re.compile(r"\[(NAME|CREDIT_CARD|EMAIL|PHONE|ADDRESS)\]")
ANY_BRACKET_RE = re.compile(r"\[[^\]]*\]")
# 7+ digit-ish run: phones (555-010-9988, 5550199) and cards alike
DIGIT_RUN_RE = re.compile(r"\d[\d .-]{5,}\d")
# 13+ digit-ish run: card numbers only
CARD_RUN_RE = re.compile(r"\d[\d .-]{11,}\d")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

SCENARIOS = [
    "e-commerce order or refund", "customer support chat", "hospital or clinic intake",
    "angry customer email", "travel booking change", "delivery driver note",
    "insurance claim message", "bank support chat", "HR onboarding note",
    "hotel reservation message", "utility company complaint", "school admin note",
    "internal team chat message", "voicemail transcript", "pharmacy pickup note",
]
CARD_FORMATS = [
    "16 digits grouped with dashes (4111-2222-3333-4444)",
    "16 digits grouped with spaces (4111 2222 3333 4444)",
    "16 continuous digits (4111222233334444)",
    "15-digit Amex style with spaces (3782 822463 10005)",
    "15-digit Amex style continuous (378282246310005)",
    "16 digits with dots as separators (4111.2222.3333.4444)",
    "inconsistent mixed separators (4111-2222 3333 4444)",
]
NAME_STYLES = [
    "common Western full names",
    "international and multicultural names",
    "single names only (no surname)",
    "formal titles included (Dr., Prof., Mr., Ms.)",
    "hyphenated surnames (Anne-Marie Sutton)",
    "names with apostrophes (O'Brien, D'Angelo)",
]
NOISE_LEVELS = [
    "clean professional tone",
    "casual slang and abbreviations",
    "typos and misspellings",
    "fragmented shorthand notes",
    "angry or rude customer tone",
    "non-native English phrasing",
]
ENTITY_LOADS = [
    "exactly one PII entity",
    "two different PII entities",
    "name + credit card + email together",
    "multiple entities including phone and address",
    "the same entity value mentioned twice in the message",
    "PII at the very start of the message",
    "PII at the very end of the message",
]


# ---------------------------------------------------------------- QA gate ---

def _alnum_tokens(s: str) -> list[str]:
    """Punctuation-insensitive word tokens: 'contact:mary.j@x.com' -> contact mary j x com."""
    return re.sub(r"[^0-9A-Za-z]+", " ", s).split()


def qa_gate(raw: str, target: str) -> str | None:
    """Return a rejection reason, or None if the pair is acceptable."""
    if not raw.strip() or not target.strip():
        return "empty field"
    if not (20 <= len(raw) <= 600):
        return "input length out of range"
    for tok in ANY_BRACKET_RE.findall(target):
        if tok[1:-1] not in TAGS:
            return f"unknown tag {tok}"
    # Target minus tags must reconstruct the input at the word level:
    # alphanumeric tokens, in order, nothing rewritten. Punctuation is
    # ignored so PII glued to separators ("contact:bob@x.com|4111...") passes.
    # ponytail: whole-word deletions by the teacher are not detected here;
    # Phase 2 recall/copy-fidelity metrics catch systematic drift.
    residue = _alnum_tokens(TAG_RE.sub(" ", target))
    raw_tokens = iter(_alnum_tokens(raw))
    if not all(w in raw_tokens for w in residue):
        return "target drifted from input"
    stripped = TAG_RE.sub(" ", target)
    if EMAIL_RE.search(stripped):
        return "email left unredacted"
    if DIGIT_RUN_RE.search(stripped):
        return "digit run left unredacted"
    if target.count("[CREDIT_CARD]") > len(CARD_RUN_RE.findall(raw)):
        return "card tag without card in input"
    if target.count("[EMAIL]") > len(EMAIL_RE.findall(raw)):
        return "email tag without email in input"
    has_tags = bool(TAG_RE.search(target))
    if not has_tags and (EMAIL_RE.search(raw) or CARD_RUN_RE.search(raw)):
        return "PII in input but no tags in target"
    if raw == target and has_tags:
        return "identical raw/target with tags"
    return None


def norm_key(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip().lower())


def load_seen(path: Path) -> set[str]:
    seen: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                seen.add(norm_key(json.loads(line)["raw_input"]))
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a torn last line from an interrupted run
    return seen


# ------------------------------------------------------------- teacher mode ---

class PIIPair(BaseModel):
    raw_input: str = Field(description="A 1-3 sentence message containing the requested PII")
    redacted_target: str = Field(description="The same message with PII replaced by tags only")


class SyntheticBatch(BaseModel):
    pairs: list[PIIPair]


SYSTEM_PROMPT = """You are a synthetic training-data generator for a tiny privacy model.
You produce pairs: a raw message and its redacted version.

Hard rules:
1. redacted_target must match raw_input word-for-word, except PII spans are
   replaced by exactly one of: [NAME], [CREDIT_CARD], [EMAIL], [PHONE], [ADDRESS].
2. Never paraphrase, never add or drop words outside PII spans.
3. Credit card digits: use exactly the format requested. Phone numbers, emails,
   names and addresses must look realistic and vary between examples.
4. Messages are 1-3 sentences, 20-600 characters.
5. If asked for a zero-PII message, redacted_target equals raw_input exactly.
Respond with JSON."""


def make_client():
    try:
        import instructor
        from openai import OpenAI
    except ImportError:
        sys.exit("Missing deps: pip install -r requirements.txt")
    env: dict[str, str] = {}
    env_path = Path(".env.local")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    base, key = env.get("AMAZEEAI_BASE_URL"), env.get("AMAZEEAI_API_KEY")
    if not base or not key:
        sys.exit(".env.local must define AMAZEEAI_BASE_URL and AMAZEEAI_API_KEY")
    return instructor.from_openai(
        OpenAI(base_url=base, api_key=key), mode=instructor.Mode.JSON
    )


def batch_instruction(rng: random.Random, batch_size: int) -> str:
    if rng.random() < 0.10:
        return (
            f"Generate {batch_size} diverse conversational messages that contain "
            "NO personally identifiable information at all (no names, cards, "
            "emails, phones, addresses). For each, redacted_target equals "
            "raw_input exactly."
        )
    return (
        f"Generate {batch_size} diverse pairs.\n"
        f"- Scenario: {rng.choice(SCENARIOS)}\n"
        f"- If a credit card appears, format: {rng.choice(CARD_FORMATS)}\n"
        f"- Name style: {rng.choice(NAME_STYLES)}\n"
        f"- Tone/noise: {rng.choice(NOISE_LEVELS)}\n"
        f"- Entity load: {rng.choice(ENTITY_LOADS)}\n"
        "- Vary sentence structures across pairs; avoid formulaic "
        "'confirm/repeat the same information' phrasing unless the entity "
        "load explicitly requires repetition."
    )


def run_teacher(args: argparse.Namespace) -> None:
    client = make_client()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(out_path)
    rng = random.Random()
    accepted = len(seen)
    reject_stats: dict[str, int] = {}
    api_failures = 0
    max_batches = (args.target - accepted) * 3 // max(args.batch_size, 1) + 10

    try:
        import tqdm  # noqa: F401 — optional nicety, guarded below
        progress = tqdm.tqdm(total=args.target, initial=accepted, desc="pairs")
    except ImportError:
        class _P:
            def update(self, *_): pass
            def close(self): pass
        progress = _P()

    with out_path.open("a", encoding="utf-8") as out:
        for _ in range(max_batches):
            if accepted >= args.target:
                break
            try:
                batch = client.chat.completions.create(
                    model=args.model,
                    response_model=SyntheticBatch,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": batch_instruction(rng, args.batch_size)},
                    ],
                    temperature=1.0,
                    max_retries=2,
                )
            except Exception as exc:  # noqa: BLE001 — log and move on
                api_failures += 1
                print(f"\nAPI failure ({api_failures}): {exc}", file=sys.stderr)
                time.sleep(2)
                continue
            for pair in batch.pairs:
                reason = qa_gate(pair.raw_input, pair.redacted_target)
                if reason:
                    reject_stats[reason] = reject_stats.get(reason, 0) + 1
                    continue
                key = norm_key(pair.raw_input)
                if key in seen:
                    reject_stats["duplicate"] = reject_stats.get("duplicate", 0) + 1
                    continue
                seen.add(key)
                accepted += 1
                progress.update(1)
                out.write(json.dumps({
                    "raw_input": pair.raw_input,
                    "redacted_target": pair.redacted_target,
                    "source": "teacher",
                }) + "\n")
            out.flush()
            time.sleep(0.3)
    progress.close()

    print(f"\nAccepted: {accepted} -> {out_path}")
    print(f"API failures: {api_failures}")
    for reason, n in sorted(reject_stats.items(), key=lambda kv: -kv[1]):
        print(f"Rejected ({reason}): {n}")


# --------------------------------------------------------------- seed mode ---

SEED_NAMES = [
    "Alice Smith", "Marcus Vance", "Elena Rostova", "Devon Lee", "Hiroshi Tanaka",
    "Fatima Al-Sayed", "Dr. Lena Weber", "Mr. O'Brien", "Anne-Marie Sutton",
    "Prof. Nkechi Obi", "Kwame Mensah", "Sofia Rossi", "Priya Nair", "Ivan Petrov",
    "Chen Wei", "Sunny Rae", "Lars Eriksen", "Amara Diallo", "Tomás Rivera",
    "Ingrid Johansson", "single-name Zephyr", "Ravi Patel",
]
SEED_EMAILS = [
    "user123@gmail.com", "contact@acme.org", "devon.lee@workstation.io",
    "a.kwarteng@clinic.health", "dev+test@mail.co", "mvela@x.org",
    "bob.smith@corp.com", "support@shopnow.net", "x.y@z.co", "r.patel@guest.net",
]
SEED_ADDRESSES = [
    "742 Evergreen Terrace, Springfield", "221B Baker Street, London",
    "12 Rue de Rivoli, Paris", "88 Market St, San Francisco",
    "Kurfuerstendamm 21, Berlin", "3 Via Roma, Milan",
    "1600 Pine Hollow Rd, Austin", "5 Chemin des Fleurs, Geneva",
]
SEED_TEMPLATES = [
    ("Hi, I am {name}. Charge {card} and reach me at {email}.",
     "Hi, I am [NAME]. Charge [CREDIT_CARD] and reach me at [EMAIL]."),
    ("Refund card {card} belonging to {name}.",
     "Refund card [CREDIT_CARD] belonging to [NAME]."),
    ("Email receipt to {email}.", "Email receipt to [EMAIL]."),
    ("My VISA is {card}.", "My VISA is [CREDIT_CARD]."),
    ("Ship to {address} and call {phone} on arrival.",
     "Ship to [ADDRESS] and call [PHONE] on arrival."),
    ("name {name} email {email} card {card} phone {phone}",
     "name [NAME] email [EMAIL] card [CREDIT_CARD] phone [PHONE]"),
    ("guest {name}, phone {phone}, room 12",
     "guest [NAME], phone [PHONE], room 12"),
    ("Reached out to {name} at {phone} about the {card} charge.",
     "Reached out to [NAME] at [PHONE] about the [CREDIT_CARD] charge."),
    ("Meeting moved to 3pm, room 402, bring the slides.",
     "Meeting moved to 3pm, room 402, bring the slides."),
    ("Package left with the front desk, no signature needed.",
     "Package left with the front desk, no signature needed."),
]


def seed_card(rng: random.Random) -> str:
    digits = "".join(str(rng.randint(0, 9)) for _ in range(16))
    style = rng.choice(["dashes", "spaces", "continuous", "dots", "mixed"])
    if style == "dashes":
        return "-".join(digits[i:i + 4] for i in range(0, 16, 4))
    if style == "spaces":
        return " ".join(digits[i:i + 4] for i in range(0, 16, 4))
    if style == "dots":
        return ".".join(digits[i:i + 4] for i in range(0, 16, 4))
    if style == "mixed":
        return f"{digits[:4]}-{digits[4:8]} {digits[8:12]} {digits[12:]}"
    return digits


def seed_phone(rng: random.Random) -> str:
    d = "".join(str(rng.randint(0, 9)) for _ in range(10))
    style = rng.choice(["dashes", "dots", "continuous", "parens", "intl"])
    if style == "dashes":
        return f"{d[:3]}-{d[3:6]}-{d[6:]}"
    if style == "dots":
        return f"{d[:3]}.{d[3:6]}.{d[6:]}"
    if style == "continuous":
        return d
    if style == "parens":
        return f"({d[:3]}) {d[3:6]}-{d[6:]}"
    return f"+44 20 {d[:4]} {d[4:8]}"


def run_seed(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    seen = load_seen(out_path)
    accepted = 0
    rejected = 0
    with out_path.open("a", encoding="utf-8") as out:
        while accepted < args.target:
            tmpl_in, tmpl_out = rng.choice(SEED_TEMPLATES)
            raw = tmpl_in.format(
                name=rng.choice(SEED_NAMES),
                email=rng.choice(SEED_EMAILS),
                card=seed_card(rng),
                phone=seed_phone(rng),
                address=rng.choice(SEED_ADDRESSES),
            )
            target = tmpl_out
            if qa_gate(raw, target) is not None:
                rejected += 1
                continue
            key = norm_key(raw)
            if key in seen:
                continue
            seen.add(key)
            accepted += 1
            out.write(json.dumps({
                "raw_input": raw, "redacted_target": target, "source": "seed",
            }) + "\n")
    print(f"Seed pairs accepted: {accepted} -> {out_path} (rejected {rejected})")


# -------------------------------------------------------------------- main ---

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_teacher = sub.add_parser("teacher", help="Generate pairs with a teacher LLM")
    p_teacher.add_argument("--target", type=int, default=4000)
    p_teacher.add_argument("--batch-size", type=int, default=8)
    p_teacher.add_argument("--model", default="gemini-2.5-flash")
    p_teacher.add_argument("--out", default="data/synthetic.jsonl")

    p_seed = sub.add_parser("seed", help="Generate template-based seed pairs")
    p_seed.add_argument("--target", type=int, default=200)
    p_seed.add_argument("--out", default="data/seed.jsonl")

    args = parser.parse_args()
    if args.mode == "teacher":
        run_teacher(args)
    else:
        run_seed(args)


if __name__ == "__main__":
    main()
