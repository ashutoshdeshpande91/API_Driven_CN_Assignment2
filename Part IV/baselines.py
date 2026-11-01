"""Baseline comparison for Part II: is fine-tuning better than prompting?

Run from the project root (next to app.py):
    python -m src.baselines --test data/test.jsonl --hand hand            # all systems
    python -m src.baselines --systems gemini_fewshot --hand hand          # one system

Systems compared (same scorer, same parser):
    t5_zeroshot   google/flan-t5-base, prefix prompt only
    t5_fewshot    google/flan-t5-base, 2 worked examples in the prompt
    gemini_fewshot hosted LLM via src/llm.py, 6 worked examples (one per event type)
    t5_finetuned  models/ca-extractor

Writes reports/baselines.json.

!! CHECK THE TWO LOADER FUNCTIONS (load_jsonl / load_hand) against your files. !!
"""
import argparse
import json
import os
import time

from src.schema import (EVENT_TYPES, FIELD_ORDER, PREFIX, format_target,
                        normalize_for_scoring, parse_output)

FINETUNED_DIR = "models/ca-extractor"      # override with --finetuned models/oem-extractor
BASE_MODEL = "google/flan-t5-base"

SYSTEM_EXTRACT = (
    "You extract structured details from a notice or circular. Reply with ONE line only, in exactly this "
    "format: key: value ; key: value ; ...  Use these keys in this order: " + ", ".join(FIELD_ORDER) + ". "
    + ("event_type must be one of: " + ", ".join(EVENT_TYPES) + ". " if "event_type" in FIELD_ORDER else "")
    + "Dates are YYYY-MM-DD. Write NA for a field the text does not state. Do not add explanations."
)


# ---------------------------------------------------------------- data loading
def _as_dict(x):
    return parse_output(x) if isinstance(x, str) else dict(x)


def load_jsonl(path):
    """Expect one JSON object per line. Adjust the key names below to your files."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            text = r.get("text") or r.get("notice") or r.get("input")
            gold = r.get("target") or r.get("fields") or r.get("output")
            rows.append((text, _as_dict(gold)))
    return rows


def load_hand():
    """src/handwritten.py: adjust to however your 13 notices are stored."""
    from src import handwritten as h
    items = getattr(h, "NOTICES", None) or getattr(h, "HANDWRITTEN", None)
    rows = []
    for it in items:
        if isinstance(it, dict):
            text = it.get("text") or it.get("notice")
            gold = it.get("gold") or it.get("fields") or it.get("target")
        else:
            text, gold = it
        rows.append((text, _as_dict(gold)))
    return rows


# ---------------------------------------------------------------- scoring
def score_one(pred, gold):
    """Strict and lenient full-record match, judged on the gold fields."""
    strict = all(pred.get(k, "").strip() == str(v).strip() for k, v in gold.items())
    lenient = all(normalize_for_scoring(k, pred.get(k, "")) == normalize_for_scoring(k, str(v))
                  for k, v in gold.items())
    return strict, lenient


def evaluate(predict, rows, label):
    strict = lenient = 0
    times = []
    for i, (text, gold) in enumerate(rows):
        t0 = time.perf_counter()
        try:
            raw = predict(text)
        except Exception as e:            # keep going; count as wrong
            print(f"  [{label}] item {i} failed: {e}")
            raw = ""
        times.append(time.perf_counter() - t0)
        s, l = score_one(parse_output(raw), gold)
        strict += s
        lenient += l
    n = len(rows)
    return {"n": n, "full_exact": round(100 * strict / n, 1), "full_lenient": round(100 * lenient / n, 1),
            "exact_count": strict, "lenient_count": lenient,
            "sec_per_notice": round(sum(times) / n, 3)}


# ---------------------------------------------------------------- predictors
def _shots(train_rows, k_types):
    """One example per requested event type, taken deterministically from the training set."""
    if not any("event_type" in g for _, g in train_rows[:5]):     # schema without event types (scheme data)
        return train_rows[:max(2, len(k_types))]
    seen, shots = set(), []
    for text, gold in train_rows:
        ev = gold.get("event_type")
        if ev in k_types and ev not in seen:
            seen.add(ev)
            shots.append((text, gold))
    return shots


def make_t5(path, shots=()):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    demo = "".join(f"{PREFIX}{t}\n{format_target(g)}\n\n" for t, g in shots)

    def predict(text):
        prompt = demo + PREFIX + text
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=160, num_beams=1)
        return tok.decode(out[0], skip_special_tokens=True)
    return predict


def make_gemini(shots):
    from src.llm import chat
    demo = "".join(f"NOTICE:\n{t}\nANSWER: {format_target(g)}\n\n" for t, g in shots)

    def predict(text):
        time.sleep(0.5)                    # stay under the free-tier rate limit
        line = chat(SYSTEM_EXTRACT, f"{demo}NOTICE:\n{text}\nANSWER:", max_tokens=400, temperature=0.0)
        return line.splitlines()[0] if line else ""   # first line only
    return predict


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--hand", action="store_true", help="also score the hand-written notices in src/handwritten.py")
    ap.add_argument("--hand-file", default=None, help="jsonl of hand-written notices (text, target); overrides --hand")
    ap.add_argument("--finetuned", default=FINETUNED_DIR, help="folder of the fine-tuned model")
    ap.add_argument("--out", default="reports/baselines.json")
    ap.add_argument("--systems", nargs="*",
                    default=["t5_zeroshot", "t5_fewshot", "gemini_fewshot", "t5_finetuned"])
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N test notices")
    a = ap.parse_args()

    train = load_jsonl(a.train)
    test = load_jsonl(a.test)[: a.limit or None]
    sets = {"synthetic_test": test}
    if a.hand_file:
        sets["handwritten"] = load_jsonl(a.hand_file)
    elif a.hand:
        sets["handwritten"] = load_hand()

    builders = {
        "t5_zeroshot": lambda: make_t5(BASE_MODEL),
        "t5_fewshot": lambda: make_t5(BASE_MODEL, _shots(train, {"DIVIDEND", "BONUS_ISSUE"})),
        "gemini_fewshot": lambda: make_gemini(_shots(train, set(EVENT_TYPES))),
        "t5_finetuned": lambda: make_t5(a.finetuned),
    }
    results = {}
    for name in a.systems:
        print(f"== {name}")
        predict = builders[name]()
        results[name] = {s: evaluate(predict, rows, name) for s, rows in sets.items()}
        print(json.dumps(results[name], indent=2))

    os.makedirs("reports", exist_ok=True)
    out = a.out
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update(results)
    json.dump(old, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
