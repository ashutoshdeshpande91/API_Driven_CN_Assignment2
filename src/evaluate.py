"""Evaluate a model on the held-out synthetic test set.

    python -m src.evaluate                                        # fine-tuned model
    python -m src.evaluate --model-dir google/flan-t5-base --tag base   # BEFORE fine-tuning (for the report)
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.extractor import Extractor
from src.schema import normalize_for_scoring, parse_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/ca-extractor")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--tag", default="finetuned")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[:args.limit]

    ex = Extractor(args.model_dir)
    raw = ex.extract_raw([r["text"] for r in rows])
    preds = [parse_output(o) for o in raw]

    field_ok, field_tot, lenient_ok, lenient_full = Counter(), Counter(), Counter(), 0
    per_event = defaultdict(lambda: [0, 0])
    full_ok, errors = 0, []
    for r, raw_out, p in zip(rows, raw, preds):
        gold = r["fields"]
        ok_all, ok_len = True, True
        for k, v in gold.items():
            field_tot[k] += 1
            if p.get(k) == v:
                field_ok[k] += 1
            else:
                ok_all = False
            if normalize_for_scoring(k, p.get(k)) == normalize_for_scoring(k, v):
                lenient_ok[k] += 1
            else:
                ok_len = False
        lenient_full += ok_len
        ev = gold["event_type"]
        per_event[ev][1] += 1
        per_event[ev][0] += int(p.get("event_type") == ev)
        if ok_all:
            full_ok += 1
        else:
            errors.append({"id": r["id"], "gold": r["target"], "pred": raw_out})

    n = len(rows)
    metrics = {
        "tag": args.tag,
        "model": args.model_dir,
        "n_test": n,
        "event_type_accuracy": field_ok["event_type"] / n,
        "full_record_exact_match": full_ok / n,
        "field_exact_match": {k: field_ok[k] / field_tot[k] for k in field_tot},
        "full_record_lenient": lenient_full / n,
        "field_lenient_match": {k: lenient_ok[k] / field_tot[k] for k in field_tot},
        "per_event_accuracy": {k: a / b for k, (a, b) in per_event.items()},
    }
    Path("reports").mkdir(exist_ok=True)
    with open(f"reports/metrics_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(f"reports/errors_{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "gold", "pred"])
        w.writeheader()
        w.writerows(errors)

    print(json.dumps(metrics, indent=2))
    print(f"{len(errors)} records with at least one wrong field -> reports/errors_{args.tag}.csv")


if __name__ == "__main__":
    main()
