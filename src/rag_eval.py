"""Measure retrieval quality for the rulebook (RAG) tab.

Each question has a phrase that must appear in the right passage and a part of the source file name.
A question counts as a hit when one of the top-k passages comes from the expected document AND
contains the phrase.

    python -m src.rag_eval              # hit-rate only (no language model needed)
    python -m src.rag_eval --with-llm   # also writes the written answers to reports/rag_answers.csv

A miss is not always a retrieval failure: check the printed top passage first, because the expected
phrase could be worded differently in the PDF text.
"""
import argparse
import csv
import json
import re
from pathlib import Path

# source = part of the file name; phrases = any one of them must appear in the passage
QUESTIONS = [
    {"q": "What advance notice is required before the record date of a rights issue?",
     "phrases": ["at least 3 working days"], "source": "Master_Circular"},
    {"q": "Within how many working days must a rights issue be completed after board approval?",
     "phrases": ["23 working days"], "source": "Master_Circular"},
    {"q": "How soon after the record date can bonus shares be traded?",
     "phrases": ["t+2 day"], "source": "Master_Circular"},
    {"q": "What does an issuer proposing a bonus issue have to apply for after the board meeting?",
     "phrases": ["in-principle approval"], "source": "Master_Circular"},
    {"q": "What fine is charged per day for a delay in completing a bonus issue?",
     "phrases": ["20,000 per day"], "source": "Master_Circular"},
    {"q": "How long before a rights issue closes must trading in rights entitlements stop?",
     "phrases": ["three working days prior"], "source": "Master_Circular"},
    {"q": "What is the minimum and maximum period a rights issue stays open for subscription?",
     "phrases": ["minimum period of seven days"], "source": "ICDR"},
    {"q": "How long must a buy-back tender offer remain open?",
     "phrases": ["remain open for a period"], "source": "BuyBack"},
    {"q": "What is the maximum size of a buy-back as a share of paid-up capital and free reserves?",
     "phrases": ["twenty-five per cent"], "source": "BuyBack"},
    {"q": "By when must the public announcement of a buy-back be made after the board resolution?",
     "phrases": ["within two working days"], "source": "BuyBack"},
    {"q": "Can a company buy back its shares through negotiated deals?",
     "phrases": ["negotiated deals"], "source": "BuyBack"},
    {"q": "For which purposes does a listed entity fix a record date?",
     "phrases": ["declaration of dividend"], "source": "LODR"},
    {"q": "How much advance notice did SEBI require before a stock exchange changes the settlement cycle?",
     "phrases": ["advance notice of at least one month"], "source": "T1_Rolling"},
]


def norm(text):
    return re.sub(r"\s+", " ", text.lower())


def phrase_hit(text, phrases):
    t = norm(text)
    return any(norm(p) in t for p in phrases)


def score_question(item, hits):
    """Rank (1-based) of the first correct passage, or None; plus whether the source appeared at all."""
    src = item["source"].lower()
    rank = next((i for i, h in enumerate(hits, 1)
                 if src in h["source"].lower() and phrase_hit(h["text"], item["phrases"])), None)
    return {"rank": rank, "source_in_topk": any(src in h["source"].lower() for h in hits)}


def summarise(results):
    n = len(results)
    ranks = [r["rank"] for r in results]
    return {
        "questions": n,
        "hit_at_k": sum(r is not None for r in ranks) / n,
        "hit_at_1": sum(r == 1 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "source_in_topk": sum(r["source_in_topk"] for r in results) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--index", default="rag_index")
    ap.add_argument("--with-llm", action="store_true")
    args = ap.parse_args()

    from src.rag import Retriever
    retriever = Retriever(args.index)

    results, rows = [], []
    for item in QUESTIONS:
        hits = retriever.search(item["q"], k=args.k)
        res = score_question(item, hits)
        results.append(res)
        top = hits[0]
        rows.append({"question": item["q"], "expected_source": item["source"],
                     "phrase": item["phrases"][0], "rank_of_correct": res["rank"],
                     "source_in_topk": res["source_in_topk"], "top1_source": top["source"],
                     "top1_page": top["page"], "top1_score": round(top["score"], 3)})
        mark = f"rank {res['rank']}" if res["rank"] else "MISS"
        print(f"[{mark:7}] {item['q']}")
        if not res["rank"]:
            print(f"          top passage ({top['source']} p.{top['page']}): {top['text'][:160]!r}")

    summary = summarise(results)
    Path("reports").mkdir(exist_ok=True)
    with open("reports/rag_eval.json", "w", encoding="utf-8") as f:
        json.dump({"k": args.k, **summary, "rows": rows}, f, indent=2)
    with open("reports/rag_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nRight passage in top {args.k}: {summary['hit_at_k']:.0%}   ranked first: "
          f"{summary['hit_at_1']:.0%}   MRR: {summary['mrr']:.2f}   -> reports/rag_eval.json")

    if args.with_llm:
        from src import llm
        from src.ui_helpers import friendly_llm_error
        if not llm.llm_available():
            print("No language-model key set; skipping written answers.")
            return
        with open("reports/rag_answers.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["question", "answer", "sources"])
            for item in QUESTIONS:
                hits = retriever.search(item["q"], k=args.k)
                try:
                    ans = llm.answer_with_context(item["q"], hits)
                except Exception as e:
                    ans = f"(failed: {friendly_llm_error(e)})"
                w.writerow([item["q"], ans, "; ".join(f"{h['source']} p.{h['page']}" for h in hits)])
        print("Wrote reports/rag_answers.csv - read the answers and grade them yourself.")


if __name__ == "__main__":
    main()
