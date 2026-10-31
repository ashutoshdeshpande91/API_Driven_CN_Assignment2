"""Synthetic corporate-action notice generator.

All companies, ISINs and amounts are fictional. The notices imitate listed-company
disclosures so the project needs no confidential data.

Usage (from the project root):
    python -m src.synth --n 3000 --seed 42
"""
import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from src.schema import EVENT_TYPES, format_target

PREFIXES = ["Aravali", "Sahyadri", "Konark", "Nilgiri", "Vindhya", "Kaveri", "Himalaya", "Deccan",
            "Malabar", "Godavari", "Satpura", "Narmada", "Shivalik", "Tapti", "Zanskar", "Kosi",
            "Bhima", "Chambal", "Sutlej", "Ghatprabha"]
SECTORS = ["Textiles", "Pharma", "Steels", "Auto Components", "Foods", "Chemicals", "Infratech",
           "Power", "Logistics", "Cements", "Software", "Finserv", "Paints", "Packaging"]
SUFFIXES = ["Limited", "Ltd."]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
CURRENCY = ["Rs.", "Rs", "INR"]


def _isin_check_digit(body: str) -> int:
    digits = "".join(str(int(c, 36)) for c in body)
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def make_isin(rng):
    body = "INE" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(6)) + "01"
    return body + str(_isin_check_digit(body))


def make_company(rng):
    return f"{rng.choice(PREFIXES)} {rng.choice(SECTORS)} {rng.choice(SUFFIXES)}"


def ordinal(n):
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def fmt_date(d, rng):
    style = rng.choice(["long", "dash", "us", "ordinal", "slash"])
    m = MONTHS[d.month - 1]
    if style == "long":
        return f"{d.day} {m} {d.year}"
    if style == "dash":
        return f"{d.day:02d}-{m[:3]}-{d.year}"
    if style == "us":
        return f"{m} {d.day}, {d.year}"
    if style == "ordinal":
        return f"{ordinal(d.day)} {m} {d.year}"
    return f"{d.day:02d}/{d.month:02d}/{d.year}"  # dd/mm/yyyy (Indian convention)


def random_weekday(rng, base=date(2026, 1, 5), span=540):
    d = base + timedelta(days=rng.randint(0, span))
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def cur(rng):
    return rng.choice(CURRENCY)


def num(x):
    return f"{x:g}"


def gen_dividend(rng, company, rec):
    amt = rng.choice([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 5.5, 6, 7.5, 8, 10, 12, 15, 20, 25])
    typ = rng.choice(["interim", "final", "special"])
    art = "an" if typ == "interim" else "a"
    fv = rng.choice([1, 2, 5, 10])
    fy = rng.choice([2026, 2027])
    c = cur(rng)
    core = rng.choice([
        f"The Board has declared {art} {typ} dividend of {c} {num(amt)} per equity share of face value {c} {fv}/- each for the financial year ended 31st March {fy}.",
        f"The Directors recommended {art} {typ} dividend at {c} {num(amt)} per share (face value {c} {fv} each) on the paid-up equity share capital.",
        f"Declaration of {typ} dividend: {c} {num(amt)} per equity share (on shares of {c} {fv}/- each).",
    ])
    pay = random_weekday(rng, base=rec + timedelta(days=10), span=15)
    side = f"The dividend will be paid on or before {fmt_date(pay, rng)} to eligible shareholders."
    return core, {"amount": num(amt)}, side


def gen_split(rng, company, rec):
    old = rng.choice([2, 5, 10, 10, 10, 100])
    new = rng.choice([d for d in (1, 2, 5) if d < old and old % d == 0])
    mult = old // new
    c = cur(rng)
    core = rng.choice([
        f"The Board approved the sub-division (split) of 1 equity share of face value {c} {old}/- each into {mult} equity shares of face value {c} {new}/- each.",
        f"Sub-division of the existing equity shares of the Company from face value of {c} {old}/- per share to {c} {new}/- per share.",
        f"The Company proposes a stock split whereby each equity share of nominal value {c} {old} will be split into {mult} shares of {c} {new} each.",
    ])
    side = "The Board Meeting commenced at 11:30 AM and concluded at 02:45 PM."
    return core, {"old_fv": str(old), "new_fv": str(new)}, side


def gen_bonus(rng, company, rec):
    a = rng.choice([1, 1, 2, 3])
    b = rng.choice([1, 2, 3, 4, 5, 10])
    core = rng.choice([
        f"The Board has approved the issue of bonus equity shares in the ratio of {a}:{b}, i.e. {a} new fully paid-up equity share(s) for every {b} equity share(s) held.",
        f"Issue of bonus shares in the proportion of {a} equity share(s) for every {b} equity share(s) held by the shareholders, by capitalisation of free reserves.",
        f"Recommendation of bonus issue: ratio {a}:{b} (fully paid-up bonus shares of the Company).",
    ])
    side = "The bonus shares will rank pari passu with the existing equity shares."
    return core, {"ratio": f"{a}:{b}"}, side


def gen_rights(rng, company, rec):
    a = rng.choice([1, 2, 3])
    b = rng.choice([2, 3, 4, 5, 7, 10, 15])
    price = rng.choice([25, 40, 60, 75, 100, 125, 150, 250, 400, 650, 1000, 1500, 2500])
    n = rng.choice([500000, 1200000, 2500000, 4000000, 7500000, 12000000])
    c = cur(rng)
    core = rng.choice([
        f"The Board approved a rights issue of up to {n:,} equity shares at a price of {c} {price}/- per share, in the ratio of {a}:{b}, i.e. {a} rights equity share(s) for every {b} equity share(s) held.",
        f"Rights Issue: {a} equity share(s) for every {b} held, at an issue price of {c} {price} per rights equity share, aggregating up to {n:,} shares.",
        f"Offer of equity shares on rights basis in the entitlement ratio of {a}:{b} at {c} {price} each.",
    ])
    side = "The Rights Issue Committee will decide the issue opening and closing dates separately."
    return core, {"ratio": f"{a}:{b}", "price": str(price)}, side


def gen_buyback(rng, company, rec):
    price = rng.choice([120, 180, 250, 340, 500, 750, 1100, 1800, 3200, 4800])
    size = rng.choice([10, 25, 50, 100, 250, 500, 1000, 2000])
    fv = rng.choice([1, 2, 5, 10])
    c = cur(rng)
    core = rng.choice([
        f"The Board approved the buy-back of fully paid-up equity shares of face value {c} {fv}/- each at a price of {c} {price}/- per equity share, payable in cash, for an aggregate amount not exceeding {c} {size} crore, through the tender offer route.",
        f"Buyback of equity shares through tender offer at {c} {price} per share; maximum buyback size {c} {size} crore (excluding transaction costs).",
        f"Approval of buy-back at a price of {c} {price} per equity share (face value {c} {fv}) aggregating to {c} {size} crore.",
    ])
    side = "The buy-back will be undertaken in accordance with the SEBI (Buy-Back of Securities) Regulations, 2018."
    return core, {"price": str(price), "size_cr": str(size)}, side


def gen_merger(rng, company, rec):
    other = make_company(rng)
    a = rng.choice([1, 2, 3, 5, 7, 9, 11])
    b = rng.choice([2, 3, 4, 5, 10, 13, 20])
    core = rng.choice([
        f"Scheme of Amalgamation of {other} with {company}: shareholders of {other} will receive {a} fully paid-up equity share(s) of {company} for every {b} fully paid-up equity share(s) held in {other}.",
        f"The Board approved the Scheme of Arrangement between {other} and {company}. Share exchange ratio: {a} equity share(s) of {company} for every {b} equity share(s) of {other}.",
        f"Merger of {other} into {company} pursuant to a Scheme of Amalgamation; swap ratio {a}:{b} ({a} new share(s) of {company} against {b} share(s) of {other}).",
    ])
    side = "The Scheme is subject to approval of the stock exchanges, SEBI, shareholders, creditors and the National Company Law Tribunal."
    return core, {"ratio": f"{a}:{b}"}, side


BUILDERS = {
    "DIVIDEND": gen_dividend,
    "STOCK_SPLIT": gen_split,
    "BONUS_ISSUE": gen_bonus,
    "RIGHTS_ISSUE": gen_rights,
    "BUYBACK": gen_buyback,
    "MERGER": gen_merger,
}

INTROS = [
    "We wish to inform you that the Board of Directors of {company} (the Company), at its meeting held on {bm}, has approved the following:",
    "This is to inform you that, pursuant to the approval of the Board of Directors dated {bm}, the Company has announced the following corporate action:",
    "In continuation of our earlier intimation, the Board of Directors of the Company on {bm} considered and approved the following:",
    "Please note the following corporate action announced by {company}:",
]
REC_SENTS = [
    "The record date for determining the eligible shareholders has been fixed as {d}.",
    "The Company has fixed {d} as the Record Date for the said purpose.",
    "Record Date: {d}.",
    "Shareholders whose names appear in the Register of Members / list of Beneficial Owners as on {d} (Record Date) shall be eligible.",
]
EX_SENTS = [
    "The ex-date will be {d}.",
    "Ex-Date: {d}.",
    "The shares will trade ex-entitlement from {d}.",
]
CLOSERS = [
    "Kindly take the above information on record.",
    "This is for your information and records.",
    "Thanking you. For {company}, Company Secretary & Compliance Officer.",
    "",
]


def header(rng, company, isin):
    return rng.choice([
        f"Subject: Intimation under Regulation 30 - {company} (ISIN: {isin})",
        f"Scrip: {company} | ISIN: {isin} | Subject: Outcome of Board Meeting",
        f"To, The Listing Department, National Stock Exchange of India Ltd. / BSE Ltd. Sub: {company}, ISIN {isin}",
        f"Corporate Announcement - {company} - ISIN: {isin}",
    ])


def make_notice(rng, event):
    company = make_company(rng)
    isin = make_isin(rng)
    rec = random_weekday(rng)
    ex_stated = rng.random() < 0.6   # many announcements state only the record date
    core, extra, side = BUILDERS[event](rng, company, rec)
    bm = fmt_date(rec - timedelta(days=rng.randint(10, 45)), rng)

    parts = [
        header(rng, company, isin),
        rng.choice(INTROS).format(company=company, bm=bm),
        core,
        rng.choice(REC_SENTS).format(d=fmt_date(rec, rng)),
    ]
    if ex_stated:
        parts.append(rng.choice(EX_SENTS).format(d=fmt_date(rec, rng)))  # T+1: ex-date = record date (see README)
    if side:
        parts.append(side)
    closer = rng.choice(CLOSERS).format(company=company)
    if closer:
        parts.append(closer)

    fields = {
        "event_type": event,
        "company": company,
        "isin": isin,
        "record_date": rec.isoformat(),
        "ex_date": rec.isoformat() if ex_stated else "NA",
    }
    fields.update(extra)
    return " ".join(parts), fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        event = EVENT_TYPES[i % len(EVENT_TYPES)]  # balanced classes
        text, fields = make_notice(rng, event)
        rows.append({"id": f"N{i:05d}", "event_type": event, "text": text,
                     "fields": fields, "target": format_target(fields)})
    rng.shuffle(rows)

    n_train, n_val = int(0.8 * len(rows)), int(0.1 * len(rows))
    splits = {"train": rows[:n_train], "val": rows[n_train:n_train + n_val], "test": rows[n_train + n_val:]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(items)} rows")

    with open(out / "samples.txt", "w", encoding="utf-8") as f:
        for r in splits["train"][:6]:
            f.write(f"[{r['event_type']}]\n{r['text']}\n=> {r['target']}\n\n")
    print(f"Wrote {out}/train.jsonl, val.jsonl, test.jsonl, samples.txt")


if __name__ == "__main__":
    main()
