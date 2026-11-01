"""Synthetic, anonymised two-wheeler finance scheme circulars (fictional financiers and models).
    python -m src.synth_schemes            -> data_oem/train.jsonl, val.jsonl, test.jsonl, hindi_probe.txt
Each line: {"text": <circular>, "target": "<key: value ; ...>"}  (rename keys if your train.py differs)
"""
import json
import os
import random
from datetime import date, timedelta

from src.schema_oem import format_target

FINANCIERS = ["Sunrise Finserv", "Pragati Capital", "Lakeshore Bank", "Orion Credit", "Meridian Finance",
              "Kaveri Housing & Auto Finance", "Bluepeak NBFC"]
SCHEMES = ["Festive Dhamaka", "Utsav Offer", "Diwali Delight", "Navratri Special", "Happy Ride", "Shubh Drive"]
MODELS = ["Zephyr 110", "Raptor 150", "Vega 125", "Urban E1", "Stallion 160", "Breeze 100"]
ZONES = ["North", "South", "East", "West", "Central"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def _d(d, style):
    if style == 0:
        return d.isoformat()
    if style == 1:
        return d.strftime("%d/%m/%Y")
    if style == 2:
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    return d.strftime("%d-%b-%Y")


def make(rng):
    f = dict(financier=rng.choice(FINANCIERS), scheme=rng.choice(SCHEMES), model=rng.choice(MODELS),
             zone=rng.choice(ZONES))
    f["rate"] = f"{rng.choice([6.99, 7.5, 8.25, 8.99, 9.5, 10.75, 11.99, 12.5]):g}"
    f["proc_fee"] = rng.choice(["0", "499", "999", "1500", "1999", "2500"])
    f["down_payment"] = rng.choice(["5", "10", "15", "20", "25"])
    months = rng.choice([12, 18, 24, 30, 36, 48])
    f["tenure"] = str(months)
    f["cashback"] = rng.choice(["NA", "NA", "500", "1000", "1500", "2000"])
    start = date(2026, 9, 1) + timedelta(days=rng.randint(0, 40))
    end = start + timedelta(days=rng.randint(15, 60))
    f["valid_from"], f["valid_to"] = start.isoformat(), end.isoformat()
    st = rng.randint(0, 3)
    pf = "nil" if f["proc_fee"] == "0" else f"Rs. {int(f['proc_fee']):,}"
    ten = (f"{months} months" if (months % 12 or rng.random() < .5)
           else (f"{months // 12} year" if months == 12 else f"{months // 12} years"))
    cb = "" if f["cashback"] == "NA" else rng.choice([f" Cashback of Rs. {int(f['cashback']):,} on loan disbursal.",
                                                      f" Customers also get a cashback of Rs. {f['cashback']}."])
    t = rng.randint(0, 2)
    if t == 0:
        text = (f"CIRCULAR - {f['financier']}\nSub: {f['scheme']} scheme for {f['model']} ({f['zone']} zone)\n"
                f"Valid {_d(start, st)} to {_d(end, st)}.\nRate of interest {f['rate']}% p.a.; processing fee {pf}; "
                f"minimum down payment {f['down_payment']}% of on-road price; tenure up to {ten}.{cb}")
    elif t == 1:
        text = (f"{f['financier']} is pleased to announce the {f['scheme']} for {f['zone']} region dealers. "
                f"On the {f['model']}, ROI starts at {f['rate']} per cent per annum with a processing charge of {pf}. "
                f"Loans are available for a maximum of {ten} with {f['down_payment']}% down payment."
                f"{cb} Offer period: {_d(start, st)} - {_d(end, st)}.")
    else:
        text = (f"Scheme: {f['scheme']} | Financier: {f['financier']} | Model: {f['model']} | Zone: {f['zone']}\n"
                f"ROI: {f['rate']}% | PF: {pf} | DP: {f['down_payment']}% | Tenure: {ten}\n"
                f"Validity: from {_d(start, st)} till {_d(end, st)}.{cb}")
    return text, format_target(f)


HINDI_PROBE = ("परिपत्र - सनराइज़ फिनसर्व\nविषय: फेस्टिव धमाका योजना, ज़ेफ़िर 110 (उत्तर क्षेत्र)\n"
               "वैधता 2026-10-20 से 2026-11-30 तक। ब्याज दर 8.99% प्रति वर्ष; प्रोसेसिंग शुल्क रु. 999; "
               "न्यूनतम डाउन पेमेंट 10%; अवधि 36 माह तक।")


def main(n=900, seed=7, out="data_oem"):
    rng = random.Random(seed)
    rows = [make(rng) for _ in range(n)]
    os.makedirs(out, exist_ok=True)
    cuts = {"train": rows[: int(.8 * n)], "val": rows[int(.8 * n): int(.9 * n)], "test": rows[int(.9 * n):]}
    for name, rs in cuts.items():
        with open(f"{out}/{name}.jsonl", "w", encoding="utf-8") as fh:
            for text, tgt in rs:
                fh.write(json.dumps({"text": text, "target": tgt}, ensure_ascii=False) + "\n")
    open(f"{out}/hindi_probe.txt", "w", encoding="utf-8").write(HINDI_PROBE)
    print({k: len(v) for k, v in cuts.items()}, "->", out)


if __name__ == "__main__":
    main()
