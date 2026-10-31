"""Hand-written test notices in wording the synthetic generator never produces.

All companies and ISINs are fictional. Each notice is a deliberate stress test: dates written as
03.11.2026, ratios in words, amounts with commas, abbreviations, one-line alerts, and ratios that
could be confused with other numbers.  Score the model on them with:

    python -m src.handwritten
    python -m src.evaluate --data data/handwritten.jsonl --tag handwritten
"""
import json
from pathlib import Path

from src.schema import format_target

NOTICES = [
    ("H01", "DIVIDEND", "Nilgiri Cements Limited", "INE7KJ2L5019", "2026-08-14", "NA", {"amount": "12"},
     "Dear Sir/Madam, Kindly note that Nilgiri Cements Limited (ISIN: INE7KJ2L5019) has declared a final "
     "dividend of Rs. 12/- (Rupees Twelve only) per equity share for FY 2025-26. The Company has fixed "
     "Friday, 14 August 2026 as the record date for payment of the dividend."),
    ("H02", "DIVIDEND", "Kosi Textiles Ltd.", "INE5MN8Q1010", "2026-11-20", "2026-11-20", {"amount": "2.5"},
     "Intimation of Record Date - Kosi Textiles Ltd. | ISIN - INE5MN8Q1010. Interim dividend @ \u20b9 2.50 "
     "per share (face value \u20b9 5) declared by the Board on 03.11.2026. Record date: 20.11.2026. "
     "Ex-date: 20.11.2026."),
    ("H03", "STOCK_SPLIT", "Zanskar Foods Limited", "INE2PQ4R7017", "2026-12-04", "NA",
     {"old_fv": "10", "new_fv": "5"},
     "The Board of Directors of Zanskar Foods Limited (ISIN INE2PQ4R7017) has approved the sub-division of "
     "each equity share of nominal value Rs.10 into two equity shares of nominal value Rs.5 each. The record "
     "date for the sub-division has been fixed as Friday, 4 December 2026."),
    ("H04", "STOCK_SPLIT", "Satpura Packaging Ltd", "INE9TV3W2014", "2026-11-27", "NA",
     {"old_fv": "2", "new_fv": "1"},
     "Satpura Packaging Ltd, ISIN INE9TV3W2014, announces a stock split in the ratio of 1:2, i.e. every equity "
     "share of face value Rs. 2/- will be split into two shares of Rs. 1/- each. Record date: November 27, 2026."),
    ("H05", "BONUS_ISSUE", "Konark Auto Components Limited", "INE4RS6T3011", "2026-12-11", "NA",
     {"ratio": "1:2"},
     "Konark Auto Components Limited (ISIN: INE4RS6T3011) announces the issue of bonus equity shares in the "
     "ratio of one (1) equity share for every two (2) equity shares held. The record date for the bonus issue "
     "is 11 December 2026."),
    ("H06", "BONUS_ISSUE", "Malabar Pharma Ltd", "INE8UW9X4018", "2026-12-18", "2026-12-18",
     {"ratio": "3:5"},
     "Subject: Bonus Issue - Malabar Pharma Ltd (INE8UW9X4018). The Board has recommended bonus shares in the "
     "proportion 3:5 (three bonus shares for every five held), subject to shareholder approval. Record date to "
     "be December 18, 2026. Ex-date December 18, 2026."),
    ("H07", "RIGHTS_ISSUE", "Brahma Logistics Limited", "INE6VX2Y8015", "2027-01-08", "NA",
     {"ratio": "1:4", "price": "175"},
     "Brahma Logistics Limited (ISIN INE6VX2Y8015) rights issue of equity shares at an issue price of "
     "Rs. 175/- per equity share in the ratio of 1 equity share for every 4 equity shares held. "
     "Record date: 8th January 2027."),
    ("H08", "RIGHTS_ISSUE", "Chambal Infratech Ltd.", "INE3WY7Z6012", "2027-01-15", "NA",
     {"ratio": "7:20", "price": "90"},
     "NOTICE - Chambal Infratech Ltd. (ISIN: INE3WY7Z6012). Rights entitlement: 7 equity shares for every 20 "
     "held; issue price Rs. 90 each (including premium). Record date 15 January 2027."),
    ("H09", "BUYBACK", "Ghatprabha Steels Limited", "INE1XZ5A9016", "2027-01-22", "NA",
     {"price": "1250", "size_cr": "400"},
     "Ghatprabha Steels Limited (ISIN INE1XZ5A9016): The Board approved the buyback of equity shares through "
     "the tender offer route at a price of Rs. 1,250/- per equity share, for a maximum aggregate amount of "
     "Rs. 400 crore. Record date for determining entitlement: 22 January 2027."),
    ("H10", "BUYBACK", "Tapti Power Ltd.", "INE0YA6B1013", "2027-01-29", "NA",
     {"price": "310", "size_cr": "75"},
     "Tapti Power Ltd. (INE0YA6B1013) buy-back announcement: price Rs 310 per share; size Rs. 75 crore "
     "(excluding costs); tender offer; record date 29/01/2027."),
    ("H11", "MERGER", "Himalaya Foods Limited", "INE5ZB4C2019", "2027-02-05", "NA", {"ratio": "3:8"},
     "Scheme of Amalgamation: Himalaya Foods Limited (INE5ZB4C2019) with Sutlej Foods Ltd. Share exchange "
     "ratio: 3 equity shares of Himalaya Foods Limited for every 8 equity shares held in Sutlej Foods Ltd. "
     "Record date for the allotment of shares: 5 February 2027."),
    ("H12", "DIVIDEND", "Vindhya Chemicals Ltd", "INE6AC8D3010", "2027-02-26", "2027-02-26", {"amount": "4"},
     "ALERT: Vindhya Chemicals Ltd INE6AC8D3010 - Dividend Rs 4 per share. RD 26-Feb-2027. XD 26-Feb-2027."),
    ("H13", "MERGER", "Narmada Finserv Limited", "INE7BD9E4011", "2027-03-12", "NA", {"ratio": "9:10"},
     "Arrangement between Narmada Finserv Limited (ISIN INE7BD9E4011) and Godavari Finserv Ltd.: shareholders "
     "of Godavari Finserv Ltd. will be allotted 9 equity shares of Narmada Finserv Limited for every 10 equity "
     "shares held. The record date is 12 March 2027."),
]


def build_rows():
    rows = []
    for id_, event, company, isin, record, ex, extra, text in NOTICES:
        fields = {"event_type": event, "company": company, "isin": isin,
                  "record_date": record, "ex_date": ex, **extra}
        rows.append({"id": id_, "event_type": event, "text": text, "fields": fields,
                     "target": format_target(fields)})
    return rows


def main():
    out = Path("data/handwritten.jsonl")
    out.parent.mkdir(exist_ok=True)
    rows = build_rows()
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} hand-written notices to {out}")


if __name__ == "__main__":
    main()
