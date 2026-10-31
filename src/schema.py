"""Shared schema: event types, field order, and the text format the model learns.

Target format (no braces, because the T5 tokenizer has no '{' or '}'):
    event_type: DIVIDEND ; company: X Ltd ; isin: INE... ; record_date: 2026-11-20 ; ex_date: NA ; amount: 5.5
"""

EVENT_TYPES = [
    "DIVIDEND",
    "STOCK_SPLIT",
    "BONUS_ISSUE",
    "RIGHTS_ISSUE",
    "BUYBACK",
    "MERGER",
]

FIELD_ORDER = [
    "event_type",
    "company",
    "isin",
    "record_date",
    "ex_date",
    "amount",    # dividend per share (Rs.)
    "old_fv",    # split: face value before
    "new_fv",    # split: face value after
    "ratio",     # bonus / rights / merger  (new : held)
    "price",     # rights issue price or buyback price (Rs.)
    "size_cr",   # buyback size (Rs. crore)
]

PREFIX = "Extract corporate action details: "
SEP = " ; "


def format_target(fields: dict) -> str:
    return SEP.join(f"{k}: {fields[k]}" for k in FIELD_ORDER if k in fields)


def parse_output(text: str) -> dict:
    """Parse model output back into a dict. Unknown keys / malformed parts are ignored."""
    out = {}
    for part in text.split(";"):
        key, sep, val = part.partition(":")
        key = key.strip()
        if sep and key in FIELD_ORDER:
            out[key] = val.strip()
    return out


NUMERIC_FIELDS = ("amount", "old_fv", "new_fv", "price", "size_cr")


def normalize_for_scoring(key, value):
    """Formatting-tolerant form of a field value: 2.50 == 2.5, 1,250 == 1250, 'Ltd.' == 'Ltd'.
    It forgives presentation only; a wrong digit, date or name is still wrong."""
    from decimal import Decimal, InvalidOperation
    v = (value or "").strip()
    if key in NUMERIC_FIELDS:
        try:
            return format(Decimal(v.replace(",", "")).normalize(), "f")
        except InvalidOperation:
            return v
    if key == "company":
        return v.rstrip(".").strip().lower()
    return v
