"""Schema for two-wheeler finance scheme circulars (Part IV adaptation).
Drop-in replacement for src/schema.py in a COPY of the project (same function names)."""

FIELD_ORDER = ["financier", "scheme", "model", "zone", "rate", "proc_fee",
               "down_payment", "tenure", "cashback", "valid_from", "valid_to"]
# rate = % p.a. ; proc_fee = Rs. ; down_payment = % of on-road price ; tenure = months (max) ;
# cashback = Rs. ; dates = YYYY-MM-DD ; NA when the circular does not state the field
EVENT_TYPES = ["SCHEME"]          # kept so code that imports it still works
PREFIX = "Extract finance scheme details: "
SEP = " ; "
NUMERIC_FIELDS = ("rate", "proc_fee", "down_payment", "tenure", "cashback")


def format_target(fields: dict) -> str:
    return SEP.join(f"{k}: {fields[k]}" for k in FIELD_ORDER if k in fields)


def parse_output(text: str) -> dict:
    out = {}
    for part in text.split(";"):
        key, sep, val = part.partition(":")
        key = key.strip()
        if sep and key in FIELD_ORDER:
            out[key] = val.strip()
    return out


def normalize_for_scoring(key, value):
    from decimal import Decimal, InvalidOperation
    v = (value or "").strip()
    if key in NUMERIC_FIELDS:
        try:
            return format(Decimal(v.replace(",", "")).normalize(), "f")
        except InvalidOperation:
            return v
    if key in ("financier", "scheme", "model"):
        return v.rstrip(".").strip().lower()
    return v
