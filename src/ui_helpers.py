"""Presentation helpers for app.py. No Streamlit import, so they are easy to test."""
from datetime import date

from src.entitlement import inr
from src.schema import EVENT_TYPES, FIELD_ORDER

EVENT_LABELS = {
    "DIVIDEND": "Dividend",
    "STOCK_SPLIT": "Stock split",
    "BONUS_ISSUE": "Bonus issue",
    "RIGHTS_ISSUE": "Rights issue",
    "BUYBACK": "Buy-back",
    "MERGER": "Merger",
}

FIELD_LABELS = {
    "event_type": "Event type",
    "company": "Company",
    "isin": "ISIN",
    "record_date": "Record date",
    "ex_date": "Ex-date",
    "amount": "Dividend per share (Rs.)",
    "old_fv": "Face value before (Rs.)",
    "new_fv": "Face value after (Rs.)",
    "ratio": "Ratio",
    "price": "Price (Rs.)",
    "size_cr": "Buy-back size (Rs. crore)",
}

REQUIRED = {
    "DIVIDEND": ["amount"],
    "STOCK_SPLIT": ["old_fv", "new_fv"],
    "BONUS_ISSUE": ["ratio"],
    "RIGHTS_ISSUE": ["ratio", "price"],
    "BUYBACK": ["price", "size_cr"],
    "MERGER": ["ratio"],
}

EXAMPLE_QUESTIONS = [
    "What advance notice is required before the record date of a rights issue?",
    "How soon after the record date can bonus shares be traded?",
    "How long must a buy-back tender offer stay open?",
    "Within how many working days must a rights issue be completed after board approval?",
]


def _is_iso(value) -> bool:
    try:
        date.fromisoformat(str(value))
        return True
    except ValueError:
        return False


def pretty_date(iso) -> str:
    """'2026-11-20' -> '20 Nov 2026'; anything else is returned unchanged."""
    if _is_iso(iso):
        d = date.fromisoformat(str(iso))
        return f"{d.day} {d.strftime('%b %Y')}"
    return str(iso)


def derived_ex_date(fields):
    """(value, is_derived). Under T+1 the ex-date is normally the record date (verify)."""
    ex, rec = fields.get("ex_date", "NA"), fields.get("record_date")
    if ex and ex != "NA":
        return ex, False
    if rec:
        return rec, True
    return None, False


def _ratio_text(value):
    """'2:5' -> '2 : 5' (short enough for a metric card)."""
    try:
        a, b = str(value).split(":")
        return f"{int(a)} : {int(b)}"
    except Exception:
        return str(value)


def _money(value):
    try:
        return inr(value)
    except Exception:
        return str(value)


def key_figures(fields):
    """The headline numbers for each event type, as (label, value) pairs. Values are kept short."""
    ev = fields.get("event_type")
    f = fields
    if ev == "DIVIDEND" and "amount" in f:
        return [("Dividend per share", _money(f["amount"]))]
    if ev == "STOCK_SPLIT" and "old_fv" in f and "new_fv" in f:
        return [("Face value (Rs.)", f"{f['old_fv']} to {f['new_fv']}")]
    if ev == "BONUS_ISSUE" and "ratio" in f:
        return [("Bonus ratio (new : held)", _ratio_text(f["ratio"]))]
    if ev == "RIGHTS_ISSUE":
        out = []
        if "ratio" in f:
            out.append(("Rights ratio (new : held)", _ratio_text(f["ratio"])))
        if "price" in f:
            out.append(("Issue price", _money(f["price"])))
        return out
    if ev == "BUYBACK":
        out = []
        if "price" in f:
            out.append(("Buy-back price", _money(f["price"])))
        if "size_cr" in f:
            out.append(("Size (Rs. crore)", str(f["size_cr"])))
        return out
    if ev == "MERGER" and "ratio" in f:
        return [("Swap ratio (new : held)", _ratio_text(f["ratio"]))]
    return []


def validate(fields):
    """Plain-language warnings when the model output looks incomplete or odd."""
    warns = []
    ev = fields.get("event_type")
    if not fields:
        return ["The model returned nothing usable. Check that the text is a corporate action notice."]
    if ev not in EVENT_TYPES:
        warns.append("The event type was not recognised. Check the notice text.")
    if not fields.get("company"):
        warns.append("No company name was extracted.")
    isin = fields.get("isin")
    if not isin:
        warns.append("No ISIN was extracted.")
    elif len(isin) != 12 or not isin.startswith("IN"):
        warns.append("The ISIN looks malformed (Indian ISINs have 12 characters starting with IN). Verify it against the notice.")
    rec = fields.get("record_date")
    if not rec:
        warns.append("No record date was extracted.")
    elif not _is_iso(rec):
        warns.append("The record date is not a valid date. Verify it against the notice.")
    for key in REQUIRED.get(ev, []):
        if not fields.get(key):
            warns.append(f"Missing: {FIELD_LABELS[key].lower()}.")
    return warns


def field_rows(fields):
    """Ordered (label, value) rows with friendly names and dates."""
    rows = []
    for key in FIELD_ORDER:
        if key not in fields:
            continue
        value = fields[key]
        if key == "event_type":
            value = EVENT_LABELS.get(value, value)
        elif key in ("record_date", "ex_date"):
            value = "Not stated in the notice" if value == "NA" else pretty_date(value)
        rows.append((FIELD_LABELS[key], value))
    return rows


def friendly_llm_error(exc) -> str:
    """Short, plain-language reason for a failed language-model call."""
    text = str(exc)
    low = text.lower()
    if "402" in text:
        return "the account has no inference credits left"
    if "401" in text or "403" in text:
        return "the key is invalid or lacks permission for this model"
    if "429" in text:
        return "rate limit reached, try again shortly"
    if "404" in text:
        return "the model name was not found for this provider"
    if "503" in text or "overloaded" in low or "high demand" in low:
        return "the model is busy right now, try again in a minute"
    if "timeout" in low or "timed out" in low:
        return "the request timed out"
    first = text.splitlines()[0] if text else type(exc).__name__
    return first[:120]
