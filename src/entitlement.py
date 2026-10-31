"""Entitlement arithmetic from extracted notice fields (illustrative: ignores tax and fractional shares)."""
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

SUPPORTED = ("DIVIDEND", "BONUS_ISSUE", "STOCK_SPLIT", "RIGHTS_ISSUE")

UNSUPPORTED_NOTES = {
    "BUYBACK": ("Not calculated: buy-back acceptance depends on the entitlement ratio, how many "
                "holders tender, and the reserved small-shareholder category."),
    "MERGER": ("Not calculated: share-swap treatment depends on the scheme's terms and its "
               "rounding rules."),
}


@dataclass
class Entitlement:
    supported: bool
    headline: str = ""
    lines: list = field(default_factory=list)      # (label, value) pairs
    warnings: list = field(default_factory=list)
    error: str = ""


def _group_indian(whole: str) -> str:
    """'400000' -> '4,00,000' (lakh/crore grouping)."""
    last3, rest, parts = whole[-3:], whole[:-3], []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return ",".join(parts + [last3])


def indian(n) -> str:
    """Integer with Indian digit grouping."""
    d = Decimal(str(n))
    sign = "-" if d < 0 else ""
    return sign + _group_indian(f"{abs(d):.0f}")


def inr(amount) -> str:
    """Rupee amount with Indian grouping; decimals shown only when needed."""
    d = Decimal(str(amount))
    sign = "-" if d < 0 else ""
    whole, _, frac = f"{abs(d):.2f}".partition(".")
    text = f"{sign}Rs. {_group_indian(whole)}"
    return text if frac == "00" else f"{text}.{frac}"


def _plain(d: Decimal) -> str:
    return format(d.normalize(), "f")


def _dec(value, name) -> Decimal:
    try:
        d = Decimal(str(value).strip().replace(",", ""))
    except (InvalidOperation, AttributeError):
        raise ValueError(f"'{name}' is missing or not a number (got {value!r}).")
    if d <= 0:
        raise ValueError(f"'{name}' must be greater than zero (got {value!r}).")
    return d


def _ratio(value):
    try:
        a, b = str(value).split(":")
        a, b = int(a), int(b)
    except Exception:
        raise ValueError(f"ratio {value!r} is not in the form a:b.")
    if a <= 0 or b <= 0:
        raise ValueError(f"ratio {value!r} must have positive parts.")
    return a, b


def _dividend(f, holding):
    amount = _dec(f.get("amount"), "amount")
    cash = amount * holding
    return Entitlement(
        True, f"{inr(cash)} cash due, before tax",
        [("Shares held on the record date", indian(holding)),
         ("Dividend per share", inr(amount)),
         ("Cash due (before tax)", inr(cash))],
        ["Tax deducted at source is not applied here."])


def _bonus(f, holding):
    a, b = _ratio(f.get("ratio"))
    new, left = divmod(holding * a, b)
    warnings = []
    if left:
        warnings.append(f"{left}/{b} of a share is left over. Fractional-share treatment (cash "
                        "payout or rounding) depends on the company, so check the notice.")
    return Entitlement(
        True, f"{indian(new)} new shares; {indian(holding + new)} in total after the bonus",
        [("Shares held on the record date", indian(holding)),
         ("Bonus ratio", f"{a} new for every {b} held"),
         ("New bonus shares (rounded down)", indian(new)),
         ("Total after the bonus", indian(holding + new))],
        warnings)


def _split(f, holding):
    old, new = _dec(f.get("old_fv"), "old_fv"), _dec(f.get("new_fv"), "new_fv")
    factor = old / new
    qty = holding * factor
    qty_int = int(qty)
    warnings = ["Quantity rises and the price falls in proportion, so the position value is unchanged."]
    if qty != qty_int:
        warnings.append("The multiplier is not a whole number, so the quantity is rounded down.")
    return Entitlement(
        True, f"{indian(qty_int)} shares after the split ({_plain(factor)}x)",
        [("Shares held on the record date", indian(holding)),
         ("Face value change", f"Rs. {_plain(old)} to Rs. {_plain(new)}"),
         ("Quantity multiplier", f"{_plain(factor)}x"),
         ("Shares after the split", indian(qty_int)),
         ("Price adjusts to about", f"1/{_plain(factor)} of the previous price")],
        warnings)


def _rights(f, holding):
    a, b = _ratio(f.get("ratio"))
    price = _dec(f.get("price"), "price")
    entitled = (holding * a) // b
    cost = price * entitled
    warnings = ["Renunciation, partial subscription and fractional entitlements are ignored."]
    if entitled == 0:
        warnings.append("This holding is too small to be entitled to any rights shares.")
    return Entitlement(
        True, f"Entitled to {indian(entitled)} rights shares; {inr(cost)} to subscribe in full",
        [("Shares held on the record date", indian(holding)),
         ("Rights ratio", f"{a} new for every {b} held"),
         ("Rights shares entitled (rounded down)", indian(entitled)),
         ("Issue price per share", inr(price)),
         ("Cost to subscribe in full", inr(cost))],
        warnings)


def calculate(fields: dict, holding: int) -> Entitlement:
    """Entitlement for `holding` shares given the extracted notice fields."""
    event = fields.get("event_type")
    if event in UNSUPPORTED_NOTES:
        return Entitlement(False, error=UNSUPPORTED_NOTES[event])
    if event not in SUPPORTED:
        return Entitlement(False, error="No calculation is available because the event type was not recognised.")
    if holding < 0:
        return Entitlement(False, error="The holding cannot be negative.")
    try:
        if event == "DIVIDEND":
            return _dividend(fields, holding)
        if event == "BONUS_ISSUE":
            return _bonus(fields, holding)
        if event == "STOCK_SPLIT":
            return _split(fields, holding)
        return _rights(fields, holding)
    except ValueError as e:
        return Entitlement(False, error=f"Could not calculate: {e}")
