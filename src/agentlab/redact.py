"""Redact common identifiers from text before it is logged or exported.

Pattern-based and deliberately conservative: it catches obvious US SSNs,
emails and phone numbers, plus caller-supplied patterns. It is a safety net
for audit traces, not a compliance guarantee - do not feed it real PHI/PII.
"""
from __future__ import annotations

import re

DEFAULT_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "EMAIL": r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b",
    "PHONE": r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)",
}


# Optional extras for healthcare-style text. Pass as `extra=HEALTH_PATTERNS`.
# Formats are illustrative (synthetic IDs such as MBR-0042); real identifiers
# vary by organization, so adapt them and never treat this as de-identification.
HEALTH_PATTERNS = {
    "MEMBER_ID": r"\bMBR-\d{3,}\b",
    "MRN": r"\bMRN[ :#-]*\d{5,}\b",
    "DOB": r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b",
}


CARD_CANDIDATE = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")


def luhn_valid(digits: str) -> bool:
    """Luhn checksum, the check digit payment card numbers carry."""
    nums = [int(c) for c in digits][::-1]
    total = sum(nums[0::2]) + sum(sum(divmod(2 * n, 10)) for n in nums[1::2])
    return total % 10 == 0


def _mask_cards(text: str) -> str:
    """Mask 13-19 digit runs (spaces or dashes allowed) that pass the Luhn
    check. The checksum keeps order numbers and timestamps from being masked
    by accident, though about one in ten random digit runs still passes."""
    def repl(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "[CARD]" if 13 <= len(digits) <= 19 and luhn_valid(digits) else m.group(0)
    return CARD_CANDIDATE.sub(repl, text)


def redact_text(text: str, extra: dict[str, str] | None = None) -> str:
    """Replace matches with [LABEL]. `extra` maps label -> regex. Also masks
    Luhn-valid card numbers as [CARD]."""
    text = _mask_cards(text)
    for label, pattern in {**DEFAULT_PATTERNS, **(extra or {})}.items():
        text = re.sub(pattern, f"[{label}]", text)
    return text


def redact_value(value, extra: dict[str, str] | None = None):
    """Redact strings recursively inside dicts and lists; other types pass through."""
    if isinstance(value, str):
        return redact_text(value, extra)
    if isinstance(value, dict):
        return {k: redact_value(v, extra) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v, extra) for v in value]
    return value


def redaction_counts(text: str, extra: dict[str, str] | None = None) -> dict[str, int]:
    """How many matches each label would redact in `text` (cards as CARD),
    without changing it. For audit summaries that report what was masked, not
    the masked values. Labels with no matches are omitted; order is by label.
    Matches are counted against the text as scanned: cards first, then the
    other patterns on the card-masked text, same as redact_text."""
    counts: dict[str, int] = {}
    scanned = _mask_cards(text)
    cards = len(re.findall(r"\[CARD\]", scanned)) - len(re.findall(r"\[CARD\]", text))
    if cards:
        counts["CARD"] = cards
    for label, pattern in {**DEFAULT_PATTERNS, **(extra or {})}.items():
        n = len(re.findall(pattern, scanned))
        if n:
            counts[label] = counts.get(label, 0) + n
    return dict(sorted(counts.items()))
