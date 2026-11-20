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


def redact_text(text: str, extra: dict[str, str] | None = None) -> str:
    """Replace matches with [LABEL]. `extra` maps label -> regex."""
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
