"""Pure (Streamlit-free) presentation helpers so they can be unit-tested."""
from pathlib import Path
from typing import Optional

_COLORS = {
    "AHEAD": "#1b7f3b", "INLINE": "#8a6d00", "BEHIND": "#b00020",
    "POSITIVE": "#1b7f3b", "NEUTRAL": "#555", "NEGATIVE": "#b00020",
    "HIGH": "#b00020", "MEDIUM": "#8a6d00", "LOW": "#1b7f3b",
    "confident": "#1b7f3b", "cautious": "#8a6d00", "defensive": "#b00020",
}
_NEUTRAL = "#555"


def badge_html(label: str, kind: str) -> str:
    color = _COLORS.get(label, _NEUTRAL)
    return (f"<span style='background:{color};color:#fff;border-radius:10px;"
            f"padding:2px 10px;font-size:12px;font-weight:600'>{label}</span>")


def chart_iframe_html(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def section_note(missing: list, key: str) -> Optional[str]:
    if key in missing:
        return "_Data unavailable for this section._"
    return None
