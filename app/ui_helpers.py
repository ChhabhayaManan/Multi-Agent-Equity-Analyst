"""Pure (Streamlit-free) presentation helpers so they can be unit-tested."""
import re
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


def fmt_market_cap(value: Optional[float]) -> str:
    """INR absolute -> Indian crore/lakh-crore string."""
    if not value:
        return "—"
    cr = value / 1e7
    if cr >= 1e5:
        return f"₹{cr / 1e5:.2f} L Cr"
    return f"₹{cr:,.0f} Cr"


def fmt_num(value: Optional[float], prefix: str = "", suffix: str = "",
            dp: int = 2) -> str:
    if value is None:
        return "—"
    return f"{prefix}{value:,.{dp}f}{suffix}"


# --- Prose restructuring (parse the LLM's saved markdown into digestible bits).
# All best-effort + fail-soft: on any odd input they degrade to "one bullet =
# the whole text", never raise. The report render falls back to raw markdown.

_CITE_CUES = (
    "price snapshot", "valuation", "shareholding", "company profile", "summary",
    "concall", "annual report", "bse ann", "nse ann", "regulation", "press release",
    "postal ballot", "trading window", "credit rating", "board meeting",
    "fy2", "fy 2", " fy", "q1 fy", "q2 fy", "q3 fy", "q4 fy",
)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_LABELED_RE = re.compile(r"-\s+\*\*(.+?)\*\*\s*:?\s*")
# Citation parenthetical, tolerating one level of nested parens like
# "(BSE Ann. 2026-07-03: ... Reg. 74 (5) of SEBI (DP) Regulations)".
_CITE_PAREN_RE = re.compile(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)")
_MOVE_1D_RE = re.compile(r"(-?\d+(?:\.\d+)?)%\s+the next (?:trading )?day")
_MOVE_5D_RE = re.compile(r"(-?\d+(?:\.\d+)?)%\s+over the next five")
# Move clause carries decimals, so match on an allow-list of chars (no '@',
# which fences off any trailing citation marker) rather than "up to a period".
_MOVE_SENT_RE = re.compile(r"The stock price moved [\d\s\w%.,\-]*")
_ABBREV = {"rs", "no", "ltd", "inc", "corp", "vs", "fig", "dr", "mr", "co"}


def _looks_like_citation(inner: str) -> bool:
    if _DATE_RE.search(inner):
        return True
    low = inner.lower()
    return any(cue in low for cue in _CITE_CUES)


def extract_citations(text: str) -> tuple[str, list[str]]:
    """Pull citation-like parentheticals out of prose so it reads clean.
    Returns (text_with_@@n@@_markers, [citation]); n is 1-based, deduped.
    Non-citation parentheticals (e.g. '(LODR)') are left untouched."""
    citations: list[str] = []
    index: dict[str, int] = {}

    def repl(m: "re.Match") -> str:
        inner = m.group(1).strip()
        if not _looks_like_citation(inner):
            return m.group(0)
        markers = []
        for part in re.split(r"\s*;\s*", inner):
            part = part.strip()
            if not part:
                continue
            if part not in index:
                citations.append(part)
                index[part] = len(citations)
            markers.append(f"@@{index[part]}@@")
        return "".join(markers)

    clean = _CITE_PAREN_RE.sub(repl, text)
    clean = re.sub(r"\s+([,.;])", r"\1", clean)   # tidy space left before punct
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    return clean, citations


def split_bullets(text: str) -> list[str]:
    """Turn a prose blob into bullet strings. Honors existing '- ' markers;
    otherwise splits on sentence boundaries, repairing abbreviation/short splits."""
    text = re.sub(r"^#{1,6}\s+.*(?:\n|$)", "", text.strip()).strip()
    # Only line-start dashes are real bullets; an inline " - " (e.g. a
    # hyphenated name like "Johnson Controls - Hitachi") is not a list item.
    dashed = re.split(r"(?:^|\n)[ \t]*-\s+", text)
    dashed = [p.strip(" .") for p in dashed if p.strip(" .-")]
    if len(dashed) > 1:
        return dashed
    raw = re.split(r"(?<=[.!?@])\s+(?=[A-Z])", text)
    out: list[str] = []
    for seg in (s.strip() for s in raw):
        if not seg:
            continue
        tail = out[-1].rstrip(".").split()[-1].lower() if out else ""
        if out and (len(out[-1]) < 25 or tail in _ABBREV):
            out[-1] = f"{out[-1]} {seg}"
        else:
            out.append(seg)
    return out or [text]


def split_labeled(text: str) -> tuple[str, list[tuple[str, str]]]:
    """For '- **Label**: body' prose (events use dates as labels, docs use
    categories). Returns (intro_before_first_label, [(label, body)])."""
    text = re.sub(r"^#{1,6}\s+.*(?:\n|$)", "", text.strip()).strip()
    matches = list(_LABELED_RE.finditer(text))
    if not matches:
        return text, []
    intro = text[:matches[0].start()].strip().rstrip(":").strip()
    groups: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip().rstrip(". ").strip()
        groups.append((m.group(1).strip(), body))
    return intro, groups


def parse_price_moves(body: str) -> tuple[Optional[float], Optional[float]]:
    """Extract (1-day %, 5-day %) from an event body's 'moved X%...' sentence."""
    d1 = _MOVE_1D_RE.search(body)
    d5 = _MOVE_5D_RE.search(body)
    return (float(d1.group(1)) if d1 else None,
            float(d5.group(1)) if d5 else None)


def strip_move_sentence(body: str) -> str:
    """Drop the 'The stock price moved …' clause; it becomes chips instead."""
    return re.sub(r"\s*\.?\s*" + _MOVE_SENT_RE.pattern, "", body).strip(" .")


def prose_html(text: str, citations: list[str]) -> str:
    """Swap @@n@@ markers for styled <sup> chips carrying the source as tooltip."""
    def sup(m: "re.Match") -> str:
        n = int(m.group(1))
        title = citations[n - 1] if 0 < n <= len(citations) else ""
        title = title.replace('"', "&quot;")
        return f"<sup class='cite' title=\"{title}\">{n}</sup>"
    return re.sub(r"@@(\d+)@@", sup, text)


def move_chip_html(label: str, pct: Optional[float]) -> str:
    """Green/red % pill for an event price move; '' when the move is unknown."""
    if pct is None:
        return ""
    color = "#3fb950" if pct >= 0 else "#f0616d"
    sign = "+" if pct >= 0 else ""
    return (f"<span class='movechip' style='background:{color}1f;color:{color};"
            f"border:1px solid {color}55'>{label} {sign}{pct:.2f}%</span>")


def css_block() -> str:
    """Shared 'fintech-pro' polish. Injected once per page via st.markdown.
    Light-touch: styles native widgets (metrics/tabs/containers), no layout
    hacks, so a Streamlit internals change degrades gracefully to plain."""
    return """<style>
.block-container {padding-top: 2.6rem; max-width: 1120px;}
[data-testid="stMetric"] {
  background: #161b26; border: 1px solid #26304a;
  border-radius: 12px; padding: 14px 16px;
}
[data-testid="stMetricLabel"] p {opacity: .65; font-size: .78rem;
  letter-spacing: .03em; text-transform: uppercase;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
div[data-baseweb="tab-list"] {gap: 6px; border-bottom: 1px solid #26304a;}
div[data-baseweb="tab-border"] {display: none;}
div[data-baseweb="tab-highlight"] {background: transparent;}
button[role="tab"] {
  background: #161b26; border: 1px solid #26304a; border-bottom: none;
  border-radius: 10px 10px 0 0; padding: 9px 18px; margin-bottom: -1px;
}
button[role="tab"] p {font-weight: 600; font-size: 0.95rem;}
button[role="tab"][aria-selected="true"] {
  background: #1f2a44; border-color: #3a4a6b;
}
button[role="tab"][aria-selected="true"] p {color: #8fb4ff;}
[data-testid="stExpander"] details {border-radius: 10px; border-color: #26304a;}
hr {border-color: #26304a;}
.pill {display:inline-block;background:#26304a;color:#c7d2fe;border-radius:8px;
  padding:1px 8px;font-size:11px;font-weight:600;margin-left:6px;}
sup.cite {background:#26304a;color:#8fb4ff;border-radius:5px;padding:0 4px;
  font-size:9px;font-weight:700;margin:0 1px 0 2px;cursor:help;vertical-align:super;}
.bullet {position:relative;padding-left:18px;margin:7px 0;line-height:1.6;}
.bullet:before {content:'▸';position:absolute;left:2px;color:#4c8dff;}
.movechip {display:inline-block;border-radius:6px;padding:1px 8px;font-size:11px;
  font-weight:600;margin:4px 6px 0 0;}
.grouptitle {font-weight:700;color:#cfe0ff;font-size:0.92rem;margin-bottom:2px;}
.evtdate {display:inline-block;font-weight:700;color:#8fb4ff;font-size:0.86rem;
  letter-spacing:.02em;}
.muted {color:#9aa4b8;font-size:0.9rem;margin:2px 0 10px;line-height:1.55;}
.srcfoot {margin-top:12px;padding-top:8px;border-top:1px solid #26304a;
  font-size:11px;color:#7a8699;line-height:2;}
.srcfoot .cite {margin-right:4px;}
</style>"""
