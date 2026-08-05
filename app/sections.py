"""Streamlit renderers, one per report tab. Every layout decision lives here;
all data shaping lives in app.ui_helpers so it stays unit-testable."""
import streamlit as st

from app.ui_helpers import (DASH, badge_html, event_rows, fmt_market_cap,
                            fmt_num, group_sources, guidance_rows, kv_rows,
                            move_chip_html, news_rows, peer_rows)

_PCT = st.column_config.NumberColumn(format="%.2f%%")

_VALUATION_LABELS = [("pe", "P/E", ""), ("pb", "P/B", ""), ("roe", "ROE", "%"),
                     ("roce", "ROCE", "%"), ("debt_equity", "Debt / Equity", ""),
                     ("dividend_yield", "Dividend yield", "%")]
_PRICE_LABELS = [("price", "Price", "₹"), ("high_52w", "52-week high", "₹"),
                 ("low_52w", "52-week low", "₹"), ("ret_1m", "1-month return", "%"),
                 ("ret_6m", "6-month return", "%"), ("ret_1y", "1-year return", "%"),
                 ("mktcap_cr", "Market cap", "cr")]
_HOLDING_LABELS = [("promoter", "Promoter", "%"), ("fii", "FII", "%"),
                   ("dii", "DII", "%"), ("public", "Public", "%")]


def _table(rows: list, column_config: dict = None) -> None:
    """st.dataframe with sensible defaults; silent no-op on empty rows."""
    if not rows:
        return
    st.dataframe(rows, use_container_width=True, hide_index=True,
                 column_config=column_config or {})


def render_run_detail(run: dict) -> None:
    """Collapsed agent diagnostics: status, attempts, items fetched, reasons."""
    if not run:
        return
    with st.expander("Run detail", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", run.get("status", "unknown"))
        c2.metric("Attempts", run.get("attempts", 0))
        fetched = run.get("fetch_count", -1)
        c3.metric("Items fetched", DASH if fetched is None or fetched < 0
                  else fetched)
        for reason in run.get("failure_reasons", []):
            st.caption(f"⚠️ {reason}")


def render_fundamentals(f, generated_at: str) -> None:
    if f is None:
        st.info("Data unavailable for this section.")
        return
    profile = f.company_profile or {}
    chips = "".join(f"<span class='chip'>{profile[k]}</span>"
                    for k in ("sector", "industry") if profile.get(k))
    if chips:
        st.markdown(chips, unsafe_allow_html=True)
    st.markdown(f.summary)
    if profile.get("description"):
        st.caption(profile["description"])
    st.markdown("#### Valuation")
    _table(kv_rows(f.valuation, _VALUATION_LABELS))
    st.markdown("#### Price snapshot")
    st.caption(f"As captured at generation · {generated_at}")
    _table(kv_rows(f.price_snapshot, _PRICE_LABELS))
    st.markdown("#### Shareholding")
    _table(kv_rows(f.shareholding, _HOLDING_LABELS))


def render_competitors(c, fundamentals, ticker: str, company_name: str) -> None:
    if c is None:
        st.info("Data unavailable for this section.")
        return
    st.markdown(f"Overall standing: {badge_html(c.overall_standing, 'standing')}",
                unsafe_allow_html=True)
    st.markdown(c.comparison_summary)
    st.markdown("#### Peer comparison")
    _table(peer_rows(c, fundamentals, ticker, company_name),
           {"P/E": st.column_config.NumberColumn(format="%.2f"),
            "ROE %": st.column_config.NumberColumn(format="%.2f"),
            "D/E": st.column_config.NumberColumn(format="%.2f"),
            "MCap (Cr)": st.column_config.NumberColumn(format="%.0f"),
            "1M %": _PCT, "3M %": _PCT, "6M %": _PCT})
    if not c.peers:
        st.caption("No peers were identified for this company.")
    for p in c.peers:
        with st.container(border=True):
            st.markdown(
                f"<div class='cardtitle'>{p.name} "
                f"<span class='pill'>{p.ticker}</span></div>"
                f"<div class='cardmeta'>Competition "
                f"{badge_html(p.competition_intensity, 'intensity')} · Target "
                f"{badge_html(p.target_standing, 'standing')}</div>",
                unsafe_allow_html=True)
            st.markdown(p.reason_for_inclusion)
            m = p.metrics or {}
            st.markdown(
                f"<span class='chip'>P/E {fmt_num(m.get('pe'))}</span>"
                f"<span class='chip'>ROE {fmt_num(m.get('roe'), suffix='%')}</span>"
                f"<span class='chip'>D/E {fmt_num(m.get('debt_equity'))}</span>"
                f"<span class='chip'>MCap "
                f"{fmt_market_cap((m.get('mktcap_cr') or 0) * 1e7 or None)}</span>",
                unsafe_allow_html=True)


def render_events(e) -> None:
    if e is None:
        st.info("Data unavailable for this section.")
        return
    if e.highlights:
        st.markdown("#### Highlights")
        for h in e.highlights:
            with st.container(border=True):
                st.markdown(f"<div class='bullet'>{h}</div>",
                            unsafe_allow_html=True)
    st.markdown("#### Timeline")
    _table(event_rows(e), {"1D %": _PCT, "5D %": _PCT})
    if not e.events:
        st.caption("No corporate events were found in the last ~90 days.")
    for ev in e.events:
        with st.container(border=True):
            st.markdown(
                f"<span class='evtdate'>{ev.date}</span> "
                f"{badge_html(ev.type, 'type')} "
                f"{badge_html(ev.significance, 'significance')}",
                unsafe_allow_html=True)
            st.markdown(f"**{ev.summary}**")
            st.markdown(f"<div class='bullet'>{ev.what_it_meant}</div>"
                        f"<div class='bullet'>{ev.how_it_affected}</div>",
                        unsafe_allow_html=True)
            chips = move_chip_html("1D", ev.price_move_1d) + \
                move_chip_html("5D", ev.price_move_5d)
            if chips:
                st.markdown(chips, unsafe_allow_html=True)
            if ev.filing_ref:
                st.caption(ev.filing_ref)


def render_news(n) -> None:
    if n is None:
        st.info("Data unavailable for this section.")
        return
    st.markdown(f"Overall sentiment: "
                f"{badge_html(n.overall_sentiment, 'sentiment')}",
                unsafe_allow_html=True)
    st.markdown(n.narrative)
    if not n.items:
        st.caption("No relevant recent articles were found.")
        return
    st.markdown("#### Articles")
    _table(news_rows(n),
           {"Score": st.column_config.NumberColumn(format="%.2f"),
            "Link": st.column_config.LinkColumn("Link", display_text="open")})
    for it in n.items:
        with st.container(border=True):
            st.markdown(f"<div class='cardtitle'>"
                        f"<a href='{it.source_url}' target='_blank'>{it.title}</a>"
                        f"</div><div class='cardmeta'>{it.date} · "
                        f"{badge_html(it.sentiment, 'sentiment')} "
                        f"<span class='chip'>{it.sentiment_score:+.2f}</span>"
                        f"</div>", unsafe_allow_html=True)
            st.markdown(it.summary)
            st.markdown(f"<div class='bullet'><b>Stock:</b> {it.impact_on_stock}"
                        f"</div><div class='bullet'><b>Sector:</b> "
                        f"{it.sector_impact}</div>", unsafe_allow_html=True)


def render_docs(d) -> None:
    if d is None:
        st.info("Data unavailable for this section.")
        return
    st.markdown(f"Management tone: {badge_html(d.management_tone, 'tone')}",
                unsafe_allow_html=True)
    st.caption(d.tone_trend)
    st.markdown(d.narrative)
    st.markdown("#### Guidance")
    _table(guidance_rows(d))
    if not d.guidance:
        st.caption("No forward guidance was found in the documents.")
    for g in d.guidance:
        with st.container(border=True):
            st.markdown(f"<div class='cardtitle'>{g.metric} — {g.value}</div>"
                        f"<div class='cardmeta'>{g.period} · {g.source}</div>"
                        f"<div class='quote'>“{g.quote}”</div>",
                        unsafe_allow_html=True)
    st.markdown("#### Risks")
    if not d.risks:
        st.caption("No risks were flagged in the documents.")
    for r in d.risks:
        with st.container(border=True):
            st.markdown(f"<div class='cardtitle'>{r.risk}</div>"
                        f"<div class='cardmeta'>{r.source}</div>"
                        f"<div class='quote'>“{r.quote}”</div>",
                        unsafe_allow_html=True)
    if d.strategy_highlights:
        st.markdown("#### Strategy")
        st.markdown("".join(f"<span class='chip'>{s}</span>"
                            for s in d.strategy_highlights),
                    unsafe_allow_html=True)


def render_sources(specialists: dict) -> None:
    groups = group_sources(specialists)
    if not groups:
        st.info("No sources were recorded for this report.")
        return
    for title, refs in groups:
        with st.container(border=True):
            st.markdown(f"<div class='grouptitle'>{title} "
                        f"<span class='pill'>{len(refs)}</span></div>",
                        unsafe_allow_html=True)
            for ref in refs:
                if ref.startswith("http"):
                    st.markdown(f"<div class='bullet srclink'>"
                                f"<a href='{ref}' target='_blank'>{ref}</a></div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='bullet'>{ref}</div>",
                                unsafe_allow_html=True)
