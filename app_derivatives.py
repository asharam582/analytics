"""Streamlit UI for Live Options/Derivatives Analytics.

Displays option chain analytics (PCR, v-PCR, OI walls, S/R, Max Pain), most
actives, and educational suggested strategies. Supports optional event filter
integration with market_events.
"""

from __future__ import annotations

import math
from typing import List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import derivatives_analytics as da


st.set_page_config(page_title="NSE Derivatives Analytics", layout="wide")


@st.cache_data(ttl=60)
def _cached_chain(type_: str, symbol: str, expiry: str) -> pd.DataFrame:
    return da.fetch_option_chain(type_, symbol, expiry)


@st.cache_data(ttl=60)
def _cached_calls() -> pd.DataFrame:
    return da.fetch_most_active_calls_stocks()


@st.cache_data(ttl=60)
def _cached_puts() -> pd.DataFrame:
    return da.fetch_most_active_puts_stocks()


@st.cache_data(ttl=60)
def _cached_basket() -> dict:
    return da.fetch_most_active_oi()


@st.cache_data(ttl=60)
def _cached_contracts(limit: int = 20) -> pd.DataFrame:
    return da.fetch_most_active_contracts(limit)


def _render_metrics(df: pd.DataFrame):
    enr = da.enrich_chain_with_totals(df)
    pcr, v_pcr = da.compute_pcr(enr)
    mp, tbl = da.max_pain(enr)
    sr = da.supports_resistances(enr)
    c1, c2, c3 = st.columns(3)
    c1.metric("PCR", f"{pcr:.2f}" if math.isfinite(pcr) else "NA")
    c2.metric("v-PCR", f"{v_pcr:.2f}" if math.isfinite(v_pcr) else "NA")
    c3.metric("Max Pain", f"{mp:.2f}" if math.isfinite(mp) else "NA")
    c4, c5 = st.columns(2)
    c4.metric("Support (Put Wall)", f"{sr.get('support', float('nan')):.2f}" if sr else "NA")
    c5.metric("Resistance (Call Wall)", f"{sr.get('resistance', float('nan')):.2f}" if sr else "NA")

    # Plot max pain payout curve
    if tbl is not None and not tbl.empty:
        chart = alt.Chart(tbl).mark_line().encode(x="strike:Q", y="total_payout:Q").properties(height=240)
        st.altair_chart(chart, use_container_width=True)
    return enr


def _per_strike_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("Option chain is empty. NSE may be rate limiting. Retry later.")
        return
    cols = [
        "strike", "underlying_value", "ce_oi", "pe_oi", "ce_volume", "pe_volume",
        "ce_ltp", "pe_ltp", "ce_iv", "pe_iv",
    ]
    view = df[[c for c in cols if c in df.columns]].copy()
    st.dataframe(view, use_container_width=True, height=460)


def main():
    st.title("NSE Derivatives / Options Analytics")
    st.caption("For education only. Not investment advice.")

    with st.sidebar:
        st.header("Controls")
        type_ = st.selectbox("Underlying Type", ["Indices", "Stocks"], index=0)
        symbol = st.text_input("Symbol", value="NIFTY" if type_ == "Indices" else "RELIANCE")
        expiry = st.text_input("Expiry (e.g., 04-Nov-2025)", value="04-Nov-2025")
        use_event_filter = st.checkbox("Use event filter (stocks only)", value=False)
        st.markdown("---")
        if st.button("Refresh", use_container_width=True):
            _cached_chain.clear()
            _cached_calls.clear()
            _cached_puts.clear()
            _cached_basket.clear()
            _cached_contracts.clear()
            st.experimental_rerun()

    tabs = st.tabs([
        "Option Chain Analytics",
        "Most Actives",
        "Suggested Structures (Edu)",
    ])

    # Option Chain Analytics
    with tabs[0]:
        try:
            chain = _cached_chain(type_, symbol.strip().upper(), expiry.strip())
        except Exception as exc:
            st.warning(f"Option chain unavailable: {exc}. Retry / Check after a minute.")
            chain = pd.DataFrame()
        if chain.empty:
            st.warning("No chain data. Ensure symbol/expiry are valid; NSE may be limiting.")
        else:
            st.caption(f"Timestamp: {chain.get('timestamp', pd.Timestamp.utcnow())}")
            enr = _render_metrics(chain)
            _per_strike_table(enr)

    # Most Actives
    with tabs[1]:
        st.subheader("Most Active Derivatives")
        c1, c2 = st.columns(2)
        calls_df = _cached_calls()
        puts_df = _cached_puts()
        c1.write("Calls (Stocks) by Volume")
        c1.dataframe(calls_df if not calls_df.empty else pd.DataFrame(), use_container_width=True, height=300)
        c2.write("Puts (Stocks) by Volume")
        c2.dataframe(puts_df if not puts_df.empty else pd.DataFrame(), use_container_width=True, height=300)
        bask = _cached_basket()
        st.markdown("---")
        colA, colB = st.columns(2)
        vol_df = bask.get("volume", pd.DataFrame()) if isinstance(bask, dict) else pd.DataFrame()
        val_df = bask.get("value", pd.DataFrame()) if isinstance(bask, dict) else pd.DataFrame()
        colA.write("Most Active by Volume (Basket)")
        colA.dataframe(vol_df, use_container_width=True, height=260)
        colB.write("Most Active by Value (Basket)")
        colB.dataframe(val_df, use_container_width=True, height=260)
        st.markdown("---")
        mac = _cached_contracts(20)
        st.write("Most Active Contracts")
        st.dataframe(mac, use_container_width=True, height=300)

    # Suggested Structures
    with tabs[2]:
        st.subheader("Suggested Structures (Educational)")
        st.caption("For education only. Not investment advice.")
        if type_ == "Indices":
            chain = _cached_chain(type_, symbol.strip().upper(), expiry.strip())
            if chain.empty:
                st.warning("No chain for strategies.")
            else:
                idea = da.generate_index_intraday_spreads(chain)
                if idea is None or idea.empty:
                    st.info("No index spread signal based on current heuristics.")
                else:
                    st.write("CE Debit Spread Proposal (Index)")
                    st.dataframe(idea, use_container_width=True)
                    st.write(
                        f"Indicative debit: {idea.attrs.get('debit', float('nan')):.2f} | "
                        f"Max loss: {idea.attrs.get('max_loss', float('nan')):.2f} | "
                        f"Max gain: {idea.attrs.get('max_gain', float('nan')):.2f}"
                    )
        else:  # Stocks
            if use_event_filter:
                try:
                    import market_events as me

                    ann = me.fetch_corporate_announcements()
                    brd = me.fetch_board_meetings()
                    act = me.fetch_corporate_actions()
                    res = me.fetch_financial_results("insurance", "Quarterly")
                    ranked = me.score_events(ann, brd, act, res)
                    event_syms: List[str] = ranked["symbol"].dropna().astype(str).tolist() if not ranked.empty else []
                except Exception as exc:
                    st.warning(f"Event filter unavailable: {exc}")
                    event_syms = []
                calls_df = _cached_calls()
                puts_df = _cached_puts()
                universe = pd.concat([calls_df, puts_df], ignore_index=True) if not calls_df.empty or not puts_df.empty else pd.DataFrame()
                ideas = da.generate_event_driven_stock_spreads(event_syms, universe_df=universe)
                if ideas is None or ideas.empty:
                    st.info("No event-driven spread suggestions (may not be in F&O or insufficient data).")
                else:
                    st.write("Event-Driven Stock Spreads (Verticals)")
                    st.dataframe(ideas, use_container_width=True)
            else:
                st.info("Enable 'Use event filter' in sidebar to generate stock spreads based on events.")

    st.info("For education only. Not investment advice.")


if __name__ == "__main__":
    main()

