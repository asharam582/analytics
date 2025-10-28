"""Streamlit UI for NSE Corporate/Event Intelligence.

Displays announcements, board meetings, corporate actions, financial results,
and a ranked watchlist derived from event scoring.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

import market_events as me


st.set_page_config(page_title="NSE Market Events", layout="wide")


@st.cache_data(ttl=60)
def _cached_announcements() -> pd.DataFrame:
    return me.fetch_corporate_announcements()


@st.cache_data(ttl=60)
def _cached_board_meetings() -> pd.DataFrame:
    return me.fetch_board_meetings()


@st.cache_data(ttl=60)
def _cached_actions() -> pd.DataFrame:
    return me.fetch_corporate_actions()


@st.cache_data(ttl=60)
def _cached_results(index: str, period: str) -> pd.DataFrame:
    return me.fetch_financial_results(index, period)


def _latest_ts(df: pd.DataFrame, cols: list[str]) -> Optional[pd.Timestamp]:
    if df is None or df.empty:
        return None
    for col in cols:
        if col in df.columns:
            ser = pd.to_datetime(df[col], errors="coerce")
            if ser.notna().any():
                return ser.max()
    return None


def _download_button(label: str, df: pd.DataFrame, key: str):
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=f"{key}.csv", mime="text/csv", key=key)


def main():
    st.title("NSE Corporate / Event Intelligence")
    st.caption("For education only. Not investment advice.")

    with st.sidebar:
        st.header("Controls")
        if st.button("Refresh All", use_container_width=True):
            _cached_announcements.clear()
            _cached_board_meetings.clear()
            _cached_actions.clear()
            _cached_results.clear()
            st.experimental_rerun()

        st.markdown("---")
        idx = st.selectbox("Results Index", ["insurance", "banking", "nbfc", "it", "auto", "all"], index=0)
        per = st.selectbox("Results Period", ["Quarterly", "Annual"], index=0)

    tabs = st.tabs([
        "Announcements",
        "Board Meetings",
        "Corporate Actions",
        "Financial Results",
        "Ranked Watchlist",
    ])

    # Announcements
    with tabs[0]:
        try:
            df = _cached_announcements()
        except Exception as exc:
            st.warning(f"Announcements unavailable: {exc}. Retry / Check after a minute.")
            df = pd.DataFrame()
        st.subheader("Corporate Announcements (Equities)")
        if df.empty:
            st.warning("No announcements fetched. NSE may be rate limiting. Retry later.")
        else:
            q = st.text_input("Search symbol/company/desc", "")
            view = df.copy()
            if q:
                ql = q.lower()
                cols = [c for c in ["symbol", "sm_name", "desc"] if c in view.columns]
                mask = pd.Series(False, index=view.index)
                for c in cols:
                    mask |= view[c].astype(str).str.lower().str.contains(ql, na=False)
                view = view[mask]
            view = view.sort_values(by=["an_dt", "sort_date"], ascending=False, na_position="last")
            st.caption(f"Latest timestamp: {_latest_ts(view, ['an_dt','sort_date'])}")
            st.dataframe(view, use_container_width=True, height=480)
            _download_button("Export CSV", view, key="announcements")

    # Board meetings
    with tabs[1]:
        try:
            df = _cached_board_meetings()
        except Exception as exc:
            st.warning(f"Board meetings unavailable: {exc}. Retry / Check after a minute.")
            df = pd.DataFrame()
        st.subheader("Board Meetings (Equities)")
        if df.empty:
            st.warning("No board meetings fetched. Retry later.")
        else:
            dfv = df.copy()
            now = pd.Timestamp.utcnow().normalize()
            if "bm_date" in dfv.columns:
                dfv["bm_date_only"] = pd.to_datetime(dfv["bm_date"], errors="coerce").dt.date
                dfv["in_next_7d"] = dfv["bm_date_only"].apply(lambda d: (pd.Timestamp(d) - now) <= pd.Timedelta(days=7) if pd.notna(d) else False)
            st.caption(f"Latest timestamp: {_latest_ts(dfv, ['bm_timestamp','bm_date'])}")
            st.dataframe(dfv.sort_values(by=["bm_date","bm_timestamp"], ascending=True), use_container_width=True, height=480)
            _download_button("Export CSV", dfv, key="board_meetings")

    # Corporate actions
    with tabs[2]:
        try:
            df = _cached_actions()
        except Exception as exc:
            st.warning(f"Corporate actions unavailable: {exc}. Retry / Check after a minute.")
            df = pd.DataFrame()
        st.subheader("Corporate Actions (Equities)")
        if df.empty:
            st.warning("No corporate actions fetched. Retry later.")
        else:
            dfv = df.copy()
            if "ex_date" in dfv.columns:
                dfv["is_upcoming_ex"] = pd.to_datetime(dfv["ex_date"], errors="coerce") >= pd.Timestamp.utcnow().normalize()
            st.caption(f"Latest timestamp: {_latest_ts(dfv, ['ex_date','rec_date'])}")
            st.dataframe(dfv.sort_values(by=["ex_date"], ascending=True), use_container_width=True, height=480)
            _download_button("Export CSV", dfv, key="corp_actions")

    # Financial results
    with tabs[3]:
        try:
            df = _cached_results(idx, per)
        except Exception as exc:
            st.warning(f"Financial results unavailable: {exc}. Retry / Check after a minute.")
            df = pd.DataFrame()
        st.subheader("Financial Results")
        if df.empty:
            st.warning("No financial results fetched.")
        else:
            dfv = df.copy()
            st.caption(f"Latest timestamp: {_latest_ts(dfv, ['broad_cast_date','exchdisstime'])}")
            # Link to iXBRL where available
            if "ixbrl" in dfv.columns:
                dfv["ixbrl_link"] = dfv["ixbrl"]
            st.dataframe(dfv, use_container_width=True, height=480)
            _download_button("Export CSV", dfv, key="fin_results")

    # Ranked watchlist
    with tabs[4]:
        try:
            ann, brd, act = _cached_announcements(), _cached_board_meetings(), _cached_actions()
            res = _cached_results(idx, per)
            ranked = me.score_events(ann, brd, act, res)
        except Exception as exc:
            st.warning(f"Scoring error: {exc}")
            ranked = pd.DataFrame()
        st.subheader("Ranked Watchlist")
        if ranked.empty:
            st.warning("No ranked symbols available yet.")
        else:
            # Simple color bands: add band column
            rview = ranked.copy()
            rview["band"] = pd.cut(rview["score"], bins=[-1, 0.8, 1.0, 1.2, 10], labels=["low", "med", "high", "very_high"])
            st.caption(f"Generated at: {pd.Timestamp.utcnow()} (UTC)")
            st.dataframe(rview, use_container_width=True, height=480)
            _download_button("Export CSV", rview, key="ranked_watchlist")

    st.info("For education only. Not investment advice.")


if __name__ == "__main__":
    main()

