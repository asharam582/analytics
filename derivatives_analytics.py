"""Derivatives and Options Analytics for NSE.

Provides hardened NSE session, fetchers for option-chain and active derivatives
lists, and analytics like PCR, v-PCR, OI walls, supports/resistances, and Max
Pain. Includes educational strategy generators with conservative defaults.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
import requests
from requests import Session
from tenacity import (RetryError, retry, retry_if_exception_type,
                      stop_after_attempt, wait_random_exponential)

# Logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")


# Hardened session constants
NSE_BASE = "https://www.nseindia.com/"
HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": NSE_BASE,
    "Connection": "keep-alive",
    "Accept-Language": "en-US,en;q=0.9",
}
# Allow disabling SSL verification for testing via env; default False as requested
VERIFY_SSL: bool = os.getenv("NSE_VERIFY_SSL", "false").lower() not in {"0", "false", "no"}
CHAIN_REFERER = "https://www.nseindia.com/option-chain"


def _rate_limit_sleep() -> None:
    time.sleep(random.uniform(1.0, 2.5))


def get_nse_session(referer: Optional[str] = None) -> Session:
    """Create and return a cookie-initialized, header-hardened NSE Session.

    Args:
        referer: Optional referer to use for warm-up (e.g., option-chain page).
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    warm_url = referer or NSE_BASE
    if referer:
        session.headers["Referer"] = referer
    try:
        _rate_limit_sleep()
        resp = session.get(warm_url, timeout=(10, 10), verify=VERIFY_SSL)
        resp.raise_for_status()
        logger.info("NSE warm-up successful; cookies: %d", len(resp.cookies))
    except Exception as exc:
        logger.warning("NSE warm-up failed: %s", exc)
    return session


class _BlockedError(RuntimeError):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=15),
    retry=retry_if_exception_type((requests.RequestException, _BlockedError)),
)
def _get_json(session: Session, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _rate_limit_sleep()
    resp = session.get(url, params=params, timeout=(10, 10), verify=VERIFY_SSL)
    if resp.status_code in (401, 403):
        raise _BlockedError(f"NSE blocked the request (HTTP {resp.status_code}). Try again later.")
    resp.raise_for_status()
    try:
        return resp.json()
    except json.JSONDecodeError:
        text = (resp.text or "")[:200]
        raise RuntimeError(f"Invalid JSON from NSE at {url}. Snippet: {text}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def snake(s: str) -> str:
        out = []
        for ch in s:
            if ch.isupper():
                out.append("_" + ch.lower())
            elif ch in ("-", " "):
                out.append("_")
            else:
                out.append(ch)
        r = "".join(out).strip("_")
        while "__" in r:
            r = r.replace("__", "_")
        return r

    df = df.copy()
    df.columns = [snake(str(c)) for c in df.columns]
    return df


# ----------------------- Fetchers -----------------------


@lru_cache(maxsize=32)
def fetch_option_chain(type_: str, symbol: str, expiry: str) -> pd.DataFrame:
    """Fetch and flatten the option chain for a specific type/symbol/expiry.

    Args:
        type_: "Indices" or "Stocks".
        symbol: Underlying symbol, e.g. "NIFTY" or a stock ticker.
        expiry: Expiry date string exactly as in NSE payload (e.g., "04-Nov-2025").

    Returns:
        DataFrame with per-strike CE/PE fields and metadata columns.
    """
    url = "https://www.nseindia.com/api/option-chain-v3"
    params = {"type": type_, "symbol": symbol, "expiry": expiry}
    # Use option-chain referer warm-up and header profile
    session = get_nse_session(referer=CHAIN_REFERER)
    session.headers["Referer"] = CHAIN_REFERER
    try:
        payload = _get_json(session, url, params)
    except Exception as exc:
        logger.warning("option-chain-v3 failed (%s). Trying fallback endpoint...", exc)
        # Fallback endpoints (no explicit expiry)
        if type_.lower().startswith("indice"):
            fb_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            fb_url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        try:
            payload = _get_json(session, fb_url)
        except Exception as exc2:
            logger.error("Option chain fetch error (fallback failed): %s", exc2)
            return pd.DataFrame()

    records = payload.get("records") or {}
    underlying_value = records.get("underlyingValue")
    timestamp = records.get("timestamp")
    data = records.get("data") or []
    if not data:
        logger.warning("Option chain data missing or empty for %s %s %s", type_, symbol, expiry)
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for item in data:
        strike = item.get("strikePrice") or item.get("strikeprice") or item.get("strike")
        ce = item.get("CE") or {}
        pe = item.get("PE") or {}
        row = {
            "symbol": symbol,
            "type": type_,
            "expiry": item.get("expiryDates") or item.get("expiryDate") or expiry,
            "strike": float(strike) if strike is not None else np.nan,
            "underlying_value": float(ce.get("underlyingValue", underlying_value) or underlying_value or np.nan),
            "timestamp": timestamp,
            # CE fields
            "ce_oi": ce.get("openInterest"),
            "ce_chg_oi": ce.get("changeinOpenInterest"),
            "ce_pchg_oi": ce.get("pchangeinOpenInterest"),
            "ce_volume": ce.get("totalTradedVolume"),
            "ce_iv": ce.get("impliedVolatility"),
            "ce_ltp": ce.get("lastPrice"),
            "ce_change": ce.get("change"),
            # PE fields
            "pe_oi": pe.get("openInterest"),
            "pe_chg_oi": pe.get("changeinOpenInterest"),
            "pe_pchg_oi": pe.get("pchangeinOpenInterest"),
            "pe_volume": pe.get("totalTradedVolume"),
            "pe_iv": pe.get("impliedVolatility"),
            "pe_ltp": pe.get("lastPrice"),
            "pe_change": pe.get("change"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = _normalize_columns(df)
    # Dtypes
    float_cols = [
        "strike",
        "underlying_value",
        "ce_oi",
        "ce_chg_oi",
        "ce_pchg_oi",
        "ce_volume",
        "ce_iv",
        "ce_ltp",
        "ce_change",
        "pe_oi",
        "pe_chg_oi",
        "pe_pchg_oi",
        "pe_volume",
        "pe_iv",
        "pe_ltp",
        "pe_change",
    ]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    logger.info("Fetched option chain: %d strikes for %s %s %s", len(df), type_, symbol, expiry)
    return df.sort_values("strike").reset_index(drop=True)


def _extract_root_data(payload: Dict[str, Any], root_key: str) -> List[Dict[str, Any]]:
    root = payload.get(root_key) or {}
    return root.get("data") or []


@lru_cache(maxsize=32)
def fetch_most_active_calls_stocks() -> pd.DataFrame:
    """Most active stock CALLS by volume.

    Returns a DataFrame from root "OPTSTK": { "data": [...] }.
    """
    url = "https://www.nseindia.com/api/snapshot-derivatives-equity?index=calls-stocks-vol"
    session = get_nse_session()
    try:
        payload = _get_json(session, url)
    except Exception as exc:
        logger.error("Fetch most active calls stocks error: %s", exc)
        return pd.DataFrame()
    data = _extract_root_data(payload, "OPTSTK")
    df = _normalize_columns(pd.DataFrame(data)) if data else pd.DataFrame()
    logger.info("Fetched most active stock CALLS: %d", len(df))
    return df


@lru_cache(maxsize=32)
def fetch_most_active_puts_stocks() -> pd.DataFrame:
    """Most active stock PUTS by volume."""
    url = "https://www.nseindia.com/api/snapshot-derivatives-equity?index=puts-stocks-vol"
    session = get_nse_session()
    try:
        payload = _get_json(session, url)
    except Exception as exc:
        logger.error("Fetch most active puts stocks error: %s", exc)
        return pd.DataFrame()
    data = _extract_root_data(payload, "OPTSTK")
    df = _normalize_columns(pd.DataFrame(data)) if data else pd.DataFrame()
    logger.info("Fetched most active stock PUTS: %d", len(df))
    return df


@lru_cache(maxsize=32)
def fetch_most_active_oi() -> Dict[str, pd.DataFrame]:
    """Fetch most active by OI/volume/value basket.

    Returns a dict of DataFrames for keys present, e.g. {"volume": df, "value": df2}.
    """
    url = "https://www.nseindia.com/api/snapshot-derivatives-equity?index=oi"
    session = get_nse_session()
    try:
        payload = _get_json(session, url)
    except Exception as exc:
        logger.error("Fetch most active OI basket error: %s", exc)
        return {}
    out: Dict[str, pd.DataFrame] = {}
    for key in ("volume", "value", "oi"):
        if key in payload:
            data = payload[key].get("data") or []
            out[key] = _normalize_columns(pd.DataFrame(data)) if data else pd.DataFrame()
    for k, v in out.items():
        logger.info("Fetched basket '%s' rows: %d", k, len(v))
    return out


@lru_cache(maxsize=32)
def fetch_most_active_contracts(limit: int = 20) -> pd.DataFrame:
    """Fetch most active contracts by volume.

    Args:
        limit: Number of contracts to request from the API (hint only).
    """
    url = f"https://www.nseindia.com/api/snapshot-derivatives-equity?index=contracts&limit={int(limit)}"
    session = get_nse_session()
    try:
        payload = _get_json(session, url)
    except Exception as exc:
        logger.error("Fetch most active contracts error: %s", exc)
        return pd.DataFrame()
    data = payload.get("volume", {}).get("data") or []
    df = _normalize_columns(pd.DataFrame(data)) if data else pd.DataFrame()
    logger.info("Fetched most active contracts: %d", len(df))
    return df


# ----------------------- Analytics -----------------------


def enrich_chain_with_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-strike totals and distances; returns a new DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in ("ce_oi", "pe_oi", "ce_volume", "pe_volume"):
        if c not in out.columns:
            out[c] = np.nan
    out["total_oi"] = (out["ce_oi"].fillna(0) + out["pe_oi"].fillna(0)).astype(float)
    out["total_volume"] = (out["ce_volume"].fillna(0) + out["pe_volume"].fillna(0)).astype(float)
    if "underlying_value" in out.columns and "strike" in out.columns:
        out["dist_from_atm"] = (out["strike"].astype(float) - out["underlying_value"].astype(float)).abs()
    return out


def compute_pcr(df: pd.DataFrame) -> Tuple[float, float]:
    """Compute PCR and v-PCR from a chain DataFrame.

    Returns:
        (pcr, v_pcr) where pcr = sum(PE OI) / sum(CE OI), v_pcr = sum(PE volume) / sum(CE volume)
    """
    if df is None or df.empty:
        return (float("nan"), float("nan"))
    ce_oi = float(df["ce_oi"].fillna(0).sum()) or 0.0
    pe_oi = float(df["pe_oi"].fillna(0).sum()) or 0.0
    ce_v = float(df["ce_volume"].fillna(0).sum()) or 0.0
    pe_v = float(df["pe_volume"].fillna(0).sum()) or 0.0
    pcr = pe_oi / ce_oi if ce_oi > 0 else float("inf")
    v_pcr = pe_v / ce_v if ce_v > 0 else float("inf")
    logger.info("PCR=%.3f vPCR=%.3f", pcr, v_pcr)
    return pcr, v_pcr


def oi_walls(df: pd.DataFrame, top_n: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return top call and put OI walls as two DataFrames (top_n each)."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    call_walls = df[["strike", "ce_oi"]].dropna().sort_values("ce_oi", ascending=False).head(top_n)
    put_walls = df[["strike", "pe_oi"]].dropna().sort_values("pe_oi", ascending=False).head(top_n)
    logger.info("Top call walls: %s | Top put walls: %s", list(call_walls["strike"]), list(put_walls["strike"]))
    return call_walls.reset_index(drop=True), put_walls.reset_index(drop=True)


def supports_resistances(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute nearest support and resistance from put/call walls.

    Returns a dict with keys: support, resistance, dist_to_support, dist_to_resistance
    """
    if df is None or df.empty:
        return {"support": np.nan, "resistance": np.nan, "dist_to_support": np.nan, "dist_to_resistance": np.nan}
    uv = float(df["underlying_value"].dropna().iloc[0]) if df["underlying_value"].notna().any() else np.nan
    cw, pw = oi_walls(df, top_n=10)
    support = np.nan
    resistance = np.nan
    if not pw.empty:
        # nearest put wall below/nearest overall
        pw["dist"] = (pw["strike"] - uv).abs()
        support = pw.sort_values("dist").iloc[0]["strike"]
    if not cw.empty:
        cw["dist"] = (cw["strike"] - uv).abs()
        resistance = cw.sort_values("dist").iloc[0]["strike"]
    dist_s = float(abs(uv - support)) if not math.isnan(uv) and not math.isnan(support) else np.nan
    dist_r = float(abs(resistance - uv)) if not math.isnan(uv) and not math.isnan(resistance) else np.nan
    out = {
        "support": float(support) if not math.isnan(support) else np.nan,
        "resistance": float(resistance) if not math.isnan(resistance) else np.nan,
        "dist_to_support": dist_s,
        "dist_to_resistance": dist_r,
    }
    logger.info("S/R computed: %s", out)
    return out


def max_pain(df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """Compute Max Pain strike and the payout table.

    The total payout at candidate strike K is approximated as:
        sum_over_strikes( call_OI * max(0, K - strike) + put_OI * max(0, strike - K) )

    Returns:
        (max_pain_strike, payout_table) where payout_table has columns [strike, total_payout].
    """
    if df is None or df.empty:
        return float("nan"), pd.DataFrame()
    strikes = df["strike"].dropna().astype(float).values
    ce_oi = df.set_index("strike")["ce_oi"].fillna(0).astype(float)
    pe_oi = df.set_index("strike")["pe_oi"].fillna(0).astype(float)
    payouts: List[Tuple[float, float]] = []
    for K in strikes:
        call_payout = ((K - strikes).clip(min=0) * ce_oi.reindex(strikes, fill_value=0).values).sum()
        put_payout = ((strikes - K).clip(min=0) * pe_oi.reindex(strikes, fill_value=0).values).sum()
        total = float(call_payout + put_payout)
        payouts.append((float(K), total))
    table = pd.DataFrame(payouts, columns=["strike", "total_payout"]).sort_values("strike")
    mp_row = table.loc[table["total_payout"].idxmin()]
    mp_strike = float(mp_row["strike"]) if not table.empty else float("nan")
    logger.info("Max Pain strike: %s", mp_strike)
    return mp_strike, table


def estimate_greeks(*_: Any, **__: Any) -> pd.DataFrame:
    """Placeholder Greeks estimator. Returns NaNs.

    TODO: Implement Black-Scholes or IV estimation later.
    """
    return pd.DataFrame({
        "delta": [np.nan],
        "gamma": [np.nan],
        "vega": [np.nan],
        "theta": [np.nan],
        "iv": [np.nan],
    })


# ----------------------- Strategies (Educational) -----------------------


def _nearest_atm_strike(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return float("nan")
    uv = float(df["underlying_value"].dropna().iloc[0]) if df["underlying_value"].notna().any() else np.nan
    s = df.iloc[(df["strike"] - uv).abs().argsort()].iloc[0]["strike"]
    return float(s)


def _infer_decreasing_call_wall(prev_df: Optional[pd.DataFrame], curr_df: Optional[pd.DataFrame]) -> bool:
    """Heuristic: top call wall OI decreased from previous snapshot to current."""
    if prev_df is None or curr_df is None or prev_df.empty or curr_df.empty:
        return False
    prev_top = prev_df.sort_values("ce_oi", ascending=False).head(1)
    curr_top = curr_df.sort_values("ce_oi", ascending=False).head(1)
    if prev_top.empty or curr_top.empty:
        return False
    prev_oi = float(prev_top["ce_oi"].iloc[0] or 0)
    curr_oi = float(curr_top["ce_oi"].iloc[0] or 0)
    return curr_oi < prev_oi


def generate_index_intraday_spreads(
    df_chain: pd.DataFrame,
    notional: float = 100000.0,
    snapshots: Optional[List[pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Suggest CE debit spreads for index options (educational).

    Rules:
      - Use underlying value as proxy price.
      - Consider a simple proxy VWAP via nearby strikes' CE LTP average.
      - If price > proxy and top call-wall OI is decreasing (if snapshots provided),
        propose a bullish CE debit spread around ATM/OTM.
    """
    if df_chain is None or df_chain.empty:
        return pd.DataFrame()
    df = enrich_chain_with_totals(df_chain)
    atm = _nearest_atm_strike(df)
    uv = float(df["underlying_value"].dropna().iloc[0]) if df["underlying_value"].notna().any() else np.nan
    # proxy vwap: mean CE LTP for +/- 3 strikes around ATM
    df_sorted = df.iloc[(df["strike"] - atm).abs().argsort()].head(7)
    proxy_vwap = float(df_sorted["ce_ltp"].replace(0, np.nan).dropna().mean()) if df_sorted["ce_ltp"].notna().any() else uv

    falling_call_wall = False
    if snapshots and len(snapshots) >= 2:
        falling_call_wall = _infer_decreasing_call_wall(snapshots[-2], snapshots[-1])

    if uv > proxy_vwap and falling_call_wall:
        # Long CE at ATM/OTM, short CE further OTM by fixed width
        width = max(50.0, round(df["strike"].diff().abs().median() or 50, -1))
        long_strike = float(df_sorted.iloc[0]["strike"])  # ATM
        short_strike = long_strike + width
        long_ltp = float(df[df["strike"] == long_strike]["ce_ltp"].fillna(0).iloc[0]) if (df["strike"] == long_strike).any() else np.nan
        short_ltp = float(df[df["strike"] == short_strike]["ce_ltp"].fillna(0).iloc[0]) if (df["strike"] == short_strike).any() else np.nan
        debit = max(0.0, (long_ltp - short_ltp)) if all(np.isfinite([long_ltp, short_ltp])) else np.nan
        lot_size = 50  # indicative default for indices; varies by symbol
        qty = max(1, int(notional / max(debit * lot_size, 1))) if np.isfinite(debit) and debit > 0 else 1
        max_loss = debit * lot_size * qty if np.isfinite(debit) else np.nan
        max_gain = (short_strike - long_strike - debit) * lot_size * qty if np.isfinite(debit) else np.nan
        rows = [
            {"action": "BUY", "type": "CE", "strike": long_strike, "ltp": long_ltp, "qty": qty},
            {"action": "SELL", "type": "CE", "strike": short_strike, "ltp": short_ltp, "qty": qty},
        ]
        out = pd.DataFrame(rows)
        out.attrs["debit"] = debit
        out.attrs["max_loss"] = max_loss
        out.attrs["max_gain"] = max_gain
        return out

    # No signal
    return pd.DataFrame()


def generate_event_driven_stock_spreads(
    event_symbols: Optional[List[str]],
    universe_df: Optional[pd.DataFrame] = None,
    notional: float = 200000.0,
) -> pd.DataFrame:
    """Propose vertical spreads for stocks present in both event list and active options.

    Args:
        event_symbols: List of stock symbols from event scoring.
        universe_df: A DataFrame of active stock options (e.g., merged calls/puts activity) to validate presence.
        notional: Target notional sizing.

    Returns:
        DataFrame of proposed structures (symbol, long_strike, short_strike, qty, debit, width, max_gain, max_loss).
    """
    if not event_symbols:
        return pd.DataFrame()
    if universe_df is None or universe_df.empty:
        return pd.DataFrame()
    active_syms = set(str(s).upper() for s in universe_df.get("underlying", pd.Series([], dtype=str)).astype(str))
    targets = [s for s in set(e.upper() for e in event_symbols) if s in active_syms]
    proposals: List[Dict[str, Any]] = []

    # For a small subset to avoid hammering
    for sym in targets[:5]:
        # We need an expiry to fetch chain; try the most common in activity
        sym_rows = universe_df[universe_df["underlying"].astype(str).str.upper() == sym]
        if sym_rows.empty:
            continue
        expiry = str(sym_rows["expiry_date"].iloc[0]) if "expiry_date" in sym_rows.columns else None
        if not expiry or expiry.lower() in ("nan", "none"):
            # Skip if we cannot deduce expiry from activity snapshot
            continue
        df_chain = fetch_option_chain("Stocks", sym, expiry)
        if df_chain.empty:
            continue
        df = enrich_chain_with_totals(df_chain)
        atm = _nearest_atm_strike(df)
        width = max(5.0, round(df["strike"].diff().abs().median() or 5))
        long_strike = atm
        short_strike = atm + width
        long_ltp = float(df[df["strike"] == long_strike]["ce_ltp"].fillna(0).iloc[0]) if (df["strike"] == long_strike).any() else np.nan
        short_ltp = float(df[df["strike"] == short_strike]["ce_ltp"].fillna(0).iloc[0]) if (df["strike"] == short_strike).any() else np.nan
        debit = max(0.0, (long_ltp - short_ltp)) if all(np.isfinite([long_ltp, short_ltp])) else np.nan
        lot_size = 1  # stock lots vary widely; treat as unit for notional estimation
        qty = max(1, int(notional / max(debit * lot_size, 1))) if np.isfinite(debit) and debit > 0 else 1
        width_abs = (short_strike - long_strike)
        max_loss = debit * lot_size * qty if np.isfinite(debit) else np.nan
        max_gain = (width_abs - debit) * lot_size * qty if np.isfinite(debit) else np.nan
        proposals.append(
            {
                "symbol": sym,
                "expiry": expiry,
                "long_strike": float(long_strike),
                "short_strike": float(short_strike),
                "qty": int(qty),
                "debit": float(debit) if np.isfinite(debit) else np.nan,
                "width": float(width_abs),
                "max_gain": float(max_gain) if np.isfinite(max_gain) else np.nan,
                "max_loss": float(max_loss) if np.isfinite(max_loss) else np.nan,
            }
        )

    out_df = pd.DataFrame(proposals)
    logger.info("Generated event-driven spreads: %d", len(out_df))
    return out_df


if __name__ == "__main__":
    # Demo: fetch one chain and compute analytics
    print("Fetching sample chain for NIFTY (demo; adjust expiry as available)...")
    try:
        demo_df = fetch_option_chain("Indices", "NIFTY", "04-Nov-2025")
    except Exception as exc:
        logger.error("Demo chain fetch failed: %s", exc)
        demo_df = pd.DataFrame()
    if demo_df.empty:
        print("Option chain empty (network or date mismatch).")
    else:
        enr = enrich_chain_with_totals(demo_df)
        pcr, v_pcr = compute_pcr(enr)
        mp, tbl = max_pain(enr)
        sr = supports_resistances(enr)
        print("PCR:", pcr, "vPCR:", v_pcr)
        print("Max Pain:", mp)
        print("S/R:", sr)
        print(enr.head())
