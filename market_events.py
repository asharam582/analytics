"""Market Events Fetcher and Scoring Utilities for NSE.

This module provides a hardened, cookie-aware NSE session and utilities to
fetch and normalize corporate feeds: announcements, board meetings, corporate
actions, and financial results. It also includes a simple scoring mechanism to
rank symbols by recent, potentially material events.

Usage:
    from market_events import (
        get_nse_session,
        fetch_corporate_announcements,
        fetch_board_meetings,
        fetch_corporate_actions,
        fetch_financial_results,
        score_events,
    )

All fetchers return pandas.DataFrame with normalized snake_case columns and
parsed datetimes where applicable.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests import Session
from tenacity import (RetryError, retry, retry_if_exception_type,
                      stop_after_attempt, wait_random_exponential)

try:
    from pydantic import BaseModel, Field
except ImportError:  # Fallback if pydantic not installed; minimal shims
    class BaseModel:  # type: ignore
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

        def model_dump(self) -> Dict[str, Any]:  # type: ignore
            return self.__dict__

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore
        return default


# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")


# Constants
NSE_BASE = "https://www.nseindia.com/"
HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": NSE_BASE,
    "Connection": "keep-alive",
}


def _snake_case(name: str) -> str:
    out = []
    for ch in name:
        if ch.isupper():
            out.append("_" + ch.lower())
        elif ch in ("-", " "):
            out.append("_")
        else:
            out.append(ch)
    s = "".join(out)
    s = s.strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_snake_case(str(c)) for c in df.columns]
    return df


def _parse_dt(val: Any, dayfirst: bool = True) -> Optional[pd.Timestamp]:
    if val in (None, "", "-"):
        return None
    # Try common patterns
    for fmt in (
        "%d-%b-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d-%b-%Y",
        "%Y-%m-%d",
        "%d%m%Y%H%M%S",
        "%d-%m-%Y",
    ):
        try:
            return pd.to_datetime(val, format=fmt, dayfirst=dayfirst, errors="raise")
        except Exception:
            continue
    # Fallback to pandas parser
    try:
        return pd.to_datetime(val, dayfirst=dayfirst, errors="coerce")
    except Exception:
        return None


def _rate_limit_sleep() -> None:
    time.sleep(random.uniform(1.0, 2.5))


def get_nse_session() -> Session:
    """Create and return a cookie-initialized, header-hardened NSE Session.

    Performs a warm-up request to establish cookies.
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        _rate_limit_sleep()
        resp = session.get(NSE_BASE, timeout=(10, 10),verify=False)
        resp.raise_for_status()
        logger.info("NSE warm-up successful; cookies obtained: %d", len(resp.cookies))
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
    resp = session.get(url, params=params, timeout=(10, 10))
    if resp.status_code == 403:
        raise _BlockedError("NSE blocked the request (HTTP 403). Try again later.")
    resp.raise_for_status()
    # Some NSE endpoints mislabel content-type; always attempt json
    try:
        return resp.json()
    except json.JSONDecodeError:
        text = (resp.text or "")[:200]
        raise RuntimeError(f"Invalid JSON from NSE at {url}. Snippet: {text}")


# Pydantic models (subset of fields for validation)
class AnnouncementItem(BaseModel):
    symbol: Optional[str] = Field(default=None)
    desc: Optional[str] = Field(default=None)
    an_dt: Optional[str] = Field(default=None)
    sm_name: Optional[str] = Field(default=None)
    attchmntFile: Optional[str] = Field(default=None)
    hasXbrl: Optional[bool] = Field(default=None)


class BoardMeetingItem(BaseModel):
    bm_symbol: Optional[str] = Field(default=None)
    bm_date: Optional[str] = Field(default=None)
    bm_purpose: Optional[str] = Field(default=None)
    sm_name: Optional[str] = Field(default=None)
    bm_timestamp: Optional[str] = Field(default=None)
    attachment: Optional[str] = Field(default=None)


class CorporateActionItem(BaseModel):
    symbol: Optional[str] = Field(default=None)
    series: Optional[str] = Field(default=None)
    faceVal: Optional[str] = Field(default=None)
    subject: Optional[str] = Field(default=None)
    exDate: Optional[str] = Field(default=None)
    recDate: Optional[str] = Field(default=None)
    comp: Optional[str] = Field(default=None)


class FinancialResultItem(BaseModel):
    symbol: Optional[str] = Field(default=None)
    companyName: Optional[str] = Field(default=None)
    audited: Optional[str] = Field(default=None)
    cumulative: Optional[str] = Field(default=None)
    period: Optional[str] = Field(default=None)
    periodEnd: Optional[str] = Field(default=None)
    broadCastDate: Optional[str] = Field(default=None)
    ixbrl: Optional[str] = Field(default=None)


def _to_dataframe(items: Iterable[BaseModel]) -> pd.DataFrame:
    rows = [getattr(it, "model_dump", lambda: it.__dict__)() for it in items]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = _normalize_columns(df)
    return df


@lru_cache(maxsize=32)
def fetch_corporate_announcements() -> pd.DataFrame:
    """Fetch and normalize corporate announcements for equities.

    Returns:
        DataFrame with columns like: symbol, desc, an_dt (datetime), sm_name, has_xbrl, attchmnt_file
    """
    url = "https://www.nseindia.com/api/corporate-announcements?index=equities"
    session = get_nse_session()
    try:
        payload = _get_json(session, url)
    except RetryError as rexc:
        logger.error("Failed to fetch announcements after retries: %s", rexc)
        return pd.DataFrame()
    except Exception as exc:
        logger.error("Announcements fetch error: %s", exc)
        return pd.DataFrame()

    data = payload if isinstance(payload, list) else payload.get("data") or payload.get("items") or []
    items = [AnnouncementItem(**d) for d in data]
    df = _to_dataframe(items)
    if df.empty:
        logger.warning("Announcements empty or blocked.")
        return df
    # Parse datetimes
    if "an_dt" in df.columns:
        df["an_dt"] = df["an_dt"].apply(lambda v: _parse_dt(v, dayfirst=True))
    if "sort_date" in df.columns:
        df["sort_date"] = df["sort_date"].apply(lambda v: _parse_dt(v, dayfirst=True))
    logger.info("Fetched announcements: %d rows", len(df))
    return df


@lru_cache(maxsize=32)
def fetch_board_meetings() -> pd.DataFrame:
    """Fetch and normalize corporate board meetings for equities."""
    url = "https://www.nseindia.com/api/corporate-board-meetings?index=equities"
    session = get_nse_session()
    try:
        payload = _get_json(session, url)
    except RetryError as rexc:
        logger.error("Failed to fetch board meetings after retries: %s", rexc)
        return pd.DataFrame()
    except Exception as exc:
        logger.error("Board meetings fetch error: %s", exc)
        return pd.DataFrame()

    data = payload if isinstance(payload, list) else payload.get("data") or []
    items = [BoardMeetingItem(**d) for d in data]
    df = _to_dataframe(items)
    if df.empty:
        logger.warning("Board meetings empty or blocked.")
        return df
    for col in ("bm_date", "bm_timestamp"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _parse_dt(v, dayfirst=True))
    logger.info("Fetched board meetings: %d rows", len(df))
    return df


@lru_cache(maxsize=32)
def fetch_corporate_actions() -> pd.DataFrame:
    """Fetch and normalize corporate actions for equities."""
    url = "https://www.nseindia.com/api/corporates-corporateActions?index=equities"
    session = get_nse_session()
    try:
        payload = _get_json(session, url)
    except RetryError as rexc:
        logger.error("Failed to fetch corporate actions after retries: %s", rexc)
        return pd.DataFrame()
    except Exception as exc:
        logger.error("Corporate actions fetch error: %s", exc)
        return pd.DataFrame()

    data = payload if isinstance(payload, list) else payload.get("data") or []
    items = [CorporateActionItem(**d) for d in data]
    df = _to_dataframe(items)
    if df.empty:
        logger.warning("Corporate actions empty or blocked.")
        return df
    for col in ("ex_date", "rec_date"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _parse_dt(v, dayfirst=True))
    logger.info("Fetched corporate actions: %d rows", len(df))
    return df


@lru_cache(maxsize=32)
def fetch_financial_results(index: str = "insurance", period: str = "Quarterly") -> pd.DataFrame:
    """Fetch and normalize financial results for a given index and period.

    Args:
        index: Segment index (e.g., "insurance", "banking").
        period: Period type (e.g., "Quarterly", "Annual").
    """
    url = "https://www.nseindia.com/api/corporates-financial-results"
    params = {"index": index, "period": period}
    session = get_nse_session()
    try:
        payload = _get_json(session, url, params=params)
    except RetryError as rexc:
        logger.error("Failed to fetch financial results after retries: %s", rexc)
        return pd.DataFrame()
    except Exception as exc:
        logger.error("Financial results fetch error: %s", exc)
        return pd.DataFrame()

    data = payload if isinstance(payload, list) else payload.get("data") or []
    items = [FinancialResultItem(**d) for d in data]
    df = _to_dataframe(items)
    if df.empty:
        logger.warning("Financial results empty or blocked.")
        return df
    for col in ("period_end", "broad_cast_date"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _parse_dt(v, dayfirst=True))
    logger.info("Fetched financial results: %d rows", len(df))
    return df


def _recency_weight(ts: Optional[pd.Timestamp]) -> float:
    if ts is None or pd.isna(ts):
        return 0.5
    # Normalize to UTC-aware Timestamp for safe subtraction
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            ts_utc = ts.tz_localize("UTC")
        else:
            ts_utc = ts.tz_convert("UTC")
    else:
        try:
            ts_parsed = pd.to_datetime(ts, errors="coerce")
            ts_utc = ts_parsed.tz_localize("UTC") if ts_parsed.tzinfo is None else ts_parsed.tz_convert("UTC")
        except Exception:
            return 0.5
    age_days = max(0.0, (now - ts_utc).total_seconds() / 86400.0)
    # Exponential decay with ~7-day half-life
    half_life = 7.0
    lam = 0.6931 / half_life
    return float(max(0.1, min(1.5, 1.0 * (2.0 ** (-age_days / half_life)) + lam)))


def _materiality_weight(event_type: str, row: pd.Series) -> float:
    et = (event_type or "").lower()
    if et == "announcement":
        return 1.2 if bool(row.get("has_xbrl", False)) else 1.0
    if et == "board_meeting":
        txt = str(row.get("bm_purpose", ""))
        if "results" in txt.lower() or "dividend" in txt.lower():
            return 1.15
        return 1.05
    if et == "corporate_action":
        subj = str(row.get("subject", "")).lower()
        for key, w in (("bonus", 1.3), ("split", 1.25), ("dividend", 1.2), ("buyback", 1.35)):
            if key in subj:
                return w
        return 1.1
    if et == "financial_result":
        audited = str(row.get("audited", "")).lower()
        consolidated = str(row.get("consolidated", "")).lower()
        w = 1.1
        if "audited" in audited:
            w += 0.1
        if "consolidated" in consolidated:
            w += 0.05
        return w
    return 1.0


def score_events(
    df_annc: pd.DataFrame,
    df_board: pd.DataFrame,
    df_actions: pd.DataFrame,
    df_results: pd.DataFrame,
) -> pd.DataFrame:
    """Score and rank symbols across event feeds.

    The score is a multiplicative blend of recency, materiality and a
    placeholder liquidity weight (fixed 1.0 for now). Duplicates by symbol are
    merged keeping the highest score.

    Returns:
        DataFrame with columns:
        symbol, name, event_type, event_dt, weight_recency, weight_materiality,
        weight_liquidity, score
    """
    rows: List[Dict[str, Any]] = []

    # Announcements
    if df_annc is not None and not df_annc.empty:
        for _, r in df_annc.iterrows():
            ts = r.get("an_dt") or r.get("sort_date")
            ts_parsed = _parse_dt(ts) if not isinstance(ts, (pd.Timestamp, type(None))) else ts
            rec = _recency_weight(ts_parsed)
            mat = _materiality_weight("announcement", r)
            liq = 1.0
            rows.append(
                {
                    "symbol": r.get("symbol"),
                    "name": r.get("sm_name"),
                    "event_type": "announcement",
                    "event_dt": ts_parsed,
                    "weight_recency": rec,
                    "weight_materiality": mat,
                    "weight_liquidity": liq,
                    "score": rec * mat * liq,
                }
            )

    # Board meetings
    if df_board is not None and not df_board.empty:
        for _, r in df_board.iterrows():
            ts = r.get("bm_timestamp") or r.get("bm_date")
            ts_parsed = _parse_dt(ts) if not isinstance(ts, (pd.Timestamp, type(None))) else ts
            rec = _recency_weight(ts_parsed)
            mat = _materiality_weight("board_meeting", r)
            liq = 1.0
            rows.append(
                {
                    "symbol": r.get("bm_symbol"),
                    "name": r.get("sm_name"),
                    "event_type": "board_meeting",
                    "event_dt": ts_parsed,
                    "weight_recency": rec,
                    "weight_materiality": mat,
                    "weight_liquidity": liq,
                    "score": rec * mat * liq,
                }
            )

    # Corporate actions
    if df_actions is not None and not df_actions.empty:
        for _, r in df_actions.iterrows():
            ts = r.get("ex_date")
            ts_parsed = _parse_dt(ts) if not isinstance(ts, (pd.Timestamp, type(None))) else ts
            rec = _recency_weight(ts_parsed)
            mat = _materiality_weight("corporate_action", r)
            liq = 1.0
            rows.append(
                {
                    "symbol": r.get("symbol"),
                    "name": r.get("comp"),
                    "event_type": "corporate_action",
                    "event_dt": ts_parsed,
                    "weight_recency": rec,
                    "weight_materiality": mat,
                    "weight_liquidity": liq,
                    "score": rec * mat * liq,
                }
            )

    # Financial results
    if df_results is not None and not df_results.empty:
        for _, r in df_results.iterrows():
            ts = r.get("broad_cast_date") or r.get("exchdisstime")
            ts_parsed = _parse_dt(ts) if not isinstance(ts, (pd.Timestamp, type(None))) else ts
            rec = _recency_weight(ts_parsed)
            mat = _materiality_weight("financial_result", r)
            liq = 1.0
            rows.append(
                {
                    "symbol": r.get("symbol"),
                    "name": r.get("company_name"),
                    "event_type": "financial_result",
                    "event_dt": ts_parsed,
                    "weight_recency": rec,
                    "weight_materiality": mat,
                    "weight_liquidity": liq,
                    "score": rec * mat * liq,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("Event scoring: no input rows. Returning empty DataFrame.")
        return df

    # Deduplicate by symbol keeping highest score
    df = df.sort_values(["symbol", "score"], ascending=[True, False])
    df = df.dropna(subset=["symbol"])  # ensure symbol available
    df = df.groupby("symbol", as_index=False).first()
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    logger.info("Scored events for %d symbols", len(df))
    return df


if __name__ == "__main__":
    # Quick demo
    print("Fetching NSE market events (one pass demo)...")
    ann = fetch_corporate_announcements()
    brd = fetch_board_meetings()
    act = fetch_corporate_actions()
    res = fetch_financial_results("insurance", "Quarterly")
    print("Announcements:", ann.head() if not ann.empty else "<empty>")
    print("Board Meetings:", brd.head() if not brd.empty else "<empty>")
    print("Actions:", act.head() if not act.empty else "<empty>")
    print("Results:", res.head() if not res.empty else "<empty>")
    ranked = score_events(ann, brd, act, res)
    print("Ranked watchlist:", ranked.head() if not ranked.empty else "<empty>")
