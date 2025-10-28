# NSE Options & Events Analytics (4-file project)

Production‑grade Python project with two core modules and two Streamlit apps:

- `market_events.py` — Corporate/events fetchers + ranked watchlist scoring
- `derivatives_analytics.py` — Options/derivatives fetchers + analytics & strategies
- `app_events.py` — Streamlit UI for corporate/event intelligence
- `app_derivatives.py` — Streamlit UI for live options analytics and suggestions (educational)

Both modules include a hardened, cookie‑aware NSE session with warm‑up, structured logging, retries, small rate‑limiting, and safe JSON parsing. Streamlit apps are responsive and cache data.

> For education only. Not investment advice.

---

## Features

- Shared NSE session helper: `get_nse_session()` with warm‑up against NSE, session headers, optional SSL verify toggle
- Corporate feeds: announcements, board meetings, corporate actions, financial results
- Event scoring: blends recency/materiality/liquidity (placeholder) and consolidates per symbol
- Options flow: option chain (Indices/Stocks), most active calls/puts, OI/volume/value baskets, most active contracts
- Chain analytics: PCR, v‑PCR, OI walls, supports/resistances, Max Pain (table + strike)
- Strategy ideas (educational): index intraday CE debit spread heuristic; event‑driven stock verticals
- Resilient networking: retries (tenacity), jittered backoff, rate‑limiting sleep, cookie warm‑up, referer profile
- Caching: in‑memory LRU in modules; `st.cache_data(ttl=60)` in UIs;
- Logging: INFO for counts/analytics; WARN for empty/blocked; robust exceptions without app crashes

---

## Requirements

- Python 3.9+
- Install dependencies:

```
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Key libraries: pandas, numpy, requests, tenacity, pydantic, streamlit, altair

---

## Configuration

- Headers are pre‑set to mimic a modern browser.
- Small rate limiter: 1.0–2.5s jittered sleep before requests.
- Retries: exponential backoff (tenacity) on network and blocked responses.
- SSL verify toggle (testing only):
  - Env var `NSE_VERIFY_SSL` controls TLS verification.
  - Default in `derivatives_analytics.py` is disabled for testing convenience (to reduce 401s if your environment MITMs SSL). Enable in production.

Examples (PowerShell):

```
# Disable (testing)
$env:NSE_VERIFY_SSL = "false"

# Enable (recommended for production)
$env:NSE_VERIFY_SSL = "true"
```

- Option‑chain referer: `derivatives_analytics.py` warms up using `https://www.nseindia.com/option-chain` to obtain cookies that reduce 401/403 for chain endpoints. Falls back to legacy endpoints when v3 fails.

---

## Running

Streamlit apps:

```
streamlit run app_events.py
streamlit run app_derivatives.py
```

Module demos (one‑off fetch + print):

```
python market_events.py
python derivatives_analytics.py
```

Virtual environment (PowerShell example on Windows):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Module APIs

### market_events.py

- `get_nse_session() -> requests.Session`
- `fetch_corporate_announcements() -> pd.DataFrame`
- `fetch_board_meetings() -> pd.DataFrame`
- `fetch_corporate_actions() -> pd.DataFrame`
- `fetch_financial_results(index: str, period: str) -> pd.DataFrame`
- `score_events(df_annc, df_board, df_actions, df_results) -> pd.DataFrame`

Returned columns use snake_case and parse datetimes where present. Typical columns:
- Announcements: `symbol`, `desc`, `an_dt`, `sm_name`, `has_xbrl`, `attchmnt_file`, `sort_date`
- Board meetings: `bm_symbol`, `bm_date`, `bm_purpose`, `bm_timestamp`, `sm_name`
- Corporate actions: `symbol`, `series`, `face_val`, `subject`, `ex_date`, `rec_date`, `comp`
- Financial results: `symbol`, `company_name`, `period`, `period_end`, `broad_cast_date`, `ixbrl`

Scoring output columns:
- `symbol`, `name`, `event_type`, `event_dt`, `weight_recency`, `weight_materiality`, `weight_liquidity`, `score`

### derivatives_analytics.py

- `get_nse_session(referer: str | None = None) -> requests.Session`
- Fetchers (DataFrames):
  - `fetch_option_chain(type_: str, symbol: str, expiry: str)` — flattens CE/PE by strike; resilient warm‑up; fallback endpoints
  - `fetch_most_active_calls_stocks()`
  - `fetch_most_active_puts_stocks()`
  - `fetch_most_active_oi() -> dict[str, pd.DataFrame]` (keys: `volume`, `value` if present)
  - `fetch_most_active_contracts(limit: int = 20)`
- Analytics:
  - `enrich_chain_with_totals(df)` adds `total_oi`, `total_volume`, `dist_from_atm`
  - `compute_pcr(df) -> tuple[float, float]` returns `(pcr, v_pcr)`
  - `oi_walls(df, top_n=3)` top call/put OI walls
  - `supports_resistances(df)` nearest support/resistance vs ATM with distances
  - `max_pain(df) -> tuple[float, pd.DataFrame]` strike and payout table
  - `estimate_greeks(...)` placeholder returns NaNs (TODO)
- Strategies (educational):
  - `generate_index_intraday_spreads(df_chain, notional=..., snapshots=None)` CE debit spread when price > proxy VWAP and top call‑wall OI decreasing
  - `generate_event_driven_stock_spreads(event_symbols, universe_df, notional=...)` proposes verticals for stocks in both event list and active options

---

## Streamlit UIs

### app_events.py

- Sidebar: Refresh all (clears cache), selectors for financial results index/period
- Tabs:
  - Announcements — searchable table, most recent first
  - Board Meetings — next 7 days highlight
  - Corporate Actions — highlight upcoming ex‑dates
  - Financial Results — index/period controls, iXBRL links where available
  - Ranked Watchlist — output of `score_events()` with simple score bands
- Export CSV buttons on each tab

### app_derivatives.py

- Sidebar controls: underlying type (`Indices`/`Stocks`), `symbol`, `expiry`, optional “Use event filter”
- Tabs:
  - Option Chain Analytics — PCR, v‑PCR, Max Pain (value + chart), call/put OI walls, nearest S/R, per‑strike table
  - Most Actives — calls/puts (stocks), OI/volume/value baskets, most active contracts
  - Suggested Structures — index CE debit spread (heuristic) or event‑driven stock verticals (if filter enabled)
- Clear disclaimer banner

---

## Troubleshooting

- 401/403 Unauthorized/Forbidden
  - Wait 30–90 seconds and retry (rate limiting).
  - Ensure warm‑up hits `https://www.nseindia.com/option-chain` (handled by `get_nse_session(referer=...)`).
  - For testing, disable SSL verification: set `NSE_VERIFY_SSL=false` (expect InsecureRequestWarning). Re‑enable for production.
  - Avoid hammering; the module sleeps 1.0–2.5s between requests and retries with jitter.

- Empty DataFrames in apps
  - NSE may be temporarily blocking; try refresh in the sidebar after a minute.
  - Validate `symbol` and `expiry` (must match NSE exact format for v3 endpoint). The code falls back to indices/equities endpoints if needed.

- Timezone/Datetime
  - All internal comparisons normalized to UTC; parsing uses safe fallbacks.

---

## Notes & Limitations

- Greeks are placeholders; Black‑Scholes/IV wiring is a future enhancement.
- Lot sizes vary across instruments; strategy sizing uses conservative placeholders.
- Do not over‑request; respect NSE servers and applicable terms of use.

---

## Examples

Python usage for option chain analytics:

```python
import pandas as pd
import derivatives_analytics as da

# Option chain for NIFTY
df = da.fetch_option_chain("Indices", "NIFTY", "04-Nov-2025")
if not df.empty:
    enr = da.enrich_chain_with_totals(df)
    pcr, v_pcr = da.compute_pcr(enr)
    mp, table = da.max_pain(enr)
    sr = da.supports_resistances(enr)
    print("PCR", pcr, "vPCR", v_pcr, "MaxPain", mp, "S/R", sr)
```

Event scoring:

```python
import market_events as me

ann = me.fetch_corporate_announcements()
brd = me.fetch_board_meetings()
act = me.fetch_corporate_actions()
res = me.fetch_financial_results("insurance", "Quarterly")
ranked = me.score_events(ann, brd, act, res)
print(ranked.head())
```

---

## License

No license is included. If you intend to distribute this code, add an appropriate license file as needed.

---

## Disclaimer

This project is for educational purposes only and does not constitute investment advice. Market data may be delayed, incomplete, or unavailable. Always verify results independently before making any financial decisions.
