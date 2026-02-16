"""
VIATOR (stateful signal generator) + Universe Pruning (keep 7 tickers per country)

Updates in this version:
- More robust ticker fixups for symbols that commonly fail in yfinance (OTC ADR issues).
- Pruning now *suppresses* yfinance download noise and treats failed tickers as "invalid" quietly.
- Payload is JSON-safe: replaces NaN/Inf with None recursively to avoid Postgres jsonb errors.
- best_country_score is now None when not available (never NaN).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import math
import contextlib
import logging

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Logging (silence yfinance spam)
# ----------------------------

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def _quiet_yfinance(level: int = logging.ERROR):
    """
    Temporarily reduce noisy loggers while calling yfinance.
    yfinance/pandas can emit a lot of warnings for missing tickers.
    """
    targets = ["yfinance", "urllib3", "requests"]
    old = {}
    for name in targets:
        lg = logging.getLogger(name)
        old[name] = lg.level
        lg.setLevel(level)
    try:
        yield
    finally:
        for name, lvl in old.items():
            logging.getLogger(name).setLevel(lvl)


# ----------------------------
# Config
# ----------------------------

@dataclass
class ViatorConfig:
    # Live downloads: period keeps things light for serverless
    history_period: str = "9mo"

    # Strategy
    momentum_lookback_days: int = 10
    rebalance_rule: str = "W-FRI"          # weekly Friday, snapped to last trading day
    top_k: int = 4
    weights: np.ndarray = field(
        default_factory=lambda: np.array([0.40, 0.30, 0.20, 0.10], dtype=float)
    )

    # Universe constraints
    min_stocks_per_country: int = 7
    keep_per_country: int = 7
    min_points: int = 80

    # Cleaning
    drop_na_threshold: float = 0.98
    prefer: str = "coverage"               # "coverage" or "recent"
    threads: bool = False                  # yfinance more reliable with threads=False


# ----------------------------
# Example Universe (edit this)
# ----------------------------

COUNTRY_TICKERS: Dict[str, List[str]] = {
    "Canada": ["RY", "TD", "BNS", "BMO", "CM", "ENB", "SU", "CNQ", "SHOP", "TRP", "BCE", "CNI"],
    "UK": ["HSBC", "BP", "SHEL", "UL", "AZN", "GSK", "BCS", "RIO", "BTI", "DEO", "RELX", "NGG"],
    "Germany": ["SAP", "DTEGY", "VWAGY", "BMWYY", "MBGYY", "BASFY", "SIEGY", "IFNNY", "ALIZY", "BAYRY", "DHLGY", "PAH3.DE"],
    "France": ["TTE", "BNPQY", "VIVHY", "ORAN", "SAN", "AIQUY", "LRLCY", "CAPMF", "SGSOY", "ENGIY", "EL", "RMS.PA"],
    "Japan": ["TM", "HMC", "SONY", "NTDOY", "MUFG", "SMFG", "NTTYY", "TAK", "KDDIY", "FANUY", "SNEJF", "MZDAY"],
    "China": ["BABA", "JD", "PDD", "BIDU", "NTES", "LI", "NIO", "XPEV", "BILI", "YUMC", "ZTO", "BEKE"],
    "South Korea": ["SSNLF", "HYMTF", "KB", "SHG", "KT", "SKM", "LPL", "KSC", "DOX", "WOR", "LGCLF", "KIMTF"],
    "Taiwan": ["TSM", "UMC", "ASX", "AUOTY", "LITE", "CHT", "IMOS", "GIGM", "TSYHY", "ACTTF", "WDC", "ACLS"],
    "Brazil": ["VALE", "PBR", "ITUB", "BBD", "BSBR", "ABEV", "ELET", "ERJ", "SUZ", "BRFS", "GGB", "SID"],
    "Mexico": ["AMX", "FMX", "CX", "CEMXY", "OMAB", "PAC", "ASR", "TV", "KOF", "GMBXF", "SIM", "WMMVY"],
    "India": ["INFY", "WIT", "HDB", "IBN", "RDY", "MMYT", "SIFY", "VEDL", "IGIC", "YTRA", "BSEFY"],
    "Australia": ["BHP", "RIO", "WDS", "FMG", "CSL", "WBC", "NABZY", "ANZBY", "TLSYY", "WOW", "MQBKY", "QBEIF"],
    "South Africa": ["SBSW", "GFI", "MTNOY", "VOD", "NPSNY", "ANGPY", "AGPPY", "SOUHY", "RNECY", "KRO", "TECK", "DRD"],
    "Netherlands": ["ASML", "ING", "PHG", "SHEL", "HEINY", "TKPHF", "WLSCY", "PNDHF", "AEG", "RNLXY", "STM"],
    "Switzerland": ["NVS", "RHHBY", "NSRGY", "UBS", "ABB", "ZURVY", "GEBN", "ROG", "CS"],
}

# ----------------------------
# Fixups for flaky/changed tickers
# ----------------------------

TICKER_FIXUPS: dict[str, str | None] = {
    # Existing
    "RDS.A": "SHEL",
    "DPWGY": "DHLGY",
    "DPSGY": "DHLGY",
    "ANZBY": "ANZGY",
    "FMG": "FSUGY",
    "ORAN": "ORANY",
    "ABB": "ABBN.SW",
    "CS": None,              # Credit Suisse delisted

    # New (from your Render logs)
    "GEBN": "GEBN.SW",        # Geberit -> SIX
    "BMWYY": "BMW.DE",        # BMW -> XETRA (more reliable than OTC ADR)
    "WLSCY": None,            # often flaky OTC; drop
    "AGPPY": None,            # often flaky OTC; drop
    "TLSYY": None,            # Telstra OTC often flaky; drop
    "WBKCY": None,            # Westpac OTC often flaky; drop
    "BSEFY": None,            # BSE ADR often flaky; drop
    "VEDL": None,             # Vedanta ADR often flaky; drop
    "CEMXY": None,            # Cemex ADR variant flaky; you already have CX
    "ELET": None,             # Eletrobras variants can be flaky; drop if failing
    "ERJ": None,              # Embraer can be flaky; drop if failing
    "BRFS": None,             # BRF can be flaky; drop if failing
    "ACTTF": None,            # often flaky OTC; drop
    "HYMTF": None,            # 404 in your log; drop
    "LGCLF": None,            # often flaky OTC; drop
    "KSC": None,              # unclear; drop
}

def apply_ticker_fixups(country_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    cleaned: Dict[str, List[str]] = {}
    for country, tickers in country_map.items():
        out: List[str] = []
        for t in tickers:
            t2 = TICKER_FIXUPS.get(t, t)
            if t2 is None:
                continue
            out.append(t2)
        cleaned[country] = list(dict.fromkeys(out))  # de-dupe keep order
    return cleaned


# ----------------------------
# JSON safety helpers
# ----------------------------

def _json_safe(x):
    """Replace NaN/Inf with None recursively (JSON-safe for Postgres jsonb)."""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return x


# ----------------------------
# Pruning (keep 7 per country)
# ----------------------------

def _extract_adj_close(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            return df["Adj Close"].copy()
        if "Close" in df.columns.get_level_values(0):
            return df["Close"].copy()
        return pd.DataFrame()
    # single ticker case
    if "Adj Close" in df.columns:
        return df[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    if "Close" in df.columns:
        return df[["Close"]].rename(columns={"Close": tickers[0]})
    return pd.DataFrame()


def prune_country_universe(
    country_map: Dict[str, List[str]],
    cfg: ViatorConfig,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    rows = []
    pruned: Dict[str, List[str]] = {}

    for country, tickers in country_map.items():
        tickers = list(dict.fromkeys([t.strip() for t in tickers if t and isinstance(t, str)]))
        n_original = len(tickers)

        if n_original == 0:
            rows.append({"country": country, "n_original": 0, "n_valid": 0, "kept": "", "dropped": ""})
            continue

        with _quiet_yfinance():
            data = yf.download(
                tickers=tickers,
                period=cfg.history_period,
                auto_adjust=False,
                progress=False,
                threads=cfg.threads,
                group_by="column",
            )

        px = _extract_adj_close(data, tickers).sort_index()

        if px.empty:
            rows.append({
                "country": country,
                "n_original": n_original,
                "n_valid": 0,
                "kept": "",
                "dropped": ", ".join(tickers[:40]) + (" ..." if len(tickers) > 40 else ""),
            })
            continue

        non_na_counts = px.notna().sum(axis=0)
        valid = non_na_counts[non_na_counts >= cfg.min_points].index.tolist()

        if len(valid) < cfg.keep_per_country:
            dropped = [t for t in tickers if t not in valid]
            rows.append({
                "country": country,
                "n_original": n_original,
                "n_valid": len(valid),
                "kept": ", ".join(valid),
                "dropped": ", ".join(dropped[:40]) + (" ..." if len(dropped) > 40 else ""),
            })
            continue

        last_valid_date = px[valid].apply(lambda s: s.dropna().index.max())
        rank_df = pd.DataFrame({"count": non_na_counts[valid], "last": last_valid_date})

        if cfg.prefer == "recent":
            rank_df = rank_df.sort_values(["last", "count"], ascending=[False, False])
        else:
            rank_df = rank_df.sort_values(["count", "last"], ascending=[False, False])

        kept = rank_df.head(cfg.keep_per_country).index.tolist()
        dropped = [t for t in tickers if t not in kept]

        pruned[country] = kept
        rows.append({
            "country": country,
            "n_original": n_original,
            "n_valid": len(valid),
            "kept": ", ".join(kept),
            "dropped": ", ".join(dropped[:40]) + (" ..." if len(dropped) > 40 else ""),
        })

    report = pd.DataFrame(rows).sort_values(["country"]).reset_index(drop=True)
    return pruned, report


# ----------------------------
# Viator signal logic (stateful)
# ----------------------------

def compute_momentum(px: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return px.pct_change(lookback)

def get_rebalance_dates(index: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    df = pd.DataFrame(index=index, data={"x": 1})
    rb = df.resample(rule).last().dropna().index
    return rb.intersection(index)

def download_prices_for_universe(tickers: List[str], cfg: ViatorConfig) -> pd.DataFrame:
    with _quiet_yfinance():
        data = yf.download(
            tickers=tickers,
            period=cfg.history_period,
            auto_adjust=False,
            progress=False,
            threads=cfg.threads,
            group_by="column",
        )
    return _extract_adj_close(data, tickers).sort_index()

def clean_prices(px: pd.DataFrame, drop_na_threshold: float) -> pd.DataFrame:
    if px.empty:
        return px
    coverage = 1.0 - px.isna().mean()
    keep = coverage[coverage >= drop_na_threshold].index.tolist()
    return px[keep].copy().ffill()

def select_country_and_holdings(
    mom: pd.DataFrame,
    date: pd.Timestamp,
    country_map: Dict[str, List[str]],
    cfg: ViatorConfig,
) -> Tuple[str, List[Tuple[str, float]], Dict[str, float]]:
    scores: Dict[str, float] = {}
    country_top: Dict[str, List[str]] = {}

    for country, tickers in country_map.items():
        available = [t for t in tickers if t in mom.columns]
        if len(available) < cfg.min_stocks_per_country:
            continue

        m = mom.loc[date, available].dropna()
        if len(m) < cfg.top_k:
            continue

        top = m.sort_values(ascending=False).head(cfg.top_k)
        w = cfg.weights[: cfg.top_k]
        score = float(np.dot(w, top.values))
        scores[country] = score
        country_top[country] = top.index.tolist()

    if not scores:
        raise ValueError(f"No eligible countries on {date.date()} (check tickers/data/thresholds).")

    best_country = max(scores, key=scores.get)
    best_tickers = country_top[best_country]
    holdings = list(zip(best_tickers, cfg.weights[: cfg.top_k].tolist()))
    return best_country, holdings, scores


def run_viator_stateful(
    country_map: Dict[str, List[str]],
    cfg: ViatorConfig | None = None,
    state: dict | None = None,
) -> dict:
    if cfg is None:
        cfg = ViatorConfig()
    if state is None:
        state = {}

    ts = datetime.now(timezone.utc)

    current_weights = pd.Series(state.get("current_weights", {}), dtype=float)
    current_country = state.get("current_country")
    last_rebalance_date = state.get("last_rebalance_date")

    all_tickers = sorted({t for lst in country_map.values() for t in lst})
    px = download_prices_for_universe(all_tickers, cfg)
    px = clean_prices(px, cfg.drop_na_threshold)

    if px.empty or len(px.index) < (cfg.momentum_lookback_days + 5):
        payload = {
            "target_weights": {k: float(v) for k, v in current_weights.items() if float(v) != 0.0},
            "selected_country": current_country,
            "lookback_days": cfg.momentum_lookback_days,
            "rebalance_rule": cfg.rebalance_rule,
        }
        return {
            "bot_id": "viator",
            "ts": ts,
            "signal": "HOLD",
            "note": "Not enough price history; holding current weights",
            "payload": _json_safe(payload),
            "state": state,
        }

    asof = px.index[-1]

    rb_dates = get_rebalance_dates(px.index, cfg.rebalance_rule)
    rebalance_due = asof in set(rb_dates)

    if last_rebalance_date is not None and pd.Timestamp(last_rebalance_date) == asof:
        rebalance_due = False

    mom = compute_momentum(px, cfg.momentum_lookback_days)

    if asof not in mom.index or mom.loc[asof].dropna().empty:
        rebalance_due = False

    signal = "HOLD"
    note = f"As of {asof.date()}: holding {current_country or 'no position'}"
    best_score = None  # NEVER NaN

    if rebalance_due:
        try:
            sel_country, holdings, scores = select_country_and_holdings(mom, asof, country_map, cfg)
            new_weights = {t: float(w) for t, w in holdings}

            current_weights = pd.Series(new_weights, dtype=float)
            current_country = sel_country
            last_rebalance_date = str(asof)

            top_names = ", ".join([t for t, _ in holdings])
            signal = "REBALANCE"
            best_score = float(scores.get(sel_country)) if sel_country in scores else None
            note = f"As of {asof.date()}: selected {sel_country}; holdings={top_names}"
        except Exception as e:
            payload = {
                "asof": str(asof),
                "target_weights": {k: float(v) for k, v in current_weights.items() if float(v) != 0.0},
                "selected_country": current_country,
                "lookback_days": cfg.momentum_lookback_days,
                "rebalance_rule": cfg.rebalance_rule,
                "error": str(e),
            }
            return {
                "bot_id": "viator",
                "ts": ts,
                "signal": "HOLD",
                "note": f"As of {asof.date()}: selection failed; holding. ({e})",
                "payload": _json_safe(payload),
                "state": state,
            }

    new_state = {
        "current_country": current_country,
        "last_rebalance_date": last_rebalance_date,
        "current_weights": {k: float(v) for k, v in current_weights.items() if float(v) != 0.0},
        "last_asof": str(asof),
    }

    payload = {
        "asof": str(asof),
        "lookback_days": cfg.momentum_lookback_days,
        "rebalance_rule": cfg.rebalance_rule,
        "selected_country": current_country,
        "target_weights": {k: float(v) for k, v in current_weights.items() if float(v) != 0.0},
        "best_country_score": best_score,
    }

    return {
        "bot_id": "viator",
        "ts": ts,
        "signal": signal,
        "note": note,
        "payload": _json_safe(payload),
        "state": new_state,
    }


def run_viator_with_pruning(
    raw_country_map: Dict[str, List[str]],
    cfg: ViatorConfig | None = None,
    state: dict | None = None,
) -> dict:
    if cfg is None:
        cfg = ViatorConfig()

    fixed = apply_ticker_fixups(raw_country_map)
    pruned_map, report = prune_country_universe(fixed, cfg)

    result = run_viator_stateful(pruned_map, cfg=cfg, state=state)

    result["universe"] = {
        "countries": list(pruned_map.keys()),
        "per_country": cfg.keep_per_country,
    }

    # WARNING: this can be big; keep for debugging, remove later if desired
    result["universe_report"] = report.to_dict(orient="records")

    # Ensure JSON-safe output
    result["payload"] = _json_safe(result.get("payload", {}))
    result["universe"] = _json_safe(result.get("universe", {}))
    result["universe_report"] = _json_safe(result.get("universe_report", []))
    return result
