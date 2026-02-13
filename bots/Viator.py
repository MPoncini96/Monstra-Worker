"""
VIATOR (stateful signal generator) + Universe Pruning (keep 7 tickers per country)

What this file gives you:
1) prune_country_universe(...)  -> downloads data, drops dead tickers, keeps exactly 7 per country
2) run_viator_stateful(...)     -> weekly rebalance (W-FRI snapped), selects best country by momentum score,
                                  outputs target_weights + updated state

Notes:
- This is "live/signal" logic meant to behave like your backtest:
  rebalance only on schedule; otherwise HOLD current weights.
- No kill-switch here (your Viator backtest doesn’t include one).
- State persistence is up to you (KV/DB/file). You pass state in; you get new_state out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


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
    min_stocks_per_country: int = 7        # you asked to keep 7 per country
    keep_per_country: int = 7              # exactly 7 survivors per country
    min_points: int = 80                   # minimum non-NA Adj Close points for ticker to be "valid"

    # Cleaning
    drop_na_threshold: float = 0.98        # used after download; drops tickers with too much missing data
    prefer: str = "coverage"               # "coverage" or "recent" ranking when keeping 7
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

# Quick fixups for known broken symbols in your log.
# Feel free to expand this list over time.
TICKER_FIXUPS: dict[str, str | None] = {
    "RDS.A": "SHEL",     # old shell
    "DPWGY": "DHLGY",    # DHL
    "DPSGY": "DHLGY",
    "ANZBY": "ANZGY",    # ANZ ADR variant (often more stable)
    "FMG": "FSUGY",      # Fortescue ADR
    "WBC": "WBKCY",      # Westpac OTC ADR-ish symbol (yfinance varies)
    "ORAN": "ORANY",     # Orange ADR variant
    "ABB": "ABBN.SW",    # ABB on SIX exchange
    "CS": None,          # Credit Suisse delisted
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
        # de-dupe keep order
        cleaned[country] = list(dict.fromkeys(out))
    return cleaned


# ----------------------------
# Pruning (keep 7 per country)
# ----------------------------

def _extract_adj_close(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" in df.columns.get_level_values(0)):
            return df["Adj Close"].copy()
        # fallback if yahoo changed column naming
        if ("Close" in df.columns.get_level_values(0)):
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
    """
    For each country:
      - download tickers in one batch (Adj Close)
      - drop tickers with < cfg.min_points non-NA points
      - keep exactly cfg.keep_per_country by ranking
    Countries unable to keep cfg.keep_per_country are dropped.

    Returns: (pruned_map, report_df)
    """
    rows = []
    pruned: Dict[str, List[str]] = {}

    for country, tickers in country_map.items():
        tickers = list(dict.fromkeys([t.strip() for t in tickers if t and isinstance(t, str)]))
        n_original = len(tickers)

        if n_original == 0:
            rows.append({"country": country, "n_original": 0, "n_valid": 0, "kept": "", "dropped": ""})
            continue

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

        # strict validity by non-NA points
        non_na_counts = px.notna().sum(axis=0)
        valid = non_na_counts[non_na_counts >= cfg.min_points].index.tolist()

        # if not enough valid, drop country
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

        # rank & keep exactly 7
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

def download_prices_for_universe(
    tickers: List[str],
    cfg: ViatorConfig,
) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        period=cfg.history_period,
        auto_adjust=False,
        progress=False,
        threads=cfg.threads,
        group_by="column",
    )
    px = _extract_adj_close(data, tickers).sort_index()
    return px

def clean_prices(px: pd.DataFrame, drop_na_threshold: float) -> pd.DataFrame:
    if px.empty:
        return px
    coverage = 1.0 - px.isna().mean()
    keep = coverage[coverage >= drop_na_threshold].index.tolist()
    px2 = px[keep].copy()
    return px2.ffill()

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
    """
    Stateful weekly rotation signal:
    - Uses country_map (ideally already pruned to 7 per country)
    - Rebalances on rb dates only
    - Selects best country based on weighted momentum of top_k tickers
    - Returns signal + payload + updated state
    """
    if cfg is None:
        cfg = ViatorConfig()
    if state is None:
        state = {}

    ts = datetime.now(timezone.utc)

    current_weights = pd.Series(state.get("current_weights", {}), dtype=float)
    current_country = state.get("current_country")
    last_rebalance_date = state.get("last_rebalance_date")  # ISO string or None

    # Universe tickers
    all_tickers = sorted({t for lst in country_map.values() for t in lst})
    px = download_prices_for_universe(all_tickers, cfg)
    px = clean_prices(px, cfg.drop_na_threshold)

    if px.empty or len(px.index) < (cfg.momentum_lookback_days + 5):
        return {
            "bot_id": "viator",
            "ts": ts,
            "signal": "HOLD",
            "note": "Not enough price history; holding current weights",
            "payload": {
                "target_weights": {k: float(v) for k, v in current_weights.items() if float(v) != 0.0},
                "selected_country": current_country,
                "lookback_days": cfg.momentum_lookback_days,
                "rebalance_rule": cfg.rebalance_rule,
            },
            "state": state,
        }

    asof = px.index[-1]

    rb_dates = get_rebalance_dates(px.index, cfg.rebalance_rule)
    rebalance_due = asof in set(rb_dates)

    # Avoid double-rebalance on same asof date
    if last_rebalance_date is not None and pd.Timestamp(last_rebalance_date) == asof:
        rebalance_due = False

    mom = compute_momentum(px, cfg.momentum_lookback_days)

    # if momentum not available on asof, don't rebalance
    if asof not in mom.index or mom.loc[asof].dropna().empty:
        rebalance_due = False

    signal = "HOLD"
    note = f"As of {asof.date()}: holding {current_country or 'no position'}"
    best_score = float("nan")

    if rebalance_due:
        try:
            sel_country, holdings, scores = select_country_and_holdings(mom, asof, country_map, cfg)
            new_weights = {t: float(w) for t, w in holdings}

            current_weights = pd.Series(new_weights, dtype=float)
            current_country = sel_country
            last_rebalance_date = str(asof)

            top_names = ", ".join([t for t, _ in holdings])
            signal = "REBALANCE"
            best_score = float(scores.get(sel_country, np.nan))
            note = f"As of {asof.date()}: selected {sel_country}; holdings={top_names}"
        except Exception as e:
            # If selection fails, HOLD
            return {
                "bot_id": "viator",
                "ts": ts,
                "signal": "HOLD",
                "note": f"As of {asof.date()}: selection failed; holding. ({e})",
                "payload": {
                    "asof": str(asof),
                    "target_weights": {k: float(v) for k, v in current_weights.items() if float(v) != 0.0},
                    "selected_country": current_country,
                    "lookback_days": cfg.momentum_lookback_days,
                    "rebalance_rule": cfg.rebalance_rule,
                },
                "state": state,
            }

    new_state = {
        "current_country": current_country,
        "last_rebalance_date": last_rebalance_date,
        "current_weights": {k: float(v) for k, v in current_weights.items() if float(v) != 0.0},
        "last_asof": str(asof),
    }

    return {
        "bot_id": "viator",
        "ts": ts,
        "signal": signal,
        "note": note,
        "payload": {
            "asof": str(asof),
            "lookback_days": cfg.momentum_lookback_days,
            "rebalance_rule": cfg.rebalance_rule,
            "selected_country": current_country,
            "target_weights": {k: float(v) for k, v in current_weights.items() if float(v) != 0.0},
            "best_country_score": best_score,
        },
        "state": new_state,
    }


# ----------------------------
# One-call helper (prune then run)
# ----------------------------

def run_viator_with_pruning(
    raw_country_map: Dict[str, List[str]],
    cfg: ViatorConfig | None = None,
    state: dict | None = None,
) -> dict:
    """
    Convenience wrapper:
      - applies known ticker fixups
      - prunes to exactly 7 per country (dropping countries that can't)
      - runs stateful Viator signal on the pruned map
    Returns signal dict with 'universe_report' included for debugging.
    """
    if cfg is None:
        cfg = ViatorConfig()

    fixed = apply_ticker_fixups(raw_country_map)
    pruned_map, report = prune_country_universe(fixed, cfg)

    result = run_viator_stateful(pruned_map, cfg=cfg, state=state)

    # attach report summary (optional; remove if you want smaller payloads)
    result["universe"] = {
        "countries": list(pruned_map.keys()),
        "per_country": cfg.keep_per_country,
    }
    result["universe_report"] = report.to_dict(orient="records")
    return result


if __name__ == "__main__":
    cfg = ViatorConfig(
        history_period="9mo",
        momentum_lookback_days=10,
        rebalance_rule="W-FRI",
        min_stocks_per_country=7,
        keep_per_country=7,
        min_points=80,
        prefer="coverage",
        threads=False,
    )

    # First run (no state yet)
    state = {}
    out = run_viator_with_pruning(COUNTRY_TICKERS, cfg=cfg, state=state)
    print(out["signal"], out["note"])
    print("Weights:", out["payload"]["target_weights"])
    print("Selected country:", out["payload"]["selected_country"])

    # Next run (persist state between calls in your DB/KV)
    state2 = out["state"]
    out2 = run_viator_with_pruning(COUNTRY_TICKERS, cfg=cfg, state=state2)
    print(out2["signal"], out2["note"])

