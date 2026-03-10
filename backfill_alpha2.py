#!/usr/bin/env python
"""
Production-grade Alpha2 backfill script with:
- proxy-based ranking
- configurable rank weights
- rebalance frequency support
- risk-off / kill-switch support
- benchmark filter support
- cash-equivalent fallback
- transaction costs + slippage
- bulk upserts into trading.bot_equity

Timing convention:
- Holdings for trading date t are selected using data available through t-1
- If t is not a rebalance day, prior holdings are carried forward
- Return applied is the return realized on trading date t
- Trading costs are charged only when holdings change
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from psycopg2.extras import execute_values

# Ensure project root import path is available
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_loader import load_env
from db import get_conn

load_env()

# ============================================================
# Configuration
# ============================================================

START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()

DEFAULT_RANK_WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)

REQUIRE_FULL_LOOKBACK_WINDOW = True
USE_CASH_EQUIVALENT_FALLBACK = True

DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 5.0

UPSERT_BATCH_SIZE = 500

LOG_LEVEL = os.getenv("ALPHA2_BACKFILL_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("alpha2_backfill")


# ============================================================
# Data structures
# ============================================================

@dataclass
class Alpha2BotConfig:
    bot_id: str
    proxies: dict[str, str]              # logical_name -> ticker
    stocks: dict[str, list[str]]         # logical_name -> constituent stocks / metadata
    lookback_days: int
    rebalance_freq: str | int | None
    stock_bench_mode: str | None
    rank_weights: np.ndarray

    cash_equivalent: str | None = None
    benchmark: str | None = None

    enable_kill_switch: bool = True
    enable_benchmark_filter: bool = False
    kill_switch_mode: str = "top_negative"      # top_negative | avg_negative | off
    benchmark_mode: str = "benchmark_positive"  # benchmark_positive | benchmark_gt_cash | off

    top_return_threshold: float = 0.0
    benchmark_return_threshold: float = 0.0

    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS


@dataclass
class BackfillRow:
    bot_id: str
    d: date
    equity: float
    ret: float
    holdings_json: str


# ============================================================
# Utility helpers
# ============================================================

def safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def clean_ticker(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    return s or None


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def bps_to_decimal(bps: float) -> float:
    return float(bps) / 10_000.0


def normalize_proxy_map(raw: Any) -> dict[str, str]:
    """
    Normalize proxies JSONB into:
      { logical_name: TICKER }
    """
    parsed = safe_json_loads(raw)
    if parsed is None:
        return {}

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected proxies to be dict, got {type(parsed).__name__}")

    result: dict[str, str] = {}
    for k, v in parsed.items():
        logical = str(k).strip()
        ticker = clean_ticker(v)
        if logical and ticker:
            result[logical] = ticker
    return result


def normalize_stocks_map(raw: Any) -> dict[str, list[str]]:
    """
    Normalize stocks JSONB into:
      { logical_name: [TICKER1, TICKER2, ...] }
    """
    parsed = safe_json_loads(raw)
    if parsed is None:
        return {}

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected stocks to be dict, got {type(parsed).__name__}")

    result: dict[str, list[str]] = {}
    for k, v in parsed.items():
        logical = str(k).strip()
        if not logical:
            continue

        if isinstance(v, list):
            tickers = []
            seen = set()
            for item in v:
                ticker = clean_ticker(item)
                if ticker and ticker not in seen:
                    seen.add(ticker)
                    tickers.append(ticker)
            result[logical] = tickers
        else:
            ticker = clean_ticker(v)
            result[logical] = [ticker] if ticker else []
    return result


def normalize_rank_weights(raw: Any) -> np.ndarray:
    parsed = safe_json_loads(raw)
    if parsed is None:
        return DEFAULT_RANK_WEIGHTS.copy()

    if not isinstance(parsed, (list, tuple)):
        return DEFAULT_RANK_WEIGHTS.copy()

    arr = np.array(parsed, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return DEFAULT_RANK_WEIGHTS.copy()

    if np.any(arr < 0):
        arr = arr[arr >= 0]

    if arr.size == 0 or arr.sum() <= 0:
        return DEFAULT_RANK_WEIGHTS.copy()

    return arr / arr.sum()


def parse_rebalance_freq(value: Any) -> int:
    """
    Supports:
    - integer-like values: 1, 5, 21
    - strings: daily, weekly, monthly
    Defaults to 1 (daily)
    """
    if value is None:
        return 1

    if isinstance(value, (int, float)) and int(value) > 0:
        return int(value)

    s = str(value).strip().lower()
    mapping = {
        "daily": 1,
        "every_day": 1,
        "weekly": 5,
        "every_week": 5,
        "monthly": 21,
        "every_month": 21,
    }
    if s in mapping:
        return mapping[s]

    try:
        n = int(s)
        return max(1, n)
    except ValueError:
        return 1


# ============================================================
# Database helpers
# ============================================================

def fetch_active_alpha2_bots() -> list[Alpha2BotConfig]:
    """
    Attempts extended config first; falls back if optional columns do not exist.
    """
    primary_query = """
        SELECT
            bot_id,
            proxies,
            stocks,
            lookback_days,
            rebalance_freq,
            stock_bench_mode,
            rank_weights,
            cash_equivalent,
            benchmark,
            COALESCE(enable_kill_switch, true) AS enable_kill_switch,
            COALESCE(enable_benchmark_filter, false) AS enable_benchmark_filter,
            COALESCE(kill_switch_mode, 'top_negative') AS kill_switch_mode,
            COALESCE(benchmark_mode, 'benchmark_positive') AS benchmark_mode,
            COALESCE(top_return_threshold, 0.0) AS top_return_threshold,
            COALESCE(benchmark_return_threshold, 0.0) AS benchmark_return_threshold,
            COALESCE(transaction_cost_bps, %s) AS transaction_cost_bps,
            COALESCE(slippage_bps, %s) AS slippage_bps
        FROM trading.alpha2
        WHERE is_active = true
        ORDER BY bot_id
    """

    fallback_query = """
        SELECT
            bot_id,
            proxies,
            stocks,
            lookback_days,
            rebalance_freq,
            stock_bench_mode,
            rank_weights
        FROM trading.alpha2
        WHERE is_active = true
        ORDER BY bot_id
    """

    rows = []
    extended = True

    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    primary_query,
                    (DEFAULT_TRANSACTION_COST_BPS, DEFAULT_SLIPPAGE_BPS),
                )
                rows = cur.fetchall()
            except Exception:
                conn.rollback()
                extended = False
                cur.execute(fallback_query)
                rows = cur.fetchall()

    configs: list[Alpha2BotConfig] = []

    if extended:
        for row in rows:
            (
                bot_id,
                proxies_raw,
                stocks_raw,
                lookback_days,
                rebalance_freq,
                stock_bench_mode,
                rank_weights_raw,
                cash_equivalent,
                benchmark,
                enable_kill_switch,
                enable_benchmark_filter,
                kill_switch_mode,
                benchmark_mode,
                top_return_threshold,
                benchmark_return_threshold,
                transaction_cost_bps,
                slippage_bps,
            ) = row

            configs.append(
                Alpha2BotConfig(
                    bot_id=str(bot_id),
                    proxies=normalize_proxy_map(proxies_raw),
                    stocks=normalize_stocks_map(stocks_raw),
                    lookback_days=int(lookback_days),
                    rebalance_freq=rebalance_freq,
                    stock_bench_mode=str(stock_bench_mode) if stock_bench_mode is not None else None,
                    rank_weights=normalize_rank_weights(rank_weights_raw),
                    cash_equivalent=clean_ticker(cash_equivalent),
                    benchmark=clean_ticker(benchmark),
                    enable_kill_switch=bool(enable_kill_switch),
                    enable_benchmark_filter=bool(enable_benchmark_filter),
                    kill_switch_mode=str(kill_switch_mode),
                    benchmark_mode=str(benchmark_mode),
                    top_return_threshold=safe_float(top_return_threshold, 0.0),
                    benchmark_return_threshold=safe_float(benchmark_return_threshold, 0.0),
                    transaction_cost_bps=safe_float(transaction_cost_bps, DEFAULT_TRANSACTION_COST_BPS),
                    slippage_bps=safe_float(slippage_bps, DEFAULT_SLIPPAGE_BPS),
                )
            )
    else:
        for row in rows:
            (
                bot_id,
                proxies_raw,
                stocks_raw,
                lookback_days,
                rebalance_freq,
                stock_bench_mode,
                rank_weights_raw,
            ) = row

            configs.append(
                Alpha2BotConfig(
                    bot_id=str(bot_id),
                    proxies=normalize_proxy_map(proxies_raw),
                    stocks=normalize_stocks_map(stocks_raw),
                    lookback_days=int(lookback_days),
                    rebalance_freq=rebalance_freq,
                    stock_bench_mode=str(stock_bench_mode) if stock_bench_mode is not None else None,
                    rank_weights=normalize_rank_weights(rank_weights_raw),
                )
            )

    return configs


def fetch_previous_equity(bot_id: str, start_date: str) -> float:
    query = """
        SELECT equity
        FROM trading.bot_equity
        WHERE bot_id = %s AND d < %s
        ORDER BY d DESC
        LIMIT 1
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (bot_id, start_date))
            row = cur.fetchone()

    if row and is_finite_number(row[0]):
        return float(row[0])
    return 1.0


def fetch_previous_holdings(bot_id: str, start_date: str) -> dict[str, float]:
    query = """
        SELECT holdings
        FROM trading.bot_equity
        WHERE bot_id = %s AND d < %s
        ORDER BY d DESC
        LIMIT 1
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (bot_id, start_date))
            row = cur.fetchone()

    if not row or row[0] is None:
        return {}

    raw = row[0]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    if not isinstance(raw, dict):
        return {}

    cleaned: dict[str, float] = {}
    for k, v in raw.items():
        if str(k).startswith("_"):
            continue
        if is_finite_number(v):
            cleaned[str(k).upper()] = float(v)
    return cleaned


def upsert_backfill_rows(rows: list[BackfillRow]) -> None:
    if not rows:
        return

    sql = """
        INSERT INTO trading.bot_equity (bot_id, d, equity, ret, holdings)
        VALUES %s
        ON CONFLICT (bot_id, d)
        DO UPDATE SET
            equity = EXCLUDED.equity,
            ret = EXCLUDED.ret,
            holdings = EXCLUDED.holdings
    """

    values = [(r.bot_id, r.d, r.equity, r.ret, r.holdings_json) for r in rows]

    with get_conn() as conn:
        with conn.cursor() as cur:
            for batch in chunked(values, UPSERT_BATCH_SIZE):
                execute_values(cur, sql, batch)
        conn.commit()


# ============================================================
# Market data helpers
# ============================================================

def download_adjusted_close(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    raw = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data missing 'Close' field.")
        close_df = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data missing 'Close' column.")
        only_symbol = symbols[0]
        close_df = raw[["Close"]].rename(columns={"Close": only_symbol})

    close_df = close_df.sort_index()
    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df[~close_df.index.duplicated(keep="last")]
    close_df = close_df.dropna(how="all")
    close_df.columns = [str(c).upper() for c in close_df.columns]

    return close_df


def build_price_frame(bot: Alpha2BotConfig) -> pd.DataFrame:
    symbols = list(dict.fromkeys(bot.proxies.values()))

    for sym in [bot.cash_equivalent, bot.benchmark]:
        if sym and sym not in symbols:
            symbols.append(sym)

    return download_adjusted_close(symbols, START_DATE, END_DATE)


# ============================================================
# Strategy helpers
# ============================================================

def get_trailing_returns(
    prices: pd.DataFrame,
    end_idx_exclusive: int,
    lookback_days: int,
    symbols: set[str] | None = None,
) -> pd.Series:
    start_idx = max(0, end_idx_exclusive - lookback_days)
    lookback = prices.iloc[start_idx:end_idx_exclusive]

    if len(lookback) < 2:
        return pd.Series(dtype=float)

    first_row = lookback.iloc[0]
    last_row = lookback.iloc[-1]

    if REQUIRE_FULL_LOOKBACK_WINDOW:
        valid_mask = lookback.notna().all(axis=0)
        first_row = first_row[valid_mask]
        last_row = last_row[valid_mask]
    else:
        valid_mask = first_row.notna() & last_row.notna()
        first_row = first_row[valid_mask]
        last_row = last_row[valid_mask]

    trailing = (last_row / first_row) - 1.0
    trailing = trailing.replace([np.inf, -np.inf], np.nan).dropna()

    if symbols:
        trailing = trailing[trailing.index.isin(symbols)]

    return trailing


def evaluate_risk_off(
    bot: Alpha2BotConfig,
    ranked_trailing: pd.Series,
    benchmark_trailing: pd.Series,
    cash_trailing: pd.Series,
) -> tuple[bool, str]:
    reasons: list[str] = []

    if bot.enable_kill_switch and bot.kill_switch_mode != "off":
        if ranked_trailing.empty:
            reasons.append("kill_switch:no_ranked_candidates")
        elif bot.kill_switch_mode == "top_negative":
            top_ret = float(ranked_trailing.max())
            if top_ret <= bot.top_return_threshold:
                reasons.append(
                    f"kill_switch:top_ret={top_ret:.6f}<=threshold={bot.top_return_threshold:.6f}"
                )
        elif bot.kill_switch_mode == "avg_negative":
            avg_ret = float(ranked_trailing.mean())
            if avg_ret <= bot.top_return_threshold:
                reasons.append(
                    f"kill_switch:avg_ret={avg_ret:.6f}<=threshold={bot.top_return_threshold:.6f}"
                )

    if bot.enable_benchmark_filter and bot.benchmark_mode != "off" and bot.benchmark:
        bench_ret = benchmark_trailing.get(bot.benchmark, np.nan)

        if pd.isna(bench_ret):
            reasons.append("benchmark_filter:no_benchmark_data")
        else:
            bench_ret = float(bench_ret)

            if bot.benchmark_mode == "benchmark_positive":
                if bench_ret <= bot.benchmark_return_threshold:
                    reasons.append(
                        f"benchmark_filter:bench_ret={bench_ret:.6f}<=threshold={bot.benchmark_return_threshold:.6f}"
                    )
            elif bot.benchmark_mode == "benchmark_gt_cash":
                cash_ret = np.nan
                if bot.cash_equivalent:
                    cash_ret = cash_trailing.get(bot.cash_equivalent, np.nan)

                if pd.isna(cash_ret):
                    reasons.append("benchmark_filter:no_cash_data")
                else:
                    cash_ret = float(cash_ret)
                    if bench_ret <= cash_ret:
                        reasons.append(
                            f"benchmark_filter:bench_ret={bench_ret:.6f}<=cash_ret={cash_ret:.6f}"
                        )

    if reasons:
        return True, "; ".join(reasons)
    return False, "risk_on"


def choose_ranked_holdings(
    ranked_trailing: pd.Series,
    rank_weights: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    if ranked_trailing.empty:
        return [], np.array([], dtype=float)

    ranked = ranked_trailing.sort_values(ascending=False)
    n = min(len(ranked), len(rank_weights))
    selected = ranked.head(n).index.tolist()
    weights = rank_weights[:n].copy()
    weights = weights / weights.sum()

    return selected, weights


def choose_holdings_for_day(
    bot: Alpha2BotConfig,
    ranked_trailing: pd.Series,
    benchmark_trailing: pd.Series,
    cash_trailing: pd.Series,
) -> tuple[list[str], np.ndarray, bool, str]:
    risk_off, reason = evaluate_risk_off(
        bot=bot,
        ranked_trailing=ranked_trailing,
        benchmark_trailing=benchmark_trailing,
        cash_trailing=cash_trailing,
    )

    if risk_off:
        if USE_CASH_EQUIVALENT_FALLBACK and bot.cash_equivalent:
            return [bot.cash_equivalent], np.array([1.0], dtype=float), True, reason
        return [], np.array([], dtype=float), True, reason

    selected, weights = choose_ranked_holdings(ranked_trailing, bot.rank_weights)

    if not selected and USE_CASH_EQUIVALENT_FALLBACK and bot.cash_equivalent:
        return [bot.cash_equivalent], np.array([1.0], dtype=float), True, "fallback:no_selected_proxies"

    return selected, weights, False, reason


def compute_day_return_and_holdings(
    day_returns: pd.Series,
    selected_symbols: list[str],
    weights: np.ndarray,
) -> tuple[float, dict[str, float]]:
    if not selected_symbols or len(weights) == 0:
        return 0.0, {}

    valid_pairs: list[tuple[str, float]] = []
    for sym, w in zip(selected_symbols, weights):
        if sym in day_returns.index and pd.notna(day_returns[sym]):
            valid_pairs.append((sym, float(w)))

    if not valid_pairs:
        return 0.0, {}

    total_weight = sum(w for _, w in valid_pairs)
    if total_weight <= 0:
        return 0.0, {}

    normalized_pairs = [(sym, w / total_weight) for sym, w in valid_pairs]
    gross_ret = sum(day_returns[sym] * w for sym, w in normalized_pairs)
    holdings = {sym: float(w) for sym, w in normalized_pairs}
    return float(gross_ret), holdings


def compute_turnover(prev_holdings: dict[str, float], new_holdings: dict[str, float]) -> float:
    all_symbols = set(prev_holdings) | set(new_holdings)
    gross_change = sum(abs(new_holdings.get(sym, 0.0) - prev_holdings.get(sym, 0.0)) for sym in all_symbols)
    return gross_change / 2.0


def compute_cost_drag(turnover: float, transaction_cost_bps: float, slippage_bps: float) -> float:
    return turnover * bps_to_decimal(float(transaction_cost_bps) + float(slippage_bps))


def is_rebalance_day(day_counter: int, rebalance_every_n_days: int) -> bool:
    """
    Rebalance on first eligible day, then every N trading days thereafter.
    day_counter is count of processed eligible strategy days, not calendar days.
    """
    if rebalance_every_n_days <= 1:
        return True
    return (day_counter % rebalance_every_n_days) == 0


def build_holdings_payload(
    holdings: dict[str, float],
    bot: Alpha2BotConfig,
    selected_count: int,
    risk_off: bool,
    risk_reason: str,
    rebalanced_today: bool,
    gross_ret: float,
    net_ret: float,
    turnover: float,
    cost_drag: float,
) -> str:
    payload = dict(holdings)
    payload["_meta"] = {
        "is_backtest": True,
        "type": "alpha2",
        "data_source": "yfinance",
        "timing": "rank_on_t_minus_1_apply_return_on_t",
        "mode": bot.stock_bench_mode,
        "lookback_days": int(bot.lookback_days),
        "rebalance_freq": bot.rebalance_freq,
        "rebalanced_today": bool(rebalanced_today),
        "num_proxies": int(selected_count),
        "rank_weights": [float(x) for x in bot.rank_weights.tolist()],
        "risk_off": bool(risk_off),
        "risk_reason": str(risk_reason),
        "cash_equivalent": bot.cash_equivalent,
        "benchmark": bot.benchmark,
        "kill_switch_mode": bot.kill_switch_mode,
        "benchmark_mode": bot.benchmark_mode,
        "gross_ret": float(gross_ret),
        "net_ret": float(net_ret),
        "turnover": float(turnover),
        "transaction_cost_bps": float(bot.transaction_cost_bps),
        "slippage_bps": float(bot.slippage_bps),
        "cost_drag": float(cost_drag),
    }
    return json.dumps(payload)


# ============================================================
# Main bot logic
# ============================================================

def backfill_single_bot(bot: Alpha2BotConfig) -> dict[str, Any]:
    logger.info(
        "Starting bot=%s mode=%s proxies=%d lookback=%d rebalance=%s benchmark=%s cash=%s",
        bot.bot_id,
        bot.stock_bench_mode,
        len(bot.proxies),
        bot.lookback_days,
        bot.rebalance_freq,
        bot.benchmark,
        bot.cash_equivalent,
    )

    if not bot.proxies:
        logger.warning("Skipping bot=%s because proxies are empty", bot.bot_id)
        return {
            "bot_id": bot.bot_id,
            "processed_days": 0,
            "risk_off_days": 0,
            "final_equity": None,
            "final_return_pct": None,
            "errors": 0,
            "skipped": True,
        }

    prices = build_price_frame(bot)
    if prices.empty:
        logger.warning("Skipping bot=%s because no price data was downloaded", bot.bot_id)
        return {
            "bot_id": bot.bot_id,
            "processed_days": 0,
            "risk_off_days": 0,
            "final_equity": None,
            "final_return_pct": None,
            "errors": 0,
            "skipped": True,
        }

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    equity = fetch_previous_equity(bot.bot_id, START_DATE)
    prev_holdings = fetch_previous_holdings(bot.bot_id, START_DATE)

    rebalance_every_n_days = parse_rebalance_freq(bot.rebalance_freq)

    logger.info(
        "bot=%s trading_days=%d starting_equity=%.6f prior_holdings=%s rebalance_every=%d",
        bot.bot_id,
        len(prices),
        equity,
        prev_holdings,
        rebalance_every_n_days,
    )

    rows_to_upsert: list[BackfillRow] = []
    errors = 0
    processed_days = 0
    risk_off_days = 0
    strategy_day_counter = 0

    proxy_symbols = set(bot.proxies.values())
    benchmark_symbols = {bot.benchmark} if bot.benchmark else set()
    cash_symbols = {bot.cash_equivalent} if bot.cash_equivalent else set()

    current_target_holdings = prev_holdings.copy() if prev_holdings else {}

    for i, trading_ts in enumerate(prices.index):
        if i == 0:
            continue

        trading_day = trading_ts.date()

        try:
            strategy_day_counter += 1
            rebalanced_today = False
            risk_off = False
            risk_reason = "carry_forward"

            if is_rebalance_day(strategy_day_counter - 1, rebalance_every_n_days):
                ranked_trailing = get_trailing_returns(
                    prices=prices,
                    end_idx_exclusive=i,
                    lookback_days=bot.lookback_days,
                    symbols=proxy_symbols,
                )

                benchmark_trailing = get_trailing_returns(
                    prices=prices,
                    end_idx_exclusive=i,
                    lookback_days=bot.lookback_days,
                    symbols=benchmark_symbols,
                )

                cash_trailing = get_trailing_returns(
                    prices=prices,
                    end_idx_exclusive=i,
                    lookback_days=bot.lookback_days,
                    symbols=cash_symbols,
                )

                selected_symbols, weights, risk_off, risk_reason = choose_holdings_for_day(
                    bot=bot,
                    ranked_trailing=ranked_trailing,
                    benchmark_trailing=benchmark_trailing,
                    cash_trailing=cash_trailing,
                )

                if risk_off:
                    risk_off_days += 1

                _, new_holdings = compute_day_return_and_holdings(
                    day_returns=returns.loc[trading_ts],
                    selected_symbols=selected_symbols,
                    weights=weights,
                )

                current_target_holdings = new_holdings
                rebalanced_today = True
            else:
                # Carry forward prior holdings if not a rebalance day
                current_target_holdings = prev_holdings.copy()

            day_ret_vec = returns.loc[trading_ts]
            gross_ret = 0.0
            for sym, w in current_target_holdings.items():
                if sym in day_ret_vec.index and pd.notna(day_ret_vec[sym]):
                    gross_ret += float(w) * float(day_ret_vec[sym])

            turnover = compute_turnover(prev_holdings, current_target_holdings) if rebalanced_today else 0.0
            cost_drag = compute_cost_drag(
                turnover=turnover,
                transaction_cost_bps=bot.transaction_cost_bps,
                slippage_bps=bot.slippage_bps,
            )
            net_ret = gross_ret - cost_drag

            equity *= (1.0 + net_ret)

            holdings_json = build_holdings_payload(
                holdings=current_target_holdings,
                bot=bot,
                selected_count=len(current_target_holdings),
                risk_off=risk_off,
                risk_reason=risk_reason,
                rebalanced_today=rebalanced_today,
                gross_ret=gross_ret,
                net_ret=net_ret,
                turnover=turnover,
                cost_drag=cost_drag,
            )

            rows_to_upsert.append(
                BackfillRow(
                    bot_id=bot.bot_id,
                    d=trading_day,
                    equity=float(equity),
                    ret=float(net_ret),
                    holdings_json=holdings_json,
                )
            )

            prev_holdings = current_target_holdings.copy()
            processed_days += 1

        except Exception as exc:
            errors += 1
            logger.exception(
                "Error processing bot=%s date=%s: %s",
                bot.bot_id,
                trading_day,
                exc,
            )

    upsert_backfill_rows(rows_to_upsert)

    final_return_pct = (equity - 1.0) * 100.0
    logger.info(
        "Completed bot=%s processed_days=%d risk_off_days=%d final_equity=%.6f final_return_pct=%.2f errors=%d",
        bot.bot_id,
        processed_days,
        risk_off_days,
        equity,
        final_return_pct,
        errors,
    )

    return {
        "bot_id": bot.bot_id,
        "processed_days": processed_days,
        "risk_off_days": risk_off_days,
        "final_equity": equity,
        "final_return_pct": final_return_pct,
        "errors": errors,
        "skipped": False,
    }


# ============================================================
# Main
# ============================================================

def backfill_alpha2() -> None:
    logger.info("Loading active Alpha2 bots from trading.alpha2")
    bots = fetch_active_alpha2_bots()

    if not bots:
        logger.warning("No active bots found in trading.alpha2")
        return

    logger.info(
        "Found %d active bot(s). Backfilling from %s to %s",
        len(bots),
        START_DATE,
        END_DATE,
    )

    results: list[dict[str, Any]] = []

    for bot in bots:
        try:
            results.append(backfill_single_bot(bot))
        except Exception as exc:
            logger.exception("Fatal error in bot=%s: %s", bot.bot_id, exc)
            results.append(
                {
                    "bot_id": bot.bot_id,
                    "processed_days": 0,
                    "risk_off_days": 0,
                    "final_equity": None,
                    "final_return_pct": None,
                    "errors": 1,
                    "skipped": True,
                }
            )

    logger.info("========== Alpha2 Backfill Summary ==========")
    for r in results:
        if r.get("skipped"):
            logger.info(
                "bot=%s status=SKIPPED/FAILED processed_days=%s errors=%s",
                r["bot_id"],
                r.get("processed_days"),
                r.get("errors"),
            )
        else:
            logger.info(
                "bot=%s processed_days=%d risk_off_days=%d final_equity=%.6f final_return_pct=%.2f errors=%d",
                r["bot_id"],
                r["processed_days"],
                r["risk_off_days"],
                r["final_equity"],
                r["final_return_pct"],
                r["errors"],
            )

    logger.info("Alpha2 backfill complete.")


if __name__ == "__main__":
    backfill_alpha2()