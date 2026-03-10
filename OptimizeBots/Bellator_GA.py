from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# Configuration
# ============================================================

START_DATE = "2018-01-01"
END_DATE = date.today().isoformat()

BENCHMARK = "VOO"
CASH_EQUIVALENT = "BIL"

TOP_N = 4
LOOKBACK_DAYS = 60

DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 5.0

USE_CASH_EQUIVALENT_FALLBACK = True
REQUIRE_FULL_LOOKBACK_WINDOW = True

# Bellator candidate pool (defense-focused / defense-adjacent)
CANDIDATE_UNIVERSE = [
    "LMT", "RTX", "GD", "NOC", "LHX",
    "LDOS", "BAH", "CACI", "SAIC", "KBR",
    "HII", "AVAV", "KTOS", "MRCY", "VSAT",
    "TXT", "CW", "HWM", "BA", "MOG-A"
]

MIN_UNIVERSE_SIZE = 6
MAX_UNIVERSE_SIZE = 15


# ============================================================
# Data classes
# ============================================================

@dataclass
class StrategyConfig:
    lookback_days: int = LOOKBACK_DAYS
    top_n: int = TOP_N
    benchmark: str | None = BENCHMARK
    cash_equivalent: str | None = CASH_EQUIVALENT

    enable_kill_switch: bool = True
    enable_benchmark_filter: bool = False

    kill_switch_mode: str = "top_negative"         # top_negative | avg_negative | off
    benchmark_mode: str = "benchmark_positive"     # benchmark_positive | benchmark_gt_cash | off

    top_return_threshold: float = 0.0
    benchmark_return_threshold: float = 0.0

    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS


@dataclass
class Chromosome:
    mask: np.ndarray        # binary inclusion vector over CANDIDATE_UNIVERSE
    weights_raw: np.ndarray # raw positive values for top_n rank weights


# ============================================================
# Download helpers
# ============================================================

def download_adjusted_close(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
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
            raise ValueError("Downloaded data missing 'Close'.")
        close_df = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data missing 'Close'.")
        close_df = raw[["Close"]].rename(columns={"Close": symbols[0]})

    close_df = close_df.sort_index()
    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df[~close_df.index.duplicated(keep="last")]
    close_df.columns = [str(c).upper() for c in close_df.columns]
    close_df = close_df.dropna(how="all")
    return close_df


def build_master_price_frame(
    candidate_universe: list[str],
    benchmark: str | None,
    cash_equivalent: str | None,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    symbols = list(dict.fromkeys(candidate_universe + [s for s in [benchmark, cash_equivalent] if s]))
    prices = download_adjusted_close(symbols, start_date, end_date)
    if prices.empty:
        raise ValueError("No price data downloaded.")
    return prices


# ============================================================
# Utility helpers
# ============================================================

def bps_to_decimal(bps: float) -> float:
    return float(bps) / 10_000.0


def compute_turnover(prev_holdings: dict[str, float], new_holdings: dict[str, float]) -> float:
    all_symbols = set(prev_holdings) | set(new_holdings)
    gross_change = sum(abs(new_holdings.get(sym, 0.0) - prev_holdings.get(sym, 0.0)) for sym in all_symbols)
    return gross_change / 2.0


def compute_cost_drag(turnover: float, transaction_cost_bps: float, slippage_bps: float) -> float:
    return turnover * bps_to_decimal(transaction_cost_bps + slippage_bps)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


def calculate_cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if total_days <= 0:
        return 0.0
    years = total_days / 365.25
    if years <= 0:
        return 0.0
    final_equity = float(equity_curve.iloc[-1])
    if final_equity <= 0:
        return -1.0
    return final_equity ** (1.0 / years) - 1.0


def calculate_annualized_volatility(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std(ddof=1) * np.sqrt(252))


def calculate_sharpe(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - rf_daily
    vol = excess.std(ddof=1)
    if vol <= 0:
        return 0.0
    return float((excess.mean() / vol) * np.sqrt(252))


def split_price_frame(
    prices: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = prices.loc[:train_end].copy()
    val = prices.loc[train_end:val_end].copy()
    test = prices.loc[val_end:].copy()

    # Remove duplicated boundary rows if they exist
    if len(val) and len(train) and val.index[0] == train.index[-1]:
        val = val.iloc[1:]
    if len(test) and len(val) and test.index[0] == val.index[-1]:
        test = test.iloc[1:]

    return train, val, test


# ============================================================
# Strategy logic
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

    if symbols is not None:
        trailing = trailing[trailing.index.isin(symbols)]

    return trailing


def evaluate_kill_switch(
    config: StrategyConfig,
    ranked_trailing: pd.Series,
    benchmark_trailing: pd.Series,
    cash_trailing: pd.Series,
) -> tuple[bool, str]:
    reasons: list[str] = []

    if config.enable_kill_switch and config.kill_switch_mode != "off":
        if ranked_trailing.empty:
            reasons.append("kill_switch:no_ranked_candidates")
        elif config.kill_switch_mode == "top_negative":
            top_ret = float(ranked_trailing.max())
            if top_ret <= config.top_return_threshold:
                reasons.append(
                    f"kill_switch:top_ret={top_ret:.6f}<=threshold={config.top_return_threshold:.6f}"
                )
        elif config.kill_switch_mode == "avg_negative":
            avg_ret = float(ranked_trailing.mean())
            if avg_ret <= config.top_return_threshold:
                reasons.append(
                    f"kill_switch:avg_ret={avg_ret:.6f}<=threshold={config.top_return_threshold:.6f}"
                )

    if config.enable_benchmark_filter and config.benchmark_mode != "off" and config.benchmark:
        bench_ret = benchmark_trailing.get(config.benchmark, np.nan)

        if pd.isna(bench_ret):
            reasons.append("benchmark_filter:no_benchmark_data")
        else:
            bench_ret = float(bench_ret)

            if config.benchmark_mode == "benchmark_positive":
                if bench_ret <= config.benchmark_return_threshold:
                    reasons.append(
                        f"benchmark_filter:bench_ret={bench_ret:.6f}<=threshold={config.benchmark_return_threshold:.6f}"
                    )

            elif config.benchmark_mode == "benchmark_gt_cash":
                cash_ret = np.nan
                if config.cash_equivalent:
                    cash_ret = cash_trailing.get(config.cash_equivalent, np.nan)

                if pd.isna(cash_ret):
                    reasons.append("benchmark_filter:no_cash_data")
                else:
                    cash_ret = float(cash_ret)
                    if bench_ret <= cash_ret:
                        reasons.append(f"benchmark_filter:bench_ret={bench_ret:.6f}<=cash_ret={cash_ret:.6f}")

    if reasons:
        return True, "; ".join(reasons)

    return False, "risk_on"


def choose_risk_on_holdings(
    ranked_trailing: pd.Series,
    top_n: int,
    weights: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    if ranked_trailing.empty:
        return [], np.array([], dtype=float)

    ranked = ranked_trailing.sort_values(ascending=False)
    selected = ranked.head(top_n).index.tolist()

    use_weights = weights[:len(selected)].copy()
    if len(use_weights) == 0:
        return [], np.array([], dtype=float)

    use_weights = use_weights / use_weights.sum()
    return selected, use_weights


def choose_holdings_for_day(
    config: StrategyConfig,
    ranked_trailing: pd.Series,
    benchmark_trailing: pd.Series,
    cash_trailing: pd.Series,
    weights: np.ndarray,
) -> tuple[list[str], np.ndarray, bool, str]:
    risk_off, reason = evaluate_kill_switch(
        config=config,
        ranked_trailing=ranked_trailing,
        benchmark_trailing=benchmark_trailing,
        cash_trailing=cash_trailing,
    )

    if risk_off:
        if USE_CASH_EQUIVALENT_FALLBACK and config.cash_equivalent:
            return [config.cash_equivalent], np.array([1.0], dtype=float), True, reason
        return [], np.array([], dtype=float), True, reason

    selected, rank_weights = choose_risk_on_holdings(
        ranked_trailing=ranked_trailing,
        top_n=config.top_n,
        weights=weights,
    )

    if not selected and USE_CASH_EQUIVALENT_FALLBACK and config.cash_equivalent:
        return [config.cash_equivalent], np.array([1.0], dtype=float), True, "fallback:no_selected_symbols"

    return selected, rank_weights, False, reason


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


def evaluate_strategy(
    prices: pd.DataFrame,
    universe: list[str],
    rank_weights: np.ndarray,
    config: StrategyConfig,
    initial_equity: float = 1.0,
) -> dict[str, Any]:
    if prices.empty:
        raise ValueError("Prices are empty.")
    if not universe:
        raise ValueError("Universe is empty.")
    if len(rank_weights) < config.top_n:
        raise ValueError("rank_weights length must be >= top_n.")

    available_cols = set(prices.columns)
    universe = [sym for sym in universe if sym in available_cols]

    if config.benchmark and config.benchmark not in available_cols:
        raise ValueError(f"Benchmark {config.benchmark} not found in prices.")
    if config.cash_equivalent and config.cash_equivalent not in available_cols:
        raise ValueError(f"Cash equivalent {config.cash_equivalent} not found in prices.")

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    equity = initial_equity
    prev_holdings: dict[str, float] = {}

    equity_records: list[tuple[pd.Timestamp, float]] = []
    return_records: list[tuple[pd.Timestamp, float]] = []

    total_turnover = 0.0
    risk_off_days = 0

    ranking_universe = set(universe)
    benchmark_symbols = {config.benchmark} if config.benchmark else set()
    cash_symbols = {config.cash_equivalent} if config.cash_equivalent else set()

    for i, trading_ts in enumerate(prices.index):
        if i == 0:
            equity_records.append((trading_ts, equity))
            return_records.append((trading_ts, 0.0))
            continue

        ranked_trailing = get_trailing_returns(
            prices=prices,
            end_idx_exclusive=i,
            lookback_days=config.lookback_days,
            symbols=ranking_universe,
        )

        benchmark_trailing = get_trailing_returns(
            prices=prices,
            end_idx_exclusive=i,
            lookback_days=config.lookback_days,
            symbols=benchmark_symbols,
        )

        cash_trailing = get_trailing_returns(
            prices=prices,
            end_idx_exclusive=i,
            lookback_days=config.lookback_days,
            symbols=cash_symbols,
        )

        selected_symbols, weights, risk_off, _reason = choose_holdings_for_day(
            config=config,
            ranked_trailing=ranked_trailing,
            benchmark_trailing=benchmark_trailing,
            cash_trailing=cash_trailing,
            weights=rank_weights,
        )

        if risk_off:
            risk_off_days += 1

        day_ret_vec = returns.loc[trading_ts]

        gross_ret, new_holdings = compute_day_return_and_holdings(
            day_returns=day_ret_vec,
            selected_symbols=selected_symbols,
            weights=weights,
        )

        turnover = compute_turnover(prev_holdings, new_holdings)
        cost_drag = compute_cost_drag(
            turnover=turnover,
            transaction_cost_bps=config.transaction_cost_bps,
            slippage_bps=config.slippage_bps,
        )

        net_ret = gross_ret - cost_drag
        equity *= (1.0 + net_ret)

        total_turnover += turnover
        prev_holdings = new_holdings

        equity_records.append((trading_ts, equity))
        return_records.append((trading_ts, net_ret))

    equity_curve = pd.Series(
        data=[v for _, v in equity_records],
        index=[t for t, _ in equity_records],
        name="equity",
    )

    daily_returns = pd.Series(
        data=[v for _, v in return_records],
        index=[t for t, _ in return_records],
        name="ret",
    )

    cagr = calculate_cagr(equity_curve)
    vol = calculate_annualized_volatility(daily_returns)
    sharpe = calculate_sharpe(daily_returns)
    max_dd = calculate_max_drawdown(equity_curve)
    avg_turnover = total_turnover / max(1, len(prices) - 1)

    return {
        "final_equity": float(equity_curve.iloc[-1]),
        "total_return": float(equity_curve.iloc[-1] - 1.0),
        "cagr": cagr,
        "annual_return": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "risk_off_days": risk_off_days,
        "avg_turnover": avg_turnover,
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "universe_size": len(universe),
    }


# ============================================================
# Genetic encoding helpers
# ============================================================

def normalize_rank_weights(weights_raw: np.ndarray, top_n: int) -> np.ndarray:
    w = np.maximum(weights_raw[:top_n].astype(float), 1e-8)
    w = np.sort(w)[::-1]   # descending to match rank order
    w = w / w.sum()
    return w


def repair_mask(mask: np.ndarray, min_size: int, max_size: int) -> np.ndarray:
    repaired = mask.copy()

    while repaired.sum() < min_size:
        zero_idx = np.where(repaired == 0)[0]
        repaired[random.choice(zero_idx)] = 1

    while repaired.sum() > max_size:
        one_idx = np.where(repaired == 1)[0]
        repaired[random.choice(one_idx)] = 0

    return repaired


def decode_chromosome(
    chrom: Chromosome,
    candidate_universe: list[str],
    top_n: int,
    min_universe_size: int,
    max_universe_size: int,
) -> tuple[list[str], np.ndarray]:
    mask = repair_mask(chrom.mask, min_universe_size, max_universe_size)
    selected = [sym for sym, bit in zip(candidate_universe, mask) if bit == 1]
    weights = normalize_rank_weights(chrom.weights_raw, top_n)
    return selected, weights


def random_chromosome(
    candidate_universe: list[str],
    top_n: int,
    min_universe_size: int,
    max_universe_size: int,
) -> Chromosome:
    m = len(candidate_universe)
    size = random.randint(min_universe_size, max_universe_size)

    mask = np.zeros(m, dtype=int)
    chosen = random.sample(range(m), size)
    mask[chosen] = 1

    weights_raw = np.random.rand(top_n)
    return Chromosome(mask=mask, weights_raw=weights_raw)


def crossover(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    child_mask = np.array(
        [parent1.mask[i] if random.random() < 0.5 else parent2.mask[i] for i in range(len(parent1.mask))],
        dtype=int,
    )

    alpha = random.random()
    child_weights = alpha * parent1.weights_raw + (1.0 - alpha) * parent2.weights_raw

    return Chromosome(mask=child_mask, weights_raw=child_weights)


def mutate(
    chrom: Chromosome,
    min_universe_size: int,
    max_universe_size: int,
    p_mask: float = 0.08,
    p_weight: float = 0.25,
    weight_sigma: float = 0.15,
) -> Chromosome:
    new_mask = chrom.mask.copy()
    for i in range(len(new_mask)):
        if random.random() < p_mask:
            new_mask[i] = 1 - new_mask[i]

    new_mask = repair_mask(new_mask, min_universe_size, max_universe_size)

    new_weights = chrom.weights_raw.copy()
    for i in range(len(new_weights)):
        if random.random() < p_weight:
            new_weights[i] += np.random.normal(0.0, weight_sigma)

    new_weights = np.maximum(new_weights, 1e-8)

    return Chromosome(mask=new_mask, weights_raw=new_weights)


# ============================================================
# Fitness
# ============================================================

def score_metrics(metrics: dict[str, Any]) -> float:
    """
    You said annual return is primary.
    This keeps CAGR as the main driver, with a modest drawdown penalty.
    """
    cagr = float(metrics["cagr"])
    max_dd = abs(float(metrics["max_drawdown"]))
    return cagr - 0.35 * max_dd


def evaluate_chromosome(
    chrom: Chromosome,
    prices_train: pd.DataFrame,
    prices_val: pd.DataFrame,
    candidate_universe: list[str],
    config: StrategyConfig,
    min_universe_size: int,
    max_universe_size: int,
) -> tuple[float, dict[str, Any]]:
    universe, rank_weights = decode_chromosome(
        chrom=chrom,
        candidate_universe=candidate_universe,
        top_n=config.top_n,
        min_universe_size=min_universe_size,
        max_universe_size=max_universe_size,
    )

    train_metrics = evaluate_strategy(
        prices=prices_train,
        universe=universe,
        rank_weights=rank_weights,
        config=config,
    )

    val_metrics = evaluate_strategy(
        prices=prices_val,
        universe=universe,
        rank_weights=rank_weights,
        config=config,
    )

    fitness = score_metrics(val_metrics)

    payload = {
        "fitness": fitness,
        "universe": universe,
        "rank_weights": rank_weights.tolist(),
        "train": train_metrics,
        "val": val_metrics,
    }
    return fitness, payload


# ============================================================
# Random search baseline
# ============================================================

def run_random_search(
    prices_train: pd.DataFrame,
    prices_val: pd.DataFrame,
    candidate_universe: list[str],
    config: StrategyConfig,
    n_samples: int = 1000,
    min_universe_size: int = MIN_UNIVERSE_SIZE,
    max_universe_size: int = MAX_UNIVERSE_SIZE,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for _ in range(n_samples):
        chrom = random_chromosome(
            candidate_universe=candidate_universe,
            top_n=config.top_n,
            min_universe_size=min_universe_size,
            max_universe_size=max_universe_size,
        )
        fitness, payload = evaluate_chromosome(
            chrom=chrom,
            prices_train=prices_train,
            prices_val=prices_val,
            candidate_universe=candidate_universe,
            config=config,
            min_universe_size=min_universe_size,
            max_universe_size=max_universe_size,
        )
        results.append(payload)

    results.sort(key=lambda x: x["fitness"], reverse=True)
    return results


# ============================================================
# Genetic algorithm
# ============================================================

def tournament_select(scored_population: list[tuple[Chromosome, float, dict[str, Any]]], k: int = 3) -> Chromosome:
    contenders = random.sample(scored_population, k=min(k, len(scored_population)))
    contenders.sort(key=lambda x: x[1], reverse=True)
    return contenders[0][0]


def run_genetic_algorithm(
    prices_train: pd.DataFrame,
    prices_val: pd.DataFrame,
    candidate_universe: list[str],
    config: StrategyConfig,
    population_size: int = 60,
    generations: int = 50,
    elite_size: int = 6,
    min_universe_size: int = MIN_UNIVERSE_SIZE,
    max_universe_size: int = MAX_UNIVERSE_SIZE,
) -> list[dict[str, Any]]:
    population = [
        random_chromosome(
            candidate_universe=candidate_universe,
            top_n=config.top_n,
            min_universe_size=min_universe_size,
            max_universe_size=max_universe_size,
        )
        for _ in range(population_size)
    ]

    best_history: list[dict[str, Any]] = []

    for gen in range(generations):
        scored_population: list[tuple[Chromosome, float, dict[str, Any]]] = []

        for chrom in population:
            fitness, payload = evaluate_chromosome(
                chrom=chrom,
                prices_train=prices_train,
                prices_val=prices_val,
                candidate_universe=candidate_universe,
                config=config,
                min_universe_size=min_universe_size,
                max_universe_size=max_universe_size,
            )
            scored_population.append((chrom, fitness, payload))

        scored_population.sort(key=lambda x: x[1], reverse=True)

        best_chrom, best_fit, best_payload = scored_population[0]
        best_history.append(best_payload)

        print(
            f"Generation {gen+1}/{generations} | "
            f"fitness={best_fit:.4f} | "
            f"val_cagr={best_payload['val']['cagr']:.4f} | "
            f"val_mdd={best_payload['val']['max_drawdown']:.4f} | "
            f"universe_size={len(best_payload['universe'])} | "
            f"weights={np.round(best_payload['rank_weights'], 4)}"
        )

        next_population: list[Chromosome] = []

        elites = [item[0] for item in scored_population[:elite_size]]
        next_population.extend(elites)

        while len(next_population) < population_size:
            p1 = tournament_select(scored_population, k=3)
            p2 = tournament_select(scored_population, k=3)

            child = crossover(p1, p2)
            child = mutate(
                child,
                min_universe_size=min_universe_size,
                max_universe_size=max_universe_size,
                p_mask=0.08,
                p_weight=0.25,
                weight_sigma=0.12,
            )
            next_population.append(child)

        population = next_population

    best_history.sort(key=lambda x: x["fitness"], reverse=True)
    return best_history


# ============================================================
# Final evaluation on held-out test set
# ============================================================

def evaluate_solution_on_test(
    solution: dict[str, Any],
    prices_test: pd.DataFrame,
    config: StrategyConfig,
) -> dict[str, Any]:
    universe = solution["universe"]
    rank_weights = np.array(solution["rank_weights"], dtype=float)

    test_metrics = evaluate_strategy(
        prices=prices_test,
        universe=universe,
        rank_weights=rank_weights,
        config=config,
    )

    return {
        "universe": universe,
        "rank_weights": rank_weights.tolist(),
        "test": test_metrics,
    }


# ============================================================
# Main experiment
# ============================================================

def main():
    random.seed(42)
    np.random.seed(42)

    config = StrategyConfig(
        lookback_days=60,
        top_n=4,
        benchmark="VOO",
        cash_equivalent="BIL",
        enable_kill_switch=True,
        enable_benchmark_filter=False,
        kill_switch_mode="top_negative",
        benchmark_mode="benchmark_positive",
        top_return_threshold=0.0,
        benchmark_return_threshold=0.0,
        transaction_cost_bps=5.0,
        slippage_bps=5.0,
    )

    prices = build_master_price_frame(
        candidate_universe=CANDIDATE_UNIVERSE,
        benchmark=config.benchmark,
        cash_equivalent=config.cash_equivalent,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    train_prices, val_prices, test_prices = split_price_frame(
        prices=prices,
        train_end="2022-12-30",
        val_end="2024-12-31",
    )

    print("Train rows:", len(train_prices))
    print("Val rows:", len(val_prices))
    print("Test rows:", len(test_prices))

    print("\n=== RANDOM SEARCH BASELINE ===")
    baseline_results = run_random_search(
        prices_train=train_prices,
        prices_val=val_prices,
        candidate_universe=CANDIDATE_UNIVERSE,
        config=config,
        n_samples=500,
    )

    best_baseline = baseline_results[0]
    print("\nBest random-search solution:")
    print("Universe:", best_baseline["universe"])
    print("Weights:", np.round(best_baseline["rank_weights"], 4))
    print("Train CAGR:", round(best_baseline["train"]["cagr"], 4))
    print("Val CAGR:", round(best_baseline["val"]["cagr"], 4))
    print("Val MaxDD:", round(best_baseline["val"]["max_drawdown"], 4))
    print("Fitness:", round(best_baseline["fitness"], 4))

    print("\n=== GENETIC ALGORITHM ===")
    ga_results = run_genetic_algorithm(
        prices_train=train_prices,
        prices_val=val_prices,
        candidate_universe=CANDIDATE_UNIVERSE,
        config=config,
        population_size=60,
        generations=40,
        elite_size=6,
    )

    best_ga = ga_results[0]
    print("\nBest GA solution:")
    print("Universe:", best_ga["universe"])
    print("Weights:", np.round(best_ga["rank_weights"], 4))
    print("Train CAGR:", round(best_ga["train"]["cagr"], 4))
    print("Val CAGR:", round(best_ga["val"]["cagr"], 4))
    print("Val MaxDD:", round(best_ga["val"]["max_drawdown"], 4))
    print("Fitness:", round(best_ga["fitness"], 4))

    if len(test_prices) > 10:
        print("\n=== HELD-OUT TEST EVALUATION ===")
        test_eval = evaluate_solution_on_test(
            solution=best_ga,
            prices_test=test_prices,
            config=config,
        )
        print("Test CAGR:", round(test_eval["test"]["cagr"], 4))
        print("Test MaxDD:", round(test_eval["test"]["max_drawdown"], 4))
        print("Test Sharpe:", round(test_eval["test"]["sharpe"], 4))
        print("Test Final Equity:", round(test_eval["test"]["final_equity"], 4))


if __name__ == "__main__":
    main()