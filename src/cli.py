from __future__ import annotations
import argparse
import json
import os
import pandas as pd

from src.data import download_prices, compute_returns
from src.factors import momentum_12_1, low_volatility, short_term_reversal_1w, build_signal
from src.backtest import make_long_short_weights, backtest_daily
from src.metrics import summary_table
from src.plots import plot_equity_curve

DEFAULT_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","JNJ","PG",
    "XOM","CVX","KO","PEP","WMT","HD","UNH","MRK","ABBV","COST",
    "V","MA","CRM","ADBE","NFLX","DIS","INTC","CSCO","ORCL","BAC"
]

def run(factor_name: str, start: str, tc_bps: float, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    prices = download_prices(DEFAULT_TICKERS, start=start)
    rets = compute_returns(prices).dropna(how="all")

    if factor_name == "mom":
        raw = momentum_12_1(prices)
    elif factor_name == "lowvol":
        raw = low_volatility(prices, window=63)
    elif factor_name == "reversal":
        raw = short_term_reversal_1w(prices)
    else:
        raise ValueError("factor_name must be one of: mom, lowvol, reversal")

    signal = build_signal(raw, normalize=True)
    signal = signal.reindex(rets.index).dropna(how="all")

    weights = make_long_short_weights(signal, long_quantile=0.8, short_quantile=0.2, dollar_neutral=True)
    bt = backtest_daily(rets, weights, lag=1, tc_bps=tc_bps)

    stats = summary_table(bt["net"])
    print("\n=== Backtest Summary (NET) ===")
    print(stats.to_string(float_format=lambda x: f"{x:0.4f}"))

    bt_path = os.path.join(outdir, f"backtest_{factor_name}.csv")
    stats_path = os.path.join(outdir, f"stats_{factor_name}.json")
    plot_path = os.path.join(outdir, f"equity_{factor_name}.png")

    bt.to_csv(bt_path, index=True)
    with open(stats_path, "w") as f:
        json.dump({k: float(v) if pd.notnull(v) else None for k, v in stats.items()}, f, indent=2)

    plot_equity_curve(bt["net"], title=f"Equity Curve (Net) - {factor_name}", path=plot_path)

    print(f"\nSaved: {bt_path}")
    print(f"Saved: {stats_path}")
    print(f"Saved: {plot_path}")

def main():
    p = argparse.ArgumentParser(description="Systematic Equity Factor Backtester")
    p.add_argument("--factor", choices=["mom","lowvol","reversal"], default="mom")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--tc_bps", type=float, default=5.0)
    p.add_argument("--outdir", default="outputs")
    args = p.parse_args()

    run(args.factor, args.start, args.tc_bps, args.outdir)

if __name__ == "__main__":
    main()
