from __future__ import annotations
import pandas as pd

def make_long_short_weights(
    signal: pd.DataFrame,
    long_quantile: float = 0.8,
    short_quantile: float = 0.2,
    dollar_neutral: bool = True
) -> pd.DataFrame:
    weights = pd.DataFrame(index=signal.index, columns=signal.columns, dtype=float)

    for dt, row in signal.iterrows():
        x = row.dropna()
        if len(x) < 10:
            continue

        q_long = x.quantile(long_quantile)
        q_short = x.quantile(short_quantile)

        longs = x[x >= q_long].index
        shorts = x[x <= q_short].index

        if len(longs) == 0 and len(shorts) == 0:
            continue

        w = pd.Series(0.0, index=row.index)

        if len(longs) > 0:
            w.loc[longs] = 1.0 / len(longs)
        if len(shorts) > 0:
            w.loc[shorts] = -1.0 / len(shorts)

        if dollar_neutral:
            pos = w[w > 0].sum()
            neg = -w[w < 0].sum()
            if pos > 0:
                w[w > 0] = w[w > 0] / pos
            if neg > 0:
                w[w < 0] = w[w < 0] / neg

        weights.loc[dt] = w

    return weights

def backtest_daily(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    lag: int = 1,
    tc_bps: float = 5.0
) -> pd.DataFrame:
    returns, weights = returns.align(weights, join="inner", axis=0)
    w = weights.shift(lag).fillna(0.0)

    gross = (w * returns).sum(axis=1)
    turnover = (w.diff().abs().sum(axis=1)).fillna(0.0)
    costs = (tc_bps / 10000.0) * turnover
    net = gross - costs

    return pd.DataFrame({"gross": gross, "turnover": turnover, "costs": costs, "net": net})
