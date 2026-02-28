from __future__ import annotations
import numpy as np
import pandas as pd

def annualized_return(r: pd.Series, periods_per_year: int = 252) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    return (1 + r).prod() ** (periods_per_year / len(r)) - 1

def annualized_vol(r: pd.Series, periods_per_year: int = 252) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    return r.std(ddof=0) * np.sqrt(periods_per_year)

def sharpe(r: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    ex = r - rf / periods_per_year
    vol = ex.std(ddof=0)
    if vol == 0:
        return float("nan")
    return ex.mean() / vol * np.sqrt(periods_per_year)

def max_drawdown(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd.min()

def summary_table(r: pd.Series) -> pd.Series:
    return pd.Series({
        "Ann.Return": annualized_return(r),
        "Ann.Vol": annualized_vol(r),
        "Sharpe": sharpe(r),
        "MaxDD": max_drawdown(r),
        "Obs": int(r.dropna().shape[0]),
    })
