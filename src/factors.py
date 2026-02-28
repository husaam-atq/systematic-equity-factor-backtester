from __future__ import annotations
import numpy as np
import pandas as pd

def zscore_cross_sectional(x: pd.Series) -> pd.Series:
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd

def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    r_12m = prices.pct_change(252)
    r_1m = prices.pct_change(21)
    return r_12m - r_1m

def short_term_reversal_1w(prices: pd.DataFrame) -> pd.DataFrame:
    return -prices.pct_change(5)

def low_volatility(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    rets = prices.pct_change()
    vol = rets.rolling(window).std()
    return -vol

def build_signal(factor_raw: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    sig = factor_raw.copy()
    if normalize:
        sig = sig.apply(zscore_cross_sectional, axis=1)
    return sig
