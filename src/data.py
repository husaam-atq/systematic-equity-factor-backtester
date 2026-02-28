from __future__ import annotations
import pandas as pd
import yfinance as yf

def download_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("tickers list is empty")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        closes = []
        for t in tickers:
            if (t, "Close") in data.columns:
                closes.append(data[(t, "Close")].rename(t))
            elif (t, "Adj Close") in data.columns:
                closes.append(data[(t, "Adj Close")].rename(t))
            else:
                raise KeyError(f"Could not find Close for ticker {t}")
        prices = pd.concat(closes, axis=1)
    else:
        col = "Close" if "Close" in data.columns else "Adj Close"
        prices = data[col].to_frame(name=tickers[0])

    prices = prices.sort_index().dropna(how="all")
    return prices

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()
