from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

def plot_equity_curve(returns: pd.Series, title: str, path: str | None = None) -> None:
    r = returns.fillna(0.0)
    equity = (1 + r).cumprod()

    plt.figure()
    plt.plot(equity.index, equity.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (Start=1.0)")
    plt.tight_layout()

    if path:
        plt.savefig(path, dpi=150)
    plt.show()
