# Systematic Equity Factor Backtester

A clean, extensible research project for systematic equity / quantamental roles.

## Features
- Downloads adjusted prices via Yahoo Finance (yfinance)
- Builds cross-sectional factor signals
- Forms long/short portfolios (top/bottom quantiles)
- Backtests with transaction costs + turnover
- Outputs CSV/JSON + equity curve plot

## Quickstart

Install:
pip install -r requirements.txt

Run:
python -m src.cli --factor mom --start 2015-01-01 --tc_bps 5
python -m src.cli --factor lowvol --start 2015-01-01 --tc_bps 5
python -m src.cli --factor reversal --start 2015-01-01 --tc_bps 5

Outputs are saved in outputs/.

## Roadmap
- Add monthly rebalancing / sector neutralization
- Add walk-forward validation
- Add accounting-based factors (earnings quality, accruals, ROIC)
