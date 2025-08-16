# BSE Algorithmic Trading ML Bot (Educational)

This project provides a modular, end-to-end Python bot for algorithmic trading on BSE-listed equities using public data (Yahoo Finance) and open-source libraries. It includes:

- Data ingestion (Yahoo Finance) for at least 10 BSE tickers
- Feature engineering (MACD, RSI, Bollinger Bands, Ichimoku, GARCH volatility, FFT seasonal features, Kalman Filter mean reversion, cointegration for pairs)
- ML ensemble (RandomForest + LSTM + XGBoost) with Optuna optimization
- Reinforcement Learning (Q-learning) for execution policy
- Backtesting (Backtrader) with transaction costs and slippage
- Walk-forward evaluation and metrics (Sharpe, MaxDD, CAGR, WinRate)
- Optional live mode using Yahoo/AlphaVantage polling and Zerodha Kite snippets for execution

## Quickstart

1) Create and activate a virtual env
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
```

3) Run a backtest (default tickers, daily data)
```bash
python bot.py --start 2015-01-01 --end 2024-12-31 --plot
```

4) Live (paper) mode
```bash
python bot.py --live --interval 5m --poll-secs 120 --paper
```

5) Hyperparameter optimization (small trial count)
```bash
python bot.py --optimize --trials 10
```

## VS Code
- Install VS Code + Python extension
- Open folder, set interpreter to `.venv`
- Use `Run and Debug` or terminal

## Notes
- Educational only. Markets carry risk; comply with SEBI regulations.
- Some components are optional and gracefully degrade if missing (e.g., XGBoost, Optuna).
- TensorFlow can be heavy; CPU works but is slower.
- Alpha Vantage and Zerodha require API keys; set env vars before running live mode.