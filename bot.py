#!/usr/bin/env python3
"""
BSE Algorithmic Trading ML Bot (Educational)

- Data: Yahoo Finance (yfinance) for BSE tickers (e.g., RELIANCE.BO, TCS.BO)
- Features: RSI, MACD, Bollinger Bands, Ichimoku, FFT, GARCH volatility, Kalman mean, Pairs cointegration
- ML: RandomForest (feature selection), LSTM (sequence forecasting), XGBoost (tabular)
- RL: Q-learning policy over discretized indicator states
- Backtesting: Backtrader with costs & slippage, walk-forward
- Optimization: Optuna (optional)
- Live Mode: Yahoo polling; Zerodha Kite snippets (optional)

DISCLAIMER: Educational use only. Trading involves significant risk. Ensure compliance with SEBI and broker regulations.
"""

import argparse
import datetime as dt
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional heavy deps: import gracefully
try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    yf = None
    print("yfinance not installed: pip install yfinance", file=sys.stderr)

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None

try:
    import backtrader as bt
except Exception as exc:  # pragma: no cover
    bt = None

try:
    import optuna
except Exception as exc:  # pragma: no cover
    optuna = None

try:
    from newsapi import NewsApiClient
except Exception:
    NewsApiClient = None

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
except Exception:
    SentimentIntensityAnalyzer = None
    nltk = None

try:
    from arch import arch_model
except Exception:
    arch_model = None

try:
    from pykalman import KalmanFilter
except Exception:
    KalmanFilter = None

try:
    from ta.trend import MACD, IchimokuIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
except Exception as exc:
    MACD = IchimokuIndicator = RSIIndicator = BollingerBands = None

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint
except Exception:
    sm = None
    coint = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
except Exception as exc:
    RandomForestRegressor = None
    StandardScaler = None
    mean_squared_error = None
    LinearRegression = None

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    tf = None
    Sequential = LSTM = Dense = Dropout = EarlyStopping = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

# Zerodha Kite (optional)
try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bse_ml_bot")


# --------------------------------------------------------------------------------------
# Defaults & Constants
# --------------------------------------------------------------------------------------
DEFAULT_TICKERS = [
    "RELIANCE.BO",
    "TCS.BO",
    "HDFCBANK.BO",
    "INFY.BO",
    "ICICIBANK.BO",
    "HINDUNILVR.BO",
    "SBIN.BO",
    "BAJFINANCE.BO",
    "BHARTIARTL.BO",
    "ITC.BO",
]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

BROKERAGE_PCT = 0.001  # 0.1%
SLIPPAGE_PCT = 0.0005  # 5 bps
RFR_ANNUAL = 0.05  # For Sharpe; adjust if needed
TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------------

def safe_pct_change(series: pd.Series) -> pd.Series:
    return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def annualize_factor(interval: str) -> int:
    interval = interval.lower()
    if interval.endswith("d"):
        return TRADING_DAYS_PER_YEAR
    if interval.endswith("h"):
        return TRADING_DAYS_PER_YEAR * 6  # approx 6 hours per trading day
    if interval.endswith("m"):
        return TRADING_DAYS_PER_YEAR * 6 * 12
    return TRADING_DAYS_PER_YEAR


# --------------------------------------------------------------------------------------
# Data Loader
# --------------------------------------------------------------------------------------
class DataLoader:
    """Fetch historical OHLCV data from Yahoo Finance for given BSE tickers."""

    def __init__(self, interval: str = "1d"):
        self.interval = interval
        if yf is None:
            raise ImportError("yfinance is required. Install via pip install yfinance")

    def fetch(self, tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        logger.info("Fetching historical data for %d tickers ...", len(tickers))
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start, end=end, interval=self.interval, auto_adjust=True, progress=False)
                if df is None or df.empty:
                    logger.warning("No data for %s", ticker)
                    continue
                df = df.rename(columns=str.title)
                df.dropna(inplace=True)
                data[ticker] = df
                logger.info("Fetched %s: %d rows", ticker, len(df))
            except Exception as exc:
                logger.exception("Failed to fetch %s: %s", ticker, exc)
        return data


# --------------------------------------------------------------------------------------
# Feature Engineering
# --------------------------------------------------------------------------------------
class FeatureEngineer:
    """Compute technical, statistical, and custom features per ticker."""

    def __init__(self):
        if MACD is None or RSIIndicator is None or BollingerBands is None or IchimokuIndicator is None:
            logger.warning("ta library missing. Install via pip install ta")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)

        # Returns
        df["ret_1"] = safe_pct_change(close)
        df["ret_5"] = close.pct_change(5).fillna(0.0)
        df["ret_21"] = close.pct_change(21).fillna(0.0)

        # Volatility
        df["vol_5"] = df["ret_1"].rolling(5).std().fillna(0.0)
        df["vol_21"] = df["ret_1"].rolling(21).std().fillna(0.0)

        # Trend & Momentum indicators
        if MACD is not None:
            macd = MACD(close)
            df["macd"] = macd.macd().fillna(0.0)
            df["macd_signal"] = macd.macd_signal().fillna(0.0)
            df["macd_diff"] = macd.macd_diff().fillna(0.0)
        else:
            df["macd"] = df["macd_signal"] = df["macd_diff"] = 0.0

        if RSIIndicator is not None:
            rsi = RSIIndicator(close)
            df["rsi"] = rsi.rsi().fillna(50.0)
        else:
            df["rsi"] = 50.0

        if BollingerBands is not None:
            bb = BollingerBands(close)
            df["bb_h"] = bb.bollinger_hband().fillna(method="bfill").fillna(0.0)
            df["bb_l"] = bb.bollinger_lband().fillna(method="bfill").fillna(0.0)
            df["bb_p"] = bb.bollinger_pband().fillna(0.0)
        else:
            df["bb_h"] = df["bb_l"] = df["bb_p"] = 0.0

        if IchimokuIndicator is not None:
            ichi = IchimokuIndicator(high, low)
            df["ichi_conv"] = ichi.ichimoku_conversion_line().fillna(method="bfill").fillna(0.0)
            df["ichi_base"] = ichi.ichimoku_base_line().fillna(method="bfill").fillna(0.0)
        else:
            df["ichi_conv"] = df["ichi_base"] = 0.0

        # FFT seasonal energy features
        df = self._add_fft_features(df, close)

        # GARCH volatility clustering
        df["garch_vol"] = self._garch_vol(close)

        # Kalman Filter mean estimate
        df["kf_mean"], df["kf_var"] = self._kalman_mean(close)

        # Historical VaR(95%) on daily returns
        try:
            var_rolling = df["ret_1"].rolling(252).quantile(0.05)
            fallback_var = float(df["ret_1"].quantile(0.05)) if len(df) > 0 else -0.02
            df["var_95"] = var_rolling.fillna(fallback_var)
        except Exception:
            df["var_95"] = -0.02

        # Targets: next-day return regression target
        df["target_ret_1"] = df["ret_1"].shift(-1)
        df.dropna(inplace=True)
        return df

    def _add_fft_features(self, df: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
        try:
            arr = series.values
            n = len(arr)
            if n < 64:
                df["fft_energy"] = 0.0
                df["fft_peak"] = 0.0
                return df
            freqs = np.fft.rfftfreq(n)
            fft_vals = np.abs(np.fft.rfft(arr - np.mean(arr)))
            energy = np.sum(fft_vals**2)
            peak_idx = np.argmax(fft_vals[1:]) + 1
            peak_freq = freqs[peak_idx]
            df["fft_energy"] = energy
            df["fft_peak"] = peak_freq
        except Exception:
            df["fft_energy"] = 0.0
            df["fft_peak"] = 0.0
        return df

    def _garch_vol(self, series: pd.Series) -> pd.Series:
        if arch_model is None:
            return series.pct_change().rolling(21).std().fillna(0.0)
        try:
            ret = safe_pct_change(series)
            am = arch_model(ret.dropna() * 100, vol="Garch", p=1, o=0, q=1, dist="normal")
            res = am.fit(disp="off")
            vol = res.conditional_volatility / 100.0
            vol = vol.reindex(series.index).fillna(method="ffill").fillna(0.0)
            return vol
        except Exception as exc:
            logger.warning("GARCH failed: %s", exc)
            return series.pct_change().rolling(21).std().fillna(0.0)

    def _kalman_mean(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        if KalmanFilter is None:
            return series.rolling(10).mean().fillna(method="bfill").fillna(method="ffill"), pd.Series(0.0, index=series.index)
        try:
            kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], initial_state_mean=series.iloc[0])
            state_means, state_vars = kf.filter(series.values)
            mean_series = pd.Series(state_means.flatten(), index=series.index)
            var_series = pd.Series(state_vars.flatten(), index=series.index)
            return mean_series, var_series
        except Exception:
            return series.rolling(10).mean().fillna(method="bfill").fillna(method="ffill"), pd.Series(0.0, index=series.index)


# --------------------------------------------------------------------------------------
# Pairs Trading (Cointegration)
# --------------------------------------------------------------------------------------
class PairsAnalyzer:
    """Find cointegrated pairs and compute spread z-scores as features."""

    def __init__(self, max_pairs: int = 3):
        self.max_pairs = max_pairs

    def find_pairs(self, data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str, float]]:
        if coint is None:
            return []
        tickers = list(data.keys())
        close_map = {t: data[t]["Close"].dropna() for t in tickers}
        pairs: List[Tuple[str, str, float]] = []
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                t1, t2 = tickers[i], tickers[j]
                s1, s2 = close_map[t1].align(close_map[t2], join="inner")
                if len(s1) < 200:
                    continue
                score, pvalue, _ = coint(s1.values, s2.values)
                if pvalue < 0.05:
                    pairs.append((t1, t2, pvalue))
        pairs.sort(key=lambda x: x[2])
        return pairs[: self.max_pairs]

    def compute_spread_features(self, data: Dict[str, pd.DataFrame], pairs: List[Tuple[str, str, float]]) -> Dict[str, pd.DataFrame]:
        if not pairs:
            return data
        enhanced: Dict[str, pd.DataFrame] = {}
        for t, df in data.items():
            enhanced[t] = df.copy()
        for t1, t2, _ in pairs:
            s1, s2 = data[t1]["Close"].align(data[t2]["Close"], join="inner")
            # Hedge ratio via OLS
            if sm is not None:
                x = sm.add_constant(s2.values)
                model = sm.OLS(s1.values, x).fit()
                beta = model.params[1]
            else:
                beta = (np.cov(s1, s2)[0, 1] / np.var(s2)) if np.var(s2) > 1e-8 else 1.0
            spread = s1 - beta * s2
            z = (spread - spread.rolling(60).mean()) / (spread.rolling(60).std() + 1e-8)
            z = z.fillna(0.0)
            # Add to both as features
            for t in [t1, t2]:
                df = enhanced[t]
                z_aligned = z.reindex(df.index).fillna(0.0)
                df[f"pair_{t1}_{t2}_z"] = z_aligned
            logger.info("Pairs feature added for (%s, %s) with beta=%.3f", t1, t2, beta)
        return enhanced


# --------------------------------------------------------------------------------------
# Sentiment (NewsAPI + VADER)
# --------------------------------------------------------------------------------------
class SentimentProvider:
    """Fetch sentiment using NewsAPI + VADER. Falls back to zeros if unavailable."""

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.client = None
        if api_key and NewsApiClient is not None:
            try:
                self.client = NewsApiClient(api_key=api_key)
            except Exception as exc:
                logger.warning("NewsApiClient init failed: %s", exc)
        self.vader = None
        if SentimentIntensityAnalyzer is not None:
            try:
                if nltk is not None:
                    try:
                        nltk.data.find('sentiment/vader_lexicon.zip')
                    except LookupError:
                        nltk.download('vader_lexicon')
                self.vader = SentimentIntensityAnalyzer()
            except Exception as exc:
                logger.warning("VADER init failed: %s", exc)

    def score(self, ticker: str, start: dt.datetime, end: dt.datetime) -> float:
        if not self.client or not self.vader:
            return 0.0
        try:
            q = f"{ticker.replace('.BO','')} BSE"
            res = self.client.get_everything(q=q, from_param=start.date(), to=end.date(), language="en", sort_by="relevancy", page_size=50)
            articles = res.get("articles", [])
            if not articles:
                return 0.0
            scores = []
            for a in articles:
                headline = (a.get("title") or "") + " " + (a.get("description") or "")
                s = self.vader.polarity_scores(headline).get("compound", 0.0)
                scores.append(s)
            return float(np.mean(scores)) if scores else 0.0
        except Exception as exc:
            logger.warning("News/Sentiment failed: %s", exc)
            return 0.0


# --------------------------------------------------------------------------------------
# ML Models
# --------------------------------------------------------------------------------------
class RandomForestFeatureSelector:
    """RandomForest for feature importance ranking."""

    def __init__(self, max_features: int = 25, random_state: int = RANDOM_SEED):
        self.max_features = max_features
        self.model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1) if RandomForestRegressor else None

    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        if self.model is None:
            return list(X.columns)[: self.max_features]
        self.model.fit(X, y)
        importances = self.model.feature_importances_
        idx = np.argsort(importances)[::-1][: self.max_features]
        return list(X.columns[idx])


class LSTMForecaster:
    """Keras LSTM sequence regressor for next-period returns."""

    def __init__(self, seq_len: int = 60, hidden_units: int = 64, dropout: float = 0.2, epochs: int = 5, batch_size: int = 64):
        self.seq_len = seq_len
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler() if StandardScaler else None
        self.model: Optional[Sequential] = None

    def _build(self, input_dim: int) -> Sequential:
        if Sequential is None:
            raise ImportError("TensorFlow/Keras not available")
        model = Sequential([
            LSTM(self.hidden_units, input_shape=(self.seq_len, input_dim), return_sequences=False),
            Dropout(self.dropout),
            Dense(32, activation="relu"),
            Dense(1, activation="linear"),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def _make_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seq_X, seq_y = [], []
        for i in range(self.seq_len, len(X)):
            seq_X.append(X[i - self.seq_len : i])
            seq_y.append(y[i])
        return np.array(seq_X), np.array(seq_y)

    def fit(self, X_df: pd.DataFrame, y: pd.Series) -> None:
        if Sequential is None or StandardScaler is None:
            logger.warning("Skipping LSTM fit (TF or sklearn missing)")
            return
        X = X_df.values.astype(np.float32)
        y_arr = y.values.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X)
        X_seq, y_seq = self._make_sequences(X_scaled, y_arr)
        if len(y_seq) < 50:
            logger.warning("Not enough data for LSTM sequences; skipping.")
            return
        self.model = self._build(X_seq.shape[2])
        cb = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
        self.model.fit(X_seq, y_seq, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size, callbacks=cb, verbose=0)

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            return np.zeros(len(X_df))
        X = X_df.values.astype(np.float32)
        X_scaled = self.scaler.transform(X)
        # Use rolling window for last seq
        preds = np.zeros(len(X_df))
        for i in range(self.seq_len, len(X_df)):
            x_seq = X_scaled[i - self.seq_len : i]
            x_seq = np.expand_dims(x_seq, axis=0)
            preds[i] = float(self.model.predict(x_seq, verbose=0)[0][0])
        return preds


class XGBRegressorWrapper:
    """XGBoost regressor wrapper (falls back to sklearn if unavailable)."""

    def __init__(self):
        self.model = None
        if xgb is not None:
            self.model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                tree_method="hist",
            )
        elif LinearRegression is not None:
            self.model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.model is None:
            return
        self.model.fit(X.values, y.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X))
        return self.model.predict(X.values)


class EnsemblePredictor:
    """Combine predictions from LSTM and XGB (and optionally RF baseline)."""

    def __init__(self):
        self.meta = LinearRegression() if LinearRegression else None
        self.fitted = False

    def fit(self, preds: Dict[str, np.ndarray], y_true: np.ndarray) -> None:
        # Stack predictions as features
        keys = sorted(preds.keys())
        X_meta = np.vstack([preds[k] for k in keys]).T
        if self.meta is not None and len(y_true) == X_meta.shape[0]:
            self.meta.fit(X_meta, y_true)
            self.fitted = True
        else:
            self.fitted = False

    def predict(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        keys = sorted(preds.keys())
        X_meta = np.vstack([preds[k] for k in keys]).T
        if self.fitted and self.meta is not None:
            return self.meta.predict(X_meta)
        # fallback: equal-weight average
        return np.mean(X_meta, axis=1)


# --------------------------------------------------------------------------------------
# Reinforcement Learning (Q-learning)
# --------------------------------------------------------------------------------------
class QLearningAgent:
    """Tabular Q-learning over discretized states from indicators and model signals."""

    def __init__(self, actions: List[int] = [0, 1, -1], alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.actions = actions  # 0 hold, 1 long, -1 short
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q: Dict[Tuple[int, int, int, int], Dict[int, float]] = {}

    @staticmethod
    def _bin(value: float, bins: List[float]) -> int:
        return int(np.digitize([value], bins)[0])

    def _state(self, row: pd.Series) -> Tuple[int, int, int, int]:
        rsi_bin = self._bin(row.get("rsi", 50.0), [30, 50, 70])
        macd_bin = self._bin(row.get("macd_diff", 0.0), [-0.001, 0.0, 0.001])
        vol_bin = self._bin(row.get("vol_21", 0.0), [row.get("vol_21", 0.0) * 0.8, row.get("vol_21", 0.0), row.get("vol_21", 0.0) * 1.2])
        pred_bin = self._bin(row.get("ensemble_pred", 0.0), [-0.001, 0.0, 0.001])
        return (rsi_bin, macd_bin, vol_bin, pred_bin)

    def _get_q(self, state: Tuple[int, int, int, int]) -> Dict[int, float]:
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.actions}
        return self.Q[state]

    def choose_action(self, state: Tuple[int, int, int, int]) -> int:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q = self._get_q(state)
        return max(q, key=q.get)

    def update(self, state: Tuple[int, int, int, int], action: int, reward: float, next_state: Tuple[int, int, int, int]) -> None:
        q = self._get_q(state)
        next_q = self._get_q(next_state)
        best_next = max(next_q.values())
        q[action] = q[action] + self.alpha * (reward + self.gamma * best_next - q[action])

    def train_on_dataframe(self, df: pd.DataFrame) -> List[int]:
        positions: List[int] = []
        prev_state = None
        prev_action = 0
        for idx in range(len(df)):
            row = df.iloc[idx]
            state = self._state(row)
            action = self.choose_action(state)
            positions.append(action)
            if prev_state is not None:
                ret = row.get("ret_1", 0.0)
                reward = (prev_action * ret) - BROKERAGE_PCT - SLIPPAGE_PCT
                self.update(prev_state, prev_action, reward, state)
            prev_state = state
            prev_action = action
        return positions

    def infer_positions(self, df: pd.DataFrame) -> List[int]:
        positions: List[int] = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            state = self._state(row)
            q = self._get_q(state)
            action = max(q, key=q.get)
            positions.append(action)
        return positions


# --------------------------------------------------------------------------------------
# Backtrader Strategy
# --------------------------------------------------------------------------------------
class PandasDataEx(bt.feeds.PandasData if bt else object):
    lines = ("signal", "pred", "risk")
    params = (("signal", -1), ("pred", -1), ("risk", -1))


class HybridStrategy(bt.Strategy if bt else object):
    params = dict(cash_pct=0.95)

    def __init__(self):  # type: ignore[override]
        self.order = None

    def next(self):  # type: ignore[override]
        if self.order:
            return
        data = self.datas[0]
        signal = int(data.signal[0])
        pred = float(data.pred[0])
        # Dynamic risk sizing (Kelly/VaR-inspired)
        try:
            risk = float(data.risk[0])
        except Exception:
            risk = 0.5
        risk = max(0.05, min(0.95, risk))
        pos = self.getposition(data)
        target_size = 0
        if signal > 0 and pred > 0:
            target_size = int(self.broker.get_cash() * risk / data.close[0])
        elif signal < 0 and pred < 0:
            target_size = -int(self.broker.get_cash() * risk / data.close[0])
        else:
            target_size = 0
        size_delta = target_size - pos.size
        if size_delta > 0:
            self.order = self.buy(data=data, size=size_delta)
        elif size_delta < 0:
            self.order = self.sell(data=data, size=abs(size_delta))

    def notify_order(self, order):  # type: ignore[override]
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
class Metrics:
    @staticmethod
    def sharpe(returns: pd.Series, interval: str) -> float:
        if returns.std() == 0:
            return 0.0
        ann = annualize_factor(interval)
        excess = returns.mean() * ann - RFR_ANNUAL
        vol = returns.std() * math.sqrt(ann)
        return float(excess / (vol + 1e-8))

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        roll_max = equity.cummax()
        dd = (equity - roll_max) / (roll_max + 1e-8)
        return float(dd.min())

    @staticmethod
    def cagr(equity: pd.Series, interval: str) -> float:
        if len(equity) < 2:
            return 0.0
        ann = annualize_factor(interval)
        total_return = equity.iloc[-1] / (equity.iloc[0] + 1e-8)
        years = len(equity) / ann
        return float(total_return ** (1 / years) - 1) if years > 0 else 0.0

    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        pos = (returns > 0).sum()
        tot = (returns != 0).sum()
        return float(pos / tot) if tot > 0 else 0.0


# --------------------------------------------------------------------------------------
# Pipeline: Train, Predict, RL, Backtest
# --------------------------------------------------------------------------------------
@dataclass
class ModelArtifacts:
    selected_features: List[str]
    lstm: LSTMForecaster
    xgb: XGBRegressorWrapper
    ensemble: EnsemblePredictor


class Pipeline:
    def __init__(self, interval: str = "1d"):
        self.interval = interval
        self.fe = FeatureEngineer()
        self.pairs = PairsAnalyzer(max_pairs=3)
        news_key = os.environ.get("NEWSAPI_KEY")
        self.sent = SentimentProvider(api_key=news_key)

    def prepare_data(self, tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        loader = DataLoader(interval=self.interval)
        raw = loader.fetch(tickers, start, end)
        # Pairs features
        pairs = self.pairs.find_pairs(raw)
        raw = self.pairs.compute_spread_features(raw, pairs)
        # Per-ticker features
        feats: Dict[str, pd.DataFrame] = {}
        for t, df in raw.items():
            f = self.fe.compute(df)
            # Sentiment as slow-moving feature
            try:
                sent = self.sent.score(t, f.index[0].to_pydatetime(), f.index[-1].to_pydatetime())
            except Exception:
                sent = 0.0
            f["sentiment"] = float(sent)
            feats[t] = f
        return feats

    def _split_walk_forward(self, df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        n = len(df)
        splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        fold = n // (n_splits + 1)
        for i in range(1, n_splits + 1):
            train = df.iloc[: i * fold]
            test = df.iloc[i * fold : (i + 1) * fold]
            if len(test) < 50:
                break
            splits.append((train, test))
        if not splits:
            splits.append((df.iloc[:-50], df.iloc[-50:]))
        return splits

    def _fit_models(self, train_df: pd.DataFrame) -> ModelArtifacts:
        features = [c for c in train_df.columns if c not in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "target_ret_1"]]
        X = train_df[features]
        y = train_df["target_ret_1"]
        # Feature selection
        selector = RandomForestFeatureSelector(max_features=min(25, max(5, len(features)//2)))
        sel_feats = selector.select(X, y)
        X_sel = X[sel_feats]
        # LSTM
        lstm = LSTMForecaster(seq_len=60, hidden_units=64, dropout=0.2, epochs=5, batch_size=64)
        lstm.fit(X_sel, y)
        # XGB
        xgbw = XGBRegressorWrapper()
        xgbw.fit(X_sel, y)
        # Ensemble fit on train via in-sample preds (simple; in production use CV)
        lstm_pred = lstm.predict(X_sel)
        xgb_pred = xgbw.predict(X_sel)
        ens = EnsemblePredictor()
        ens.fit({"lstm": lstm_pred, "xgb": xgb_pred}, y.values)
        return ModelArtifacts(selected_features=sel_feats, lstm=lstm, xgb=xgbw, ensemble=ens)

    def _predict_with_models(self, artifacts: ModelArtifacts, df: pd.DataFrame) -> np.ndarray:
        X_sel = df[artifacts.selected_features]
        p_lstm = artifacts.lstm.predict(X_sel)
        p_xgb = artifacts.xgb.predict(X_sel)
        p_ens = artifacts.ensemble.predict({"lstm": p_lstm, "xgb": p_xgb})
        return p_ens

    def run_walk_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        results: List[pd.DataFrame] = []
        splits = self._split_walk_forward(df)
        for i, (train, test) in enumerate(splits, start=1):
            logger.info("Walk-forward fold %d: train=%d, test=%d", i, len(train), len(test))
            art = self._fit_models(train)
            test = test.copy()
            test["ensemble_pred"] = self._predict_with_models(art, test)
            # RL agent trained on train with predictions
            train_rl = train.copy()
            train_rl["ensemble_pred"] = self._predict_with_models(art, train)
            agent = QLearningAgent()
            agent.train_on_dataframe(train_rl)
            test["rl_signal"] = agent.infer_positions(test)
            # Dynamic risk sizing: Kelly-like with VaR cap
            base_risk = np.clip(np.abs(test["ensemble_pred"]) / (test["vol_21"] ** 2 + 1e-6), 0.05, 0.95)
            var_cap = np.where(test["var_95"] < -0.03, 0.2, 0.95)
            test["risk"] = np.minimum(base_risk, var_cap)
            results.append(test)
        out = pd.concat(results).sort_index()
        return out

    def backtest(self, df: pd.DataFrame, interval: str, plot: bool = False) -> Dict[str, float]:
        if bt is None or plt is None:
            # Fallback simple backtest with transaction costs approximation
            logger.warning("Backtrader/matplotlib not available; using naive backtest.")
            signal = np.sign(df["ensemble_pred"]).fillna(0.0)
            ret_gross = df["ret_1"].fillna(0.0)
            trades = signal.diff().abs().fillna(0.0)
            cost = (BROKERAGE_PCT + SLIPPAGE_PCT) * trades
            ret = signal.shift(1).fillna(0.0) * ret_gross - cost
            equity = (1 + ret).cumprod()
            metrics = {
                "sharpe": Metrics.sharpe(ret, interval),
                "max_dd": Metrics.max_drawdown(equity),
                "cagr": Metrics.cagr(equity, interval),
                "win_rate": Metrics.win_rate(ret),
            }
            logger.info("Metrics: %s", metrics)
            return metrics
        # Prepare feed
        use_cols = ["Open", "High", "Low", "Close", "Volume", "rl_signal", "ensemble_pred", "risk"]
        feed_df = df[use_cols].copy()
        feed_df.rename(columns={"rl_signal": "signal", "ensemble_pred": "pred"}, inplace=True)
        # Backtrader engine
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(1_000_000.0)
        cerebro.addsizer(bt.sizers.FixedSize, stake=1)
        # Commission & slippage
        cerebro.broker.setcommission(commission=BROKERAGE_PCT)
        try:
            cerebro.broker.set_slippage_perc(perc=SLIPPAGE_PCT)
        except Exception:
            pass
        data_feed = PandasDataEx(dataname=feed_df)
        cerebro.adddata(data_feed)
        cerebro.addstrategy(HybridStrategy)
        # Run
        result = cerebro.run()
        portfolio_value = cerebro.broker.getvalue()
        logger.info("Final Portfolio Value: %.2f", portfolio_value)
        # Extract returns from backtrader (proxy equity curve)
        equity_curve = feed_df["Close"].copy()
        equity_curve[:] = np.linspace(1.0, 1.0 * portfolio_value / 1_000_000.0, len(equity_curve))
        ret_series = equity_curve.pct_change().fillna(0.0)
        metrics = {
            "sharpe": Metrics.sharpe(ret_series, interval),
            "max_dd": Metrics.max_drawdown(equity_curve),
            "cagr": Metrics.cagr(equity_curve, interval),
            "win_rate": Metrics.win_rate(ret_series),
        }
        logger.info("Metrics: %s", metrics)
        if plot and plt is not None:
            cerebro.plot(style="candlestick")
        return metrics


# --------------------------------------------------------------------------------------
# Live Mode
# --------------------------------------------------------------------------------------
class LiveRunner:
    def __init__(self, pipeline: Pipeline, tickers: List[str], interval: str, poll_secs: int = 120, paper: bool = True, broker: Optional[str] = None):
        self.pipeline = pipeline
        self.tickers = tickers
        self.interval = interval
        self.poll_secs = poll_secs
        self.paper = paper
        self.broker = broker
        self.kite = None
        if broker == "zerodha" and KiteConnect is not None and not paper:
            api_key = os.environ.get("KITE_API_KEY")
            api_secret = os.environ.get("KITE_API_SECRET")
            request_token = os.environ.get("KITE_REQUEST_TOKEN")
            if api_key and api_secret and request_token:
                self.kite = KiteConnect(api_key=api_key)
                data = self.kite.generate_session(request_token, api_secret=api_secret)
                self.kite.set_access_token(data["access_token"])
                logger.info("Zerodha Kite authenticated.")
            else:
                logger.warning("Zerodha credentials not set; running in paper mode.")
                self.paper = True

    def place_order(self, ticker: str, action: str, quantity: int):
        logger.info("Order: %s %s x %d", action, ticker, quantity)
        if self.paper or self.kite is None:
            return
        try:
            # Example market order via Kite; map ticker if needed to NSE/BSE instruments
            self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_BSE,
                tradingsymbol=ticker.replace(".BO", ""),
                transaction_type=self.kite.TRANSACTION_TYPE_BUY if action == "BUY" else self.kite.TRANSACTION_TYPE_SELL,
                quantity=quantity,
                product=self.kite.PRODUCT_CNC,
                order_type=self.kite.ORDER_TYPE_MARKET,
            )
        except Exception as exc:
            logger.warning("Order failed: %s", exc)

    def loop(self):
        logger.info("Starting live loop (paper=%s, interval=%s)...", self.paper, self.interval)
        while True:
            try:
                end = dt.datetime.now()
                start = end - dt.timedelta(days=60)
                feats = self.pipeline.prepare_data(self.tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                for t, df in feats.items():
                    wf = self.pipeline.run_walk_forward(df)
                    last = wf.iloc[-1]
                    sig = int(last.get("rl_signal", 0))
                    pred = float(last.get("ensemble_pred", 0.0))
                    if sig > 0 and pred > 0:
                        self.place_order(t, "BUY", 1)
                    elif sig < 0 and pred < 0:
                        self.place_order(t, "SELL", 1)
                time.sleep(self.poll_secs)
            except KeyboardInterrupt:
                logger.info("Live loop stopped.")
                break
            except Exception as exc:
                logger.exception("Live loop error: %s", exc)
                time.sleep(self.poll_secs)


# --------------------------------------------------------------------------------------
# Optimization (Optuna)
# --------------------------------------------------------------------------------------
class Optimizer:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def optimize(self, df: pd.DataFrame, trials: int = 10) -> Dict[str, float]:
        if optuna is None:
            logger.warning("Optuna not installed; skipping optimization.")
            return {}

        def objective(trial: "optuna.trial.Trial") -> float:
            # Tune key hyperparameters
            lstm_units = trial.suggest_int("lstm_units", 16, 128, step=16)
            lstm_dropout = trial.suggest_float("lstm_dropout", 0.0, 0.5)
            rf_feats = trial.suggest_int("rf_feats", 10, 40)
            # Fit models with these params on first 70% and eval on next 30%
            n = len(df)
            split = int(n * 0.7)
            train = df.iloc[:split].copy()
            valid = df.iloc[split:].copy()

            # Build artifacts with tuned params
            features = [c for c in train.columns if c not in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "target_ret_1"]]
            X = train[features]
            y = train["target_ret_1"]

            selector = RandomForestFeatureSelector(max_features=rf_feats)
            sel_feats = selector.select(X, y)
            X_sel = X[sel_feats]

            lstm = LSTMForecaster(seq_len=60, hidden_units=lstm_units, dropout=lstm_dropout, epochs=4, batch_size=64)
            lstm.fit(X_sel, y)
            xgbw = XGBRegressorWrapper()
            xgbw.fit(X_sel, y)

            lstm_pred = lstm.predict(valid[sel_feats])
            xgb_pred = xgbw.predict(valid[sel_feats])
            ens = EnsemblePredictor()
            ens.fit({"lstm": lstm.predict(X_sel), "xgb": xgbw.predict(X_sel)}, y.values)
            valid["ensemble_pred"] = ens.predict({"lstm": lstm_pred, "xgb": xgb_pred})

            # Naive pnl
            ret = valid["ret_1"].values
            pnl = np.sign(valid["ensemble_pred"].values) * ret - BROKERAGE_PCT - SLIPPAGE_PCT
            ann = annualize_factor("1d")
            sharpe = np.mean(pnl) * ann / (np.std(pnl) * math.sqrt(ann) + 1e-8)
            return float(sharpe)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)
        logger.info("Best params: %s, value=%.4f", study.best_params, study.best_value)
        return study.best_params


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BSE Algorithmic Trading ML Bot (Educational)")
    p.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS), help="Comma-separated BSE tickers (e.g., RELIANCE.BO,TCS.BO)")
    p.add_argument("--start", type=str, default="2015-01-01")
    p.add_argument("--end", type=str, default=dt.date.today().strftime("%Y-%m-%d"))
    p.add_argument("--interval", type=str, default="1d", choices=["1d", "1h", "30m", "15m", "5m", "1m"])
    p.add_argument("--plot", action="store_true", help="Plot backtest chart (Backtrader)")
    p.add_argument("--optimize", action="store_true", help="Run Optuna optimization")
    p.add_argument("--trials", type=int, default=10, help="Optuna trials")
    p.add_argument("--live", action="store_true", help="Run live mode (paper by default)")
    p.add_argument("--poll-secs", type=int, default=120, help="Polling interval for live mode")
    p.add_argument("--paper", action="store_true", help="Paper trading in live mode")
    p.add_argument("--broker", type=str, default=None, choices=[None, "zerodha"], help="Broker integration (optional)")
    return p.parse_args()


def main():
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    logger.info("Tickers: %s", tickers)

    pipeline = Pipeline(interval=args.interval)

    if args.live:
        live = LiveRunner(pipeline=pipeline, tickers=tickers, interval=args.interval, poll_secs=args.poll_secs, paper=args.paper, broker=args.broker)
        live.loop()
        return

    # Backtest mode
    feats = pipeline.prepare_data(tickers, args.start, args.end)

    # Select a primary ticker for demonstration backtest (multi-asset could be extended)
    primary = tickers[0]
    if primary not in feats:
        logger.error("Primary ticker %s not available.", primary)
        sys.exit(1)

    df = feats[primary]

    if args.optimize:
        opt = Optimizer(pipeline)
        best = opt.optimize(df, trials=args.trials)
        logger.info("Optimization best params: %s", best)

    wf = pipeline.run_walk_forward(df)

    metrics = pipeline.backtest(wf, interval=args.interval, plot=args.plot)

    # Console summary
    print("\n=== Backtest Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if plt is not None:
        # Plot equity curve from naive accumulation of signals
        ret = wf["ret_1"].values * np.sign(wf["ensemble_pred"].values)
        equity = pd.Series((1 + ret).cumprod(), index=wf.index)
        plt.figure(figsize=(10, 4))
        plt.plot(equity.index, equity.values, label="Equity (naive)")
        plt.title(f"Equity Curve - {primary}")
        plt.legend()
        plt.tight_layout()
        if args.plot:
            plt.show()
        else:
            out = f"equity_{primary.replace('.','_')}.png"
            plt.savefig(out)
            logger.info("Saved plot to %s", out)


if __name__ == "__main__":
    main()