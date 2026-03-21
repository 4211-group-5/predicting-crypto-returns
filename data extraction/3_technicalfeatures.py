import pandas as pd
import numpy as np

weekly_ohlcv = pd.read_csv("weekly_ohlcv.csv", parse_dates=["week"])


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal

    return macd, macd_signal, macd_hist


feature_list = []

for coin, group in weekly_ohlcv.groupby("coin"):
    group = group.sort_values("week").copy()

    # returns
    group["ret_1w"] = group["close"].pct_change()
    group["log_ret_1w"] = np.log(group["close"]).diff()

    # momentum / reversal
    group["mom_2w"] = (1 + group["ret_1w"]).rolling(2).apply(np.prod, raw=True) - 1
    group["mom_4w"] = (1 + group["ret_1w"]).rolling(4).apply(np.prod, raw=True) - 1
    group["mom_12w"] = (1 + group["ret_1w"]).rolling(12).apply(np.prod, raw=True) - 1
    group["reversal_1w_vs_4w"] = group["ret_1w"] - group["ret_1w"].rolling(4).mean()

    # technical indicators
    group["rsi_14"] = compute_rsi(group["close"], 14)

    group["ma_4"] = group["close"].rolling(4).mean()
    group["ma_12"] = group["close"].rolling(12).mean()
    group["ma_ratio_4_12"] = group["ma_4"] / group["ma_12"]

    group["volatility_4w"] = group["ret_1w"].rolling(4).std()
    group["volatility_12w"] = group["ret_1w"].rolling(12).std()

    macd, macd_signal, macd_hist = compute_macd(group["close"])
    group["macd"] = macd
    group["macd_signal"] = macd_signal
    group["macd_hist"] = macd_hist

    group["hl_spread"] = (group["high"] - group["low"]) / group["close"]

    # volume / activity
    group["log_volume_usd"] = np.log(group["volume_usd"].replace(0, np.nan))
    group["volume_usd_growth"] = group["volume_usd"].pct_change()
    group["abnormal_volume_4w"] = group["volume_usd"] / group["volume_usd"].rolling(4).mean()
    group["abnormal_volume_12w"] = group["volume_usd"] / group["volume_usd"].rolling(12).mean()

    # illiquidity
    group["illiquidity_amihud"] = group["ret_1w"].abs() / group["volume_usd"].replace(0, np.nan)

    feature_list.append(group)

weekly_features = pd.concat(feature_list, ignore_index=True)
weekly_features = weekly_features.sort_values(["coin", "week"]).reset_index(drop=True)

weekly_features.to_csv("weekly_features.csv", index=False)

print(weekly_features.head())
print(weekly_features.shape)