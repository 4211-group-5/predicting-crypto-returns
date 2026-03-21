import pandas as pd

daily_ohlcv = pd.read_csv("daily_ohlcv.csv", parse_dates=["date"])

weekly_list = []

for coin, group in daily_ohlcv.groupby("coin"):
    group = group.sort_values("date").set_index("date")

    weekly = pd.DataFrame()
    weekly["open"] = group["open"].resample("W-SUN").first()
    weekly["high"] = group["high"].resample("W-SUN").max()
    weekly["low"] = group["low"].resample("W-SUN").min()
    weekly["close"] = group["close"].resample("W-SUN").last()
    weekly["volume_coin"] = group["volume_coin"].resample("W-SUN").sum()
    weekly["volume_usd"] = group["volume_usd"].resample("W-SUN").sum()

    weekly = weekly.reset_index()
    weekly["coin"] = coin
    weekly_list.append(weekly)

weekly_ohlcv = pd.concat(weekly_list, ignore_index=True)
weekly_ohlcv = weekly_ohlcv[["coin", "date", "open", "high", "low", "close", "volume_coin", "volume_usd"]]
weekly_ohlcv = weekly_ohlcv.rename(columns={"date": "week"})


weekly_ohlcv = weekly_ohlcv[
    (weekly_ohlcv["open"] > 0) &
    (weekly_ohlcv["high"] > 0) &
    (weekly_ohlcv["low"] > 0) &
    (weekly_ohlcv["close"] > 0)
].copy()

weekly_ohlcv = weekly_ohlcv.sort_values(["coin", "week"]).reset_index(drop=True)

weekly_ohlcv.to_csv("weekly_ohlcv.csv", index=False)

print(weekly_ohlcv.head())
print(weekly_ohlcv.shape)