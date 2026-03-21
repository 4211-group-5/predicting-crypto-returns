import requests
import pandas as pd
import time

COIN_SYMBOL_MAP = {
    "BTC": "BTC",
    "ETH": "ETH",
    "XRP": "XRP",
    "LTC": "LTC",
    "BCH": "BCH",
    "EOS": "EOS",
    "ETC": "ETC",
    "DASH": "DASH",
    "IOTA": "MIOTA",  
    "OMG": "OMG",
    "XMR": "XMR",
    "ZEC": "ZEC"
}

START_DATE = "2017-01-01"
END_DATE = "2025-12-31"


def to_unix(date_str):
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


def fetch_cryptocompare_daily(api_symbol, project_coin, tsym="USD", start_date=START_DATE, end_date=END_DATE):
    url = "https://min-api.cryptocompare.com/data/v2/histoday"

    end_ts = to_unix(end_date)
    all_chunks = []

    while True:
        params = {
            "fsym": api_symbol,
            "tsym": tsym,
            "limit": 2000,
            "toTs": end_ts
        }

        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if data["Response"] != "Success":
            raise ValueError(f"Error for {project_coin} (API symbol {api_symbol}): {data}")

        rows = data["Data"]["Data"]
        if not rows:
            break

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["time"], unit="s")
        df["coin"] = project_coin   # keep your dataset label as IOTA

        df = df.rename(columns={
            "volumefrom": "volume_coin",
            "volumeto": "volume_usd"
        })

        df = df[["coin", "date", "open", "high", "low", "close", "volume_coin", "volume_usd"]]
        all_chunks.append(df)

        oldest_time = df["date"].min()
        if oldest_time <= pd.Timestamp(start_date):
            break

        end_ts = int((oldest_time - pd.Timedelta(days=1)).timestamp())
        time.sleep(0.2)

    final_df = pd.concat(all_chunks, ignore_index=True).drop_duplicates(subset=["coin", "date"])
    final_df = final_df[
        (final_df["date"] >= pd.Timestamp(start_date)) &
        (final_df["date"] <= pd.Timestamp(end_date))
    ].sort_values(["coin", "date"]).reset_index(drop=True)

    return final_df


all_daily = []

for project_coin, api_symbol in COIN_SYMBOL_MAP.items():
    print(f"Downloading {project_coin} using API symbol {api_symbol}...")
    df_coin = fetch_cryptocompare_daily(api_symbol=api_symbol, project_coin=project_coin)
    all_daily.append(df_coin)

daily_ohlcv = pd.concat(all_daily, ignore_index=True)
daily_ohlcv.to_csv("daily_ohlcv.csv", index=False)

print(daily_ohlcv.head())
print(daily_ohlcv.shape)
