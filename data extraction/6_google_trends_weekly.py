from pytrends.request import TrendReq
import pandas as pd
import time

pytrends = TrendReq(hl="en-US", tz=360)

keywords = [
    "cryptocurrency",
    "Bitcoin",
    "Ethereum",
    "XRP",
    "Litecoin",
    "Bitcoin Cash",
    "EOS",
    "Ethereum Classic",
    "Dash",
    "IOTA",
    "OMG Network",
    "Monero",
    "Zcash"
]

windows = [
    ("2017-01-01", "2018-12-31"),
    ("2018-07-01", "2020-06-30"),
    ("2020-01-01", "2021-12-31"),
    ("2021-07-01", "2023-06-30"),
    ("2023-01-01", "2025-12-31"),
]


def fetch_keyword_trends_weekly(keyword, windows, sleep_sec=4):
    all_parts = []

    for start_date, end_date in windows:
        print(f"Downloading {keyword}: {start_date} to {end_date}")

        pytrends.build_payload(
            kw_list=[keyword],
            cat=0,
            timeframe=f"{start_date} {end_date}",
            geo="",
            gprop=""
        )

        df = pytrends.interest_over_time().reset_index()

        if df.empty:
            print(f"No data returned for {keyword} in {start_date} to {end_date}")
            continue

        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])

        df = df.rename(columns={
            "date": "week",
            keyword: f"gt_{keyword.lower().replace(' ', '_')}"
        })

        all_parts.append(df)
        time.sleep(sleep_sec)

    if not all_parts:
        return pd.DataFrame(columns=["week", f"gt_{keyword.lower().replace(' ', '_')}"])

    combined = pd.concat(all_parts, ignore_index=True)
    combined = combined.sort_values("week")
    combined = combined.drop_duplicates(subset=["week"], keep="last")

    combined["week"] = pd.to_datetime(combined["week"])
    combined = combined.set_index("week").resample("W-SUN").mean().reset_index()

    return combined


all_trends = []

for kw in keywords:
    df_kw = fetch_keyword_trends_weekly(kw, windows, sleep_sec=2)
    all_trends.append(df_kw)

google_trends = all_trends[0]
for df in all_trends[1:]:
    google_trends = google_trends.merge(df, on="week", how="outer")

google_trends = google_trends.sort_values("week").reset_index(drop=True)

google_trends = google_trends[
    (google_trends["week"] >= "2017-01-01") &
    (google_trends["week"] <= "2025-12-31")
]

google_trends.to_csv("datasets/6_google_trends_weekly.csv", index=False)

print(google_trends.head())
print(google_trends.shape)