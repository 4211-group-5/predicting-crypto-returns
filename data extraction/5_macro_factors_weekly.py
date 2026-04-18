from fredapi import Fred
import pandas as pd
import yfinance as yf

FRED_API_KEY = "30886dccfd9d9a5f5239369eb3a732f2"##fred key, need to get a new one from fred website
START_DATE = "2017-01-01"
END_DATE = "2025-12-31"

fred = Fred(api_key=FRED_API_KEY)

def get_fred_series(series_id, name, start_date=START_DATE, end_date=END_DATE):
    s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    df = s.reset_index()
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"])
    df[name] = pd.to_numeric(df[name], errors="coerce")
    return df.sort_values("date")


#daily indicators

daily_series = {
    "sp500": "SP500",
    "vix": "VIXCLS",
    "usdx": "DTWEXBGS",
    "wti": "DCOILWTICO",
}

daily_frames = []

for var_name, series_id in daily_series.items():
    print(f"Downloading {var_name} from FRED: {series_id}")
    df = get_fred_series(series_id, var_name)
    daily_frames.append(df.set_index("date"))

daily_macro = pd.concat(daily_frames, axis=1).reset_index().sort_values("date")


#Gold from yfinance

print("Downloading gold proxy from yfinance: GLD")
gld = yf.download(
    "GLD",
    start=START_DATE,
    end="2026-01-01",
    progress=False,
    auto_adjust=False
)

if gld.empty:
    raise ValueError("No data returned from yfinance for GLD")

if isinstance(gld.columns, pd.MultiIndex):
    if ("Adj Close", "GLD") in gld.columns:
        gold_series = gld[("Adj Close", "GLD")]
    elif ("Close", "GLD") in gld.columns:
        gold_series = gld[("Close", "GLD")]
    else:
        raise ValueError(f"Unexpected GLD columns: {gld.columns}")
else:
    if "Adj Close" in gld.columns:
        gold_series = gld["Adj Close"]
    elif "Close" in gld.columns:
        gold_series = gld["Close"]
    else:
        raise ValueError(f"Unexpected GLD columns: {gld.columns}")

gold = gold_series.reset_index()
gold.columns = ["date", "gold"]
gold["date"] = pd.to_datetime(gold["date"])
gold["gold"] = pd.to_numeric(gold["gold"], errors="coerce")
gold = gold.sort_values("date")

daily_macro = daily_macro.merge(gold, on="date", how="outer").sort_values("date")

# Weekly market variables
macro_weekly_market = (
    daily_macro.set_index("date")
    .resample("W-SUN")
    .last()
    .reset_index()
    .rename(columns={"date": "week"})
)

macro_weekly_market["sp500_ret_1w"] = macro_weekly_market["sp500"].pct_change()
macro_weekly_market["vix_chg_1w"] = macro_weekly_market["vix"].diff()
macro_weekly_market["usdx_ret_1w"] = macro_weekly_market["usdx"].pct_change()
macro_weekly_market["wti_ret_1w"] = macro_weekly_market["wti"].pct_change()
macro_weekly_market["gold_ret_1w"] = macro_weekly_market["gold"].pct_change()


#Monthly indicators

#convert EPU daily -> monthly 
epu_daily = get_fred_series("USEPUINDXD", "epu")
epu_monthly = (
    epu_daily.set_index("date")
    .resample("MS")
    .last()
    .reset_index()
)

teu = get_fred_series("EPUTRADE", "teu")
rfree = get_fred_series("TB3MS", "rfree")
aaa = get_fred_series("AAA", "aaa")
baa = get_fred_series("BAA", "baa")
cpi = get_fred_series("CPIAUCSL", "cpi")

monthly_macro = epu_monthly.merge(teu, on="date", how="outer")
monthly_macro = monthly_macro.merge(rfree, on="date", how="outer")
monthly_macro = monthly_macro.merge(aaa, on="date", how="outer")
monthly_macro = monthly_macro.merge(baa, on="date", how="outer")
monthly_macro = monthly_macro.merge(cpi, on="date", how="outer")
monthly_macro = monthly_macro.sort_values("date")

#fix CPI missing month
monthly_macro["cpi"] = monthly_macro["cpi"].ffill()

#get monthly variables
monthly_macro["dfy"] = monthly_macro["baa"] - monthly_macro["aaa"]
monthly_macro["cpi_inflation_mom"] = monthly_macro["cpi"].pct_change(1)

#create month key
monthly_macro["month"] = monthly_macro["date"].dt.to_period("M")

#keep only latest reading per month
monthly_macro = (
    monthly_macro.sort_values("date")
    .groupby("month", as_index=False)
    .last()
)

#weekly grid and assign month key
weekly_grid = pd.DataFrame({
    "week": pd.date_range(start=START_DATE, end=END_DATE, freq="W-SUN")
})
weekly_grid["month"] = weekly_grid["week"].dt.to_period("M")

#merge monthly indicators by month
macro_weekly_monthly = weekly_grid.merge(
    monthly_macro[[
        "month", "epu", "teu", "rfree", "aaa", "baa", "dfy", "cpi", "cpi_inflation_mom"
    ]],
    on="month",
    how="left"
).drop(columns=["month"])


#TMS

tms = get_fred_series("T10Y3M", "tms")
tms_weekly = (
    tms.set_index("date")
    .resample("W-SUN")
    .last()
    .reset_index()
    .rename(columns={"date": "week"})
)


#merge macro data

macro_weekly = macro_weekly_market.merge(macro_weekly_monthly, on="week", how="outer")
macro_weekly = macro_weekly.merge(tms_weekly, on="week", how="outer")

macro_weekly = macro_weekly.sort_values("week").reset_index(drop=True)

macro_weekly = macro_weekly[
    (macro_weekly["week"] >= pd.Timestamp(START_DATE)) &
    (macro_weekly["week"] <= pd.Timestamp(END_DATE))
].copy()

macro_weekly.to_csv("5_macro_factors_weekly.csv", index=False)

print(macro_weekly.head(15))
print(macro_weekly.shape)
print(macro_weekly.columns.tolist())