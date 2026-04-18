import pandas as pd

COMMON_MODEL_START_DATE = "2018-01-07"

df = pd.read_csv("7_merged_weekly_dataset.csv", parse_dates=["week"])
df = df.sort_values(["coin", "week"]).copy()

target_cols = ["ret_1w", "log_ret_1w"]
id_cols = ["coin", "week"]

coin_specific_to_lag = [
    "open", "high", "low", "close",
    "volume_coin", "volume_usd",
    "rsi_14",
    "ma_4", "ma_12", "ma_ratio_4_12",
    "volatility_4w", "volatility_12w",
    "macd", "macd_signal", "macd_hist",
    "hl_spread",
    "mom_2w", "mom_4w", "mom_12w",
    "reversal_1w_vs_4w",
    "log_volume_usd", "volume_usd_growth",
    "abnormal_volume_4w", "abnormal_volume_12w",
    "illiquidity_amihud"
]

common_to_lag = [
    "market_return_equal", "umd",

    "sp500", "vix", "usdx", "gold", "wti",
    "sp500_ret_1w", "vix_chg_1w", "usdx_ret_1w", "gold_ret_1w", "wti_ret_1w",

    "epu", "teu", "rfree", "aaa", "baa", "dfy", "tms", "cpi", "cpi_inflation_mom"
]

monthly_origin_vars = [
    "epu", "teu", "rfree", "aaa", "baa", "dfy", "cpi", "cpi_inflation_mom"
]

google_trend_cols = [c for c in df.columns if c.startswith("gt_")]
common_to_lag.extend(google_trend_cols)


#lag coin variables within each coin

for col in coin_specific_to_lag:
    if col in df.columns:
        df[f"{col}_lag1"] = df.groupby("coin")[col].shift(1)


#lag common variables by week

existing_common = [c for c in common_to_lag if c in df.columns]

weekly_common = (
    df[["week"] + existing_common]
    .drop_duplicates(subset=["week"])
    .sort_values("week")
    .copy()
)

for col in existing_common:
    weekly_common[f"{col}_lag1"] = weekly_common[col].shift(1)


#add 1 month lag for monthly variables

weekly_common["month"] = weekly_common["week"].dt.to_period("M")

monthly_snapshot = (
    weekly_common.groupby("month")[monthly_origin_vars]
    .last()
    .reset_index()
    .sort_values("month")
)

for col in monthly_origin_vars:
    if col in monthly_snapshot.columns:
        monthly_snapshot[f"{col}_1m_ago"] = monthly_snapshot[col].shift(1)

monthly_lag_cols = ["month"] + [
    f"{col}_1m_ago" for col in monthly_origin_vars if col in monthly_snapshot.columns
]

weekly_common = weekly_common.merge(
    monthly_snapshot[monthly_lag_cols],
    on="month",
    how="left"
)

lagged_common_cols = (
    ["week"]
    + [f"{col}_lag1" for col in existing_common]
    + [f"{col}_1m_ago" for col in monthly_origin_vars if col in monthly_snapshot.columns]
)

df = df.merge(weekly_common[lagged_common_cols], on="week", how="left")


#clean return lags

df["ret_1w_ago"] = df.groupby("coin")["ret_1w"].shift(1)
df["ret_2w_ago"] = df.groupby("coin")["ret_1w"].shift(2)
df["ret_3w_ago"] = df.groupby("coin")["ret_1w"].shift(3)

lagged_feature_cols = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_1m_ago")] + [
    "ret_1w_ago", "ret_2w_ago", "ret_3w_ago"
]

final_model_df = df[id_cols + target_cols + lagged_feature_cols].copy()
final_model_df = final_model_df.dropna(subset=["ret_1w"]).reset_index(drop=True)

final_model_df = final_model_df[
    final_model_df["week"] >= pd.Timestamp(COMMON_MODEL_START_DATE)
].copy()

final_model_df = final_model_df.sort_values(["coin", "week"]).reset_index(drop=True)

final_model_df.to_csv("8_lagged_merged_dataset.csv", index=False)

print(final_model_df.head())
print(final_model_df.shape)
print(final_model_df["coin"].value_counts())
print(final_model_df["week"].min(), final_model_df["week"].max())