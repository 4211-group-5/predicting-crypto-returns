import pandas as pd

weekly_features = pd.read_csv("weekly_features.csv", parse_dates=["week"])
crypto_factors = pd.read_csv("crypto_factors_weekly.csv", parse_dates=["week"])
macro_factors = pd.read_csv("macro_factors_weekly.csv", parse_dates=["week"])
google_trends = pd.read_csv("google_trends_weekly.csv", parse_dates=["week"])

merged = weekly_features.merge(crypto_factors, on="week", how="left")
merged = merged.merge(macro_factors, on="week", how="left")
merged = merged.merge(google_trends, on="week", how="left")

merged = merged.sort_values(["coin", "week"]).reset_index(drop=True)
merged.to_csv("merged_weekly_dataset.csv", index=False)

print(merged.head())
print(merged.shape)