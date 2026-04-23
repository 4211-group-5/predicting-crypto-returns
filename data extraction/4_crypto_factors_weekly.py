import pandas as pd
import numpy as np

weekly_features = pd.read_csv("datasets/3_weekly_features.csv", parse_dates=["week"])

panel = weekly_features.sort_values(["coin", "week"]).copy()

factor_rows = []

for week, group in panel.groupby("week"):
    group = group.dropna(subset=["ret_1w"]).copy()

    if len(group) < 4:
        continue

   
    market_return_equal = group["ret_1w"].mean()

   
    group_mom = group.dropna(subset=["mom_4w"]).copy()
    if len(group_mom) >= 6:
        q30 = group_mom["mom_4w"].quantile(0.3)
        q70 = group_mom["mom_4w"].quantile(0.7)

        winners = group_mom[group_mom["mom_4w"] >= q70]["ret_1w"].mean()
        losers = group_mom[group_mom["mom_4w"] <= q30]["ret_1w"].mean()
        umd = winners - losers
    else:
        umd = np.nan

    factor_rows.append({
        "week": week,
        "market_return_equal": market_return_equal,
        "umd": umd
    })

crypto_factors = pd.DataFrame(factor_rows).sort_values("week").reset_index(drop=True)
crypto_factors.to_csv("datasets/4_crypto_factors_weekly.csv", index=False)

print(crypto_factors.head())
print(crypto_factors.shape)