import pandas as pd
import numpy as np

input_path = "datasets/8_lagged_merged_dataset.csv"
output_path = "datasets/9_final_outlier_removed_dataset.csv"

df = pd.read_csv(input_path, parse_dates=["week"])

# remove outlier in ret_1w across the whole dataset
lower_q = df["ret_1w"].quantile(0.01)
upper_q = df["ret_1w"].quantile(0.99)

df_w = df.copy()

# keep original
df_w["ret_1w_original"] = df_w["ret_1w"]
df_w["log_ret_1w_original"] = df_w["log_ret_1w"]

# remove outlier in ret_1w
df_w["ret_1w"] = df_w["ret_1w"].clip(lower=lower_q, upper=upper_q)

# Recompute log return 

valid = df_w["ret_1w"] > -1
df_w.loc[valid, "log_ret_1w"] = np.log(1 + df_w.loc[valid, "ret_1w"])
df_w.loc[~valid, "log_ret_1w"] = np.nan


df_w.to_csv(output_path, index=False)

print("Saved to:", output_path)
print("Lower clip (1%):", lower_q)
print("Upper clip (99%):", upper_q)
print("Rows clipped low:", (df["ret_1w"] < lower_q).sum())
print("Rows clipped high:", (df["ret_1w"] > upper_q).sum())
print(df_w.head())
print(df_w.shape)