import pandas as pd
import numpy as np
import statsmodels.api as sm

INPUT_CSV = "10_dropped_variables_and_renamed.csv"

# Match requested output filenames
OUTPUT_RMSE = "ols_rsme.csv"
OUTPUT_RESIDUALS = "ols_residuals.csv"

TARGET = "ret_1w"

TRAIN_START = "2018-01-01"
TRAIN_END   = "2022-12-30" 
VALID_START = "2023-01-01"  
VALID_END   = "2024-12-30"  
TEST_START  = "2025-01-01"
TEST_END    = "2026-01-08"

DATE_CANDIDATES = ["week", "date", "Date", "timestamp"]
COIN_CANDIDATES = ["coin", "symbol", "Coin", "asset"]

ROLLING_TRAIN_WEEKS = 260

# never use these as predictors
EXCLUDED_PREDICTORS = {
    "ret_1w",
    "log_ret_1w",
    "ret_1w_original",
    "log_ret_1w_original",
}


def find_first_existing_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def build_coin_gt_candidates(coin):
    return [
        f"gt_{coin}_lag1",
        f"gt_{coin.upper()}_lag1",
        f"gt_{coin.lower()}_lag1",
        f"gt_{coin}",
        f"gt_{coin.upper()}",
        f"gt_{coin.lower()}",
    ]


def choose_coin_gt_column(df, coin):
    for col in build_coin_gt_candidates(coin):
        if col in df.columns:
            return col
    return None


def choose_global_gt_column(df):
    candidates = ["gt_cryptocurrency_lag1", "gt_cryptocurrency"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_base_predictors(df, date_col, coin_col, target):
    excluded = set(EXCLUDED_PREDICTORS) | {date_col, coin_col, target}
    predictors = []
    for col in df.columns:
        if col in excluded:
            continue
        if col.startswith("gt_"):
            continue
        predictors.append(col)
    return predictors


def run_fixed_length_rolling_forecast(
    coin_model_df,
    predictors,
    date_col,
    coin,
    target,
    split_start,
    split_end,
    rolling_train_weeks,
):
    """
    Forecast only for the requested date range and return residual rows.
    This version is simplified so the final CSV matches:
    - OLS_RSME.csv: columns [coin, rmse]
    - OLS_residuals.csv: columns [coin, week, actual, prediction, residual]
    """
    split_start = pd.to_datetime(split_start)
    split_end = pd.to_datetime(split_end)

    split_df = coin_model_df[
        (coin_model_df[date_col] >= split_start)
        & (coin_model_df[date_col] <= split_end)
    ].copy()

    if split_df.empty:
        return None, None

    residual_rows = []

    for pred_date in split_df[date_col].sort_values():
        history_df = coin_model_df[coin_model_df[date_col] < pred_date].copy()
        history_df = history_df[history_df[date_col] >= pd.to_datetime(TRAIN_START)].copy()

        if len(history_df) < rolling_train_weeks:
            continue

        train_df = history_df.tail(rolling_train_weeks).copy()
        pred_row = coin_model_df[coin_model_df[date_col] == pred_date].copy()

        if pred_row.empty:
            continue

        if len(train_df) <= len(predictors) + 1:
            continue

        X_train = sm.add_constant(train_df[predictors], has_constant="add")
        y_train = train_df[target]
        X_pred = sm.add_constant(pred_row[predictors], has_constant="add")
        X_pred = X_pred.reindex(columns=X_train.columns, fill_value=0)

        try:
            model = sm.OLS(y_train, X_train).fit()
            y_hat = float(model.predict(X_pred).iloc[0])
        except Exception as e:
            print(f"{coin} | {pred_date.date()} | rolling OLS failed: {e}")
            continue

        y_true = float(pred_row[target].iloc[0])

        residual_rows.append(
            {
                "coin": coin,
                "week": pred_date,
                "actual": y_true,
                "prediction": y_hat,
                "residual": y_true - y_hat,
            }
        )

    if not residual_rows:
        return None, None

    residual_df = pd.DataFrame(residual_rows).sort_values(["coin", "week"]).reset_index(drop=True)
    rmse_row = {"coin": coin, "rmse": safe_rmse(residual_df["actual"], residual_df["prediction"])}

    return rmse_row, residual_df


# ---------------- MAIN ----------------
df = pd.read_csv(INPUT_CSV)

date_col = find_first_existing_column(df, DATE_CANDIDATES, required=True)
coin_col = find_first_existing_column(df, COIN_CANDIDATES, required=True)

df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
df = df.sort_values([coin_col, date_col]).reset_index(drop=True)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

base_predictors = build_base_predictors(df=df, date_col=date_col, coin_col=coin_col, target=TARGET)

rmse_rows = []
residual_dfs = []

coins = sorted(df[coin_col].dropna().unique())

for coin in coins:
    coin_df = df[df[coin_col] == coin].copy().sort_values(date_col).reset_index(drop=True)

    global_gt_col = choose_global_gt_column(coin_df)
    coin_gt_col = choose_coin_gt_column(coin_df, coin)

    predictors = [col for col in base_predictors if col in coin_df.columns]

    if global_gt_col is not None and global_gt_col not in predictors:
        predictors.append(global_gt_col)
    if coin_gt_col is not None and coin_gt_col not in predictors:
        predictors.append(coin_gt_col)

    predictors = list(dict.fromkeys(predictors))

    if not predictors:
        print(f"Skipping {coin}: no usable predictors.")
        continue

    needed_cols = [date_col, coin_col, TARGET] + predictors
    coin_model_df = coin_df[needed_cols].copy()
    coin_model_df = coin_model_df.dropna(subset=[TARGET] + predictors).copy()
    coin_model_df = coin_model_df.sort_values(date_col).reset_index(drop=True)
    coin_model_df = coin_model_df[coin_model_df[date_col] >= pd.to_datetime(TRAIN_START)].copy().reset_index(drop=True)

    if coin_model_df.empty:
        print(f"Skipping {coin}: no usable data after TRAIN_START.")
        continue

    rmse_row, residual_df = run_fixed_length_rolling_forecast(
        coin_model_df=coin_model_df,
        predictors=predictors,
        date_col=date_col,
        coin=coin,
        target=TARGET,
        split_start=TEST_START,
        split_end=TEST_END,
        rolling_train_weeks=ROLLING_TRAIN_WEEKS,
    )

    if rmse_row is not None:
        rmse_rows.append(rmse_row)
        residual_dfs.append(residual_df)
        print(f"{coin} | test_n={len(residual_df)} | rmse={rmse_row['rmse']:.6f}")

rmse_df = pd.DataFrame(rmse_rows)
if not rmse_df.empty:
    rmse_df = rmse_df.sort_values("coin").reset_index(drop=True)

if residual_dfs:
    residuals_df = pd.concat(residual_dfs, ignore_index=True)
    residuals_df = residuals_df.sort_values(["coin", "week"]).reset_index(drop=True)
    # match the provided file's date display more closely when saved/opened in CSV
    residuals_df["week"] = pd.to_datetime(residuals_df["week"]).dt.strftime("%-d/%-m/%Y")
else:
    residuals_df = pd.DataFrame(columns=["coin", "week", "actual", "prediction", "residual"])

rmse_df.to_csv(OUTPUT_RMSE, index=False)
residuals_df.to_csv(OUTPUT_RESIDUALS, index=False)

print("\nSaved files:")
print(f"- {OUTPUT_RMSE}")
print(f"- {OUTPUT_RESIDUALS}")
