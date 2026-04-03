import pandas as pd
import numpy as np
import statsmodels.api as sm



INPUT_CSV = "10_dropped_variables_and_renamed.csv"

OUTPUT_VALID_SUMMARY = "ols_valid_coin_summary_rolling.csv"
OUTPUT_VALID_RESIDUALS = "ols_valid_residuals_rolling.csv"

OUTPUT_TEST_SUMMARY = "ols_test_coin_summary_rolling.csv"
OUTPUT_TEST_RESIDUALS = "ols_test_residuals_rolling.csv"

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


def safe_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


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
    candidates = [
        "gt_cryptocurrency_lag1",
        "gt_cryptocurrency",
    ]
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
    split_name,
    split_start,
    split_end,
    rolling_train_weeks
):
    """
    For each forecast date in [split_start, split_end]:
    - train on the immediately preceding fixed-length rolling window
    - fit OLS
    - predict one week ahead
    """
    split_start = pd.to_datetime(split_start)
    split_end = pd.to_datetime(split_end)

    split_df = coin_model_df[
        (coin_model_df[date_col] >= split_start) &
        (coin_model_df[date_col] <= split_end)
    ].copy()

    if split_df.empty:
        return None, None

    residual_rows = []

    for pred_date in split_df[date_col].sort_values():
        # everything strictly before prediction date
        history_df = coin_model_df[coin_model_df[date_col] < pred_date].copy()

        # enforce original training start cutoff
        history_df = history_df[history_df[date_col] >= pd.to_datetime(TRAIN_START)].copy()

        if len(history_df) < rolling_train_weeks:
            # not enough rows yet for full fixed-length window
            continue

        train_df = history_df.tail(rolling_train_weeks).copy()
        pred_row = coin_model_df[coin_model_df[date_col] == pred_date].copy()

        if pred_row.empty:
            continue

        # OLS needs more rows than predictors + constant
        if len(train_df) <= len(predictors) + 1:
            continue

        X_train = sm.add_constant(train_df[predictors], has_constant="add")
        y_train = train_df[target]

        X_pred = sm.add_constant(pred_row[predictors], has_constant="add")

        # align columns in case of any mismatch
        X_pred = X_pred.reindex(columns=X_train.columns, fill_value=0)

        try:
            model = sm.OLS(y_train, X_train).fit()
            y_hat = float(model.predict(X_pred).iloc[0])
        except Exception as e:
            print(f"{coin} | {pred_date.date()} | rolling OLS failed: {e}")
            continue

        y_true = float(pred_row[target].iloc[0])

        residual_rows.append({
            "coin": coin,
            "week": pred_date,
            "split": split_name,
            "actual": y_true,
            "prediction": y_hat,
            "residual": y_true - y_hat,
            "train_start_used": train_df[date_col].min(),
            "train_end_used": train_df[date_col].max(),
            "train_n_obs": len(train_df),
        })

    if len(residual_rows) == 0:
        return None, None

    residual_df = pd.DataFrame(residual_rows).sort_values(["coin", "week"]).reset_index(drop=True)

    summary_row = {
        "coin": coin,
        "split": split_name,
        "n_obs": len(residual_df),
        "rmse": safe_rmse(residual_df["actual"], residual_df["prediction"]),
        "mae": safe_mae(residual_df["actual"], residual_df["prediction"]),
        "target": target,
        "n_predictors": len(predictors),
        "predictors_used": ", ".join(predictors),
        "rolling_train_weeks": rolling_train_weeks,
    }

    return summary_row, residual_df



df = pd.read_csv(INPUT_CSV)

date_col = find_first_existing_column(df, DATE_CANDIDATES, required=True)
coin_col = find_first_existing_column(df, COIN_CANDIDATES, required=True)

df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
df = df.sort_values([coin_col, date_col]).reset_index(drop=True)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")



base_predictors = build_base_predictors(
    df=df,
    date_col=date_col,
    coin_col=coin_col,
    target=TARGET,
)

print("\nBase non-GT predictors found in CSV:")
print(base_predictors)
print(f"Count = {len(base_predictors)}")



#RUN COIN BY COIN

valid_summary_rows = []
valid_residual_dfs = []

test_summary_rows = []
test_residual_dfs = []

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

    if len(predictors) == 0:
        print(f"Skipping {coin}: no usable predictors.")
        continue

    needed_cols = [date_col, coin_col, TARGET] + predictors
    coin_model_df = coin_df[needed_cols].copy()
    coin_model_df = coin_model_df.dropna(subset=[TARGET] + predictors).copy()
    coin_model_df = coin_model_df.sort_values(date_col).reset_index(drop=True)

    # keep only rows from TRAIN_START onward
    coin_model_df = coin_model_df[
        coin_model_df[date_col] >= pd.to_datetime(TRAIN_START)
    ].copy().reset_index(drop=True)

    if coin_model_df.empty:
        print(f"Skipping {coin}: no usable data after TRAIN_START.")
        continue

    valid_summary, valid_resid = run_fixed_length_rolling_forecast(
        coin_model_df=coin_model_df,
        predictors=predictors,
        date_col=date_col,
        coin=coin,
        target=TARGET,
        split_name="valid",
        split_start=VALID_START,
        split_end=VALID_END,
        rolling_train_weeks=ROLLING_TRAIN_WEEKS,
    )

    test_summary, test_resid = run_fixed_length_rolling_forecast(
        coin_model_df=coin_model_df,
        predictors=predictors,
        date_col=date_col,
        coin=coin,
        target=TARGET,
        split_name="test",
        split_start=TEST_START,
        split_end=TEST_END,
        rolling_train_weeks=ROLLING_TRAIN_WEEKS,
    )

    if valid_summary is not None:
        valid_summary["global_gt_used"] = global_gt_col if global_gt_col else ""
        valid_summary["coin_gt_used"] = coin_gt_col if coin_gt_col else ""
        valid_summary_rows.append(valid_summary)
        valid_residual_dfs.append(valid_resid)

    if test_summary is not None:
        test_summary["global_gt_used"] = global_gt_col if global_gt_col else ""
        test_summary["coin_gt_used"] = coin_gt_col if coin_gt_col else ""
        test_summary_rows.append(test_summary)
        test_residual_dfs.append(test_resid)

    print(
        f"{coin} | "
        f"n_predictors={len(predictors)} | "
        f"global_gt={global_gt_col} | "
        f"coin_gt={coin_gt_col} | "
        f"valid_n={0 if valid_summary is None else valid_summary['n_obs']} | "
        f"test_n={0 if test_summary is None else test_summary['n_obs']}"
    )



valid_summary_df = pd.DataFrame(valid_summary_rows)
test_summary_df = pd.DataFrame(test_summary_rows)

if not valid_summary_df.empty:
    valid_summary_df = valid_summary_df.sort_values(["coin", "split"]).reset_index(drop=True)

if not test_summary_df.empty:
    test_summary_df = test_summary_df.sort_values(["coin", "split"]).reset_index(drop=True)

if len(valid_residual_dfs) > 0:
    valid_residuals_df = pd.concat(valid_residual_dfs, ignore_index=True)
    valid_residuals_df = valid_residuals_df.sort_values(["coin", "week"]).reset_index(drop=True)
else:
    valid_residuals_df = pd.DataFrame(columns=[
        "coin", "week", "split", "actual", "prediction", "residual",
        "train_start_used", "train_end_used", "train_n_obs"
    ])

if len(test_residual_dfs) > 0:
    test_residuals_df = pd.concat(test_residual_dfs, ignore_index=True)
    test_residuals_df = test_residuals_df.sort_values(["coin", "week"]).reset_index(drop=True)
else:
    test_residuals_df = pd.DataFrame(columns=[
        "coin", "week", "split", "actual", "prediction", "residual",
        "train_start_used", "train_end_used", "train_n_obs"
    ])

valid_summary_df.to_csv(OUTPUT_VALID_SUMMARY, index=False)
valid_residuals_df.to_csv(OUTPUT_VALID_RESIDUALS, index=False)

test_summary_df.to_csv(OUTPUT_TEST_SUMMARY, index=False)
test_residuals_df.to_csv(OUTPUT_TEST_RESIDUALS, index=False)

print("\nSaved files:")
print(f"- {OUTPUT_VALID_SUMMARY}")
print(f"- {OUTPUT_VALID_RESIDUALS}")
print(f"- {OUTPUT_TEST_SUMMARY}")
print(f"- {OUTPUT_TEST_RESIDUALS}")