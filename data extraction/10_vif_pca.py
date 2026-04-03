import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ======================================================
# 1. LOAD DATA
# ======================================================
file_path = r"c:\Users\hai liang\Downloads\acro\final_model_dataset_lagged_v6_winsorized.csv"
df = pd.read_csv(file_path)

print("Dataset loaded.")
print("Shape:", df.shape)

# ======================================================
# 2. KEEP ONLY NUMERIC COLUMNS
# ======================================================
X = df.select_dtypes(include=[np.number]).copy()

print("\nNumeric columns found:", len(X.columns))
print(X.columns.tolist())

# ======================================================
# 3. EXCLUDE TARGET / NON-PREDICTOR COLUMNS
#    Edit this list if needed
# ======================================================
exclude_cols = [
    "ret_1w",
    "ret_1w_original",
    "log_ret_1w",
    "log_ret_1w_original",
    "week",
    "year",
    "month",
    "id",
    "index"
]

exclude_existing = [col for col in exclude_cols if col in X.columns]
X = X.drop(columns=exclude_existing, errors="ignore")

print("\nDropped excluded columns:", exclude_existing)
print("Remaining predictors:", len(X.columns))

# ======================================================
# 4. DROP CONSTANT COLUMNS
# ======================================================
constant_cols = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]

if constant_cols:
    print("\nDropping constant columns:")
    print(constant_cols)
    X = X.drop(columns=constant_cols)

# ======================================================
# 5. CLEAN MISSING / INF VALUES
# ======================================================
X = X.replace([np.inf, -np.inf], np.nan)

print("\nMissing values by column:")
missing_counts = X.isna().sum()
print(missing_counts[missing_counts > 0].sort_values(ascending=False))

X = X.dropna()

print("\nRows after dropna:", len(X))
print("Columns used for VIF:", len(X.columns))

# ======================================================
# 6. OPTIONAL: DROP ONE VARIABLE FROM VERY HIGHLY
#    CORRELATED PAIRS BEFORE VIF
#    This avoids huge/singular VIF problems
# ======================================================
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

corr_threshold = 0.95
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]

if to_drop:
    print(f"\nDropping variables with pairwise corr > {corr_threshold} before VIF:")
    print(to_drop)
    X = X.drop(columns=to_drop)

print("\nFinal columns used for VIF:", len(X.columns))

# ======================================================
# 7. CALCULATE VIF
# ======================================================
vif_data = pd.DataFrame()
vif_data["variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif_data = vif_data.sort_values(by="VIF", ascending=False)

# ======================================================
# 8. SAVE OUTPUT
# ======================================================
output_path = r"c:\Users\hai liang\Downloads\acro\vif_results.csv"
vif_data.to_csv(output_path, index=False)

print("\nTop 30 VIF results:")
print(vif_data.head(30))

print(f"\nSaved VIF results to:\n{output_path}")