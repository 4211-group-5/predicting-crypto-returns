# predicting-crypto-returns
DSE/QF4211 Group Project

## Setup and Installation

1. Create a virtual environment and load it (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

2. Install the required Python packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

*(Note: To run the R Markdown file or R scripts, please ensure R and RStudio are installed on your machine.)*

## Running the Data Extraction Pipeline

The data extraction scripts should be run in sequential order. You can run them using Python:

```bash
cd "data extraction"
python "1_daily_ohlcv.py"
python "2_weekly_ohlcv.py"
python "3_weekly_features.py"
python "4_crypto_factors_weekly.py"
python "5_macro_factors_weekly.py"
python "6_google_trends_weekly.py"
python "7_merged_weekly_dataset.py"
python "8_lagged_merged_dataset.py"
python "9_final_outlier_removed_dataset.py"
python "10_vif_pca.py"
cd ..
```

## Running the Models

- **ARIMA Model**:
  ```bash
  python models/arima.py
  ```

- **Ordinary Least Squares (OLS)**:
  ```bash
  python models/ordinary_least_squares.py
  ```

- **Fixed Effects Regression**:
  Open the following file and click **Run > Run All Cells** to execute: `models/fixed_effects_regression.ipynb`

- **Partial Least Squares (PLS)**:
  Open the following file and click **Run > Run All Cells** to execute: `models/partial_least_squares.ipynb`

- **Random Forest**:
  Open `models/random_forest/randomforest.Rmd` using **RStudio**. You can run the code chunks sequentially, or click the **Knit** button to generate the final report (HTML/PDF/Word). 

## Generating Residual Plots

To generate residual plots from the various models' outputs, run the plotting script:
```bash
python results/plot_residuals.py
```

## Evaluation

- **Diebold-Mariano Test (DM Test)**: 
  Open `Evaluation/dm_test.ipynb` and run all cells to evaluate the predictions against the benchmark.
- **R-squared Analysis**: 
  Open `Evaluation/R_square.Rmd` in RStudio and knit the document, or run it sequentially using R to get R-squared metrics.

## Feature Analysis

### SHAP Values
1. **Generate SHAP Values**: 
   Run the R script to calculate and generate SHAP data:
   ```bash
   Rscript "feature analysis/SHAP/shap_generation.R"
   ```
2. **Plot SHAP Values**: 
   Open `feature analysis/SHAP/shap_plots.ipynb` and execute all cells to visualize the SHAP analysis.

### Variable Importance Projection (VIP)
- **PLS VIP**: 
  Open `feature analysis/VIP/pls.ipynb` and run all cells to examine variable importance.
