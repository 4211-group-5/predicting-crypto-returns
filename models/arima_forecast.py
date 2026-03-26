import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
import statsmodels.api as sm

def main():
    data_path = '../datasets/10_dropped_variables_and_renamed.csv'
    results_folder = '../results/ARIMAX/'
    
    # Loading data from data_path
    df = pd.read_csv(data_path)
    
    # Ensure 'week' column is in datetime format
    df['week'] = pd.to_datetime(df['week'], format='%d/%m/%Y')
    
    train_valid_end = pd.to_datetime('2024-12-30')
    test_start = pd.to_datetime('2025-01-01')
    test_end = pd.to_datetime('2026-01-08')
    
    target_col = 'ret_1w'
    exclude_cols = ['coin', 'week', 'ret_1w']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Get unique coins in the dataset
    coins = df['coin'].unique()
    
    results = []
    
    for coin in coins:
        print(f"\nProcessing {coin}...")
        df_coin = df[df['coin'] == coin].sort_values('week').set_index('week')
        
        # Historical data (Train + Valid)
        train_data = df_coin[df_coin.index <= train_valid_end][target_col]
        test_data = df_coin[(df_coin.index >= test_start) & (df_coin.index <= test_end)][target_col]

        train_exog = df_coin[df_coin.index <= train_valid_end][feature_cols]
        test_exog = df_coin[(df_coin.index >= test_start) & (df_coin.index <= test_end)][feature_cols]

        if not len(train_data) or not len(test_data):
            print(f"Insufficient data for {coin}.")
            continue

        # Fit auto_arima on train data to find the best order
        try:
            auto_model = auto_arima(
                train_data,
                exogenous=train_exog,
                start_p=1, start_q=1,
                max_p=3, max_q=3,
                d=0, max_d=0,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

            order = auto_model.order
            # Default to a simple ARIMA(1,0,1) if auto_arima fails to find a better model
            if order == (0, 0, 0):
                order = (1, 0, 1)
            print(f"Selected ARIMA order for {coin}: {order}")
        except Exception as e:
            print(f"Error fitting auto_arima for {coin}: {e}")
            continue

        window_size = len(train_data)
        history_y = list(train_data.values)
        history_exog = train_exog.values.tolist()
        
        # Perform rolling out-of-sample forecast using a fixed-size window
        for i, dt in enumerate(test_data.index):
            # The current estimation window is the last 'window_size' observations
            current_y = history_y[-window_size:]
            current_exog = np.array(history_exog[-window_size:])

            current_exog_test = test_exog.iloc[i].values.reshape(1, -1)
            
            try:
                # Re-estimate the model with the selected order
                model = sm.tsa.ARIMA(
                    current_y,
                    exog=current_exog,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                fit_model = model.fit(method_kwargs={'maxiter':1000})
                
                # Predict one step ahead
                pred_result = fit_model.forecast(steps=1, exog=current_exog_test)
                pred = pred_result.values[0] if isinstance(pred_result, pd.Series) else pred_result[0]
            except Exception as e:
                print(f"Error fitting ARIMA for {coin} at {dt}: {e}")
                pred = np.nan
                
            actual = test_data.iloc[i]
            results.append({
                'coin': coin,
                'week': dt,
                'actual_ret_1w': actual,
                'predicted_ret_1w': pred,
                'mse': (actual - pred) ** 2
            })
            
            # Update history with the actual observed value for the next step's window
            history_y.append(actual)
            history_exog.append(current_exog_test.flatten())
            
        if results:
            results_df = pd.DataFrame(results)
            # Save results to output_path
            results_df.to_csv(f"{results_folder}arimax_forecast_results.csv", index=False)
        
    # Drop rows with NaN predictions
    results_df_clean = results_df.dropna(subset=['mse'])

    # Overall RMSE
    rmse_overall = np.sqrt(np.mean(
        results_df_clean['mse']
    ))
    print(f"\nOverall RMSE: {rmse_overall:.6f}")

    # RMSE per coin
    rmse_per_coin = results_df_clean.groupby('coin').apply(
        lambda x: np.sqrt(np.mean((x['mse'])))
    ).reset_index(name='RMSE')

    print("\nRMSE per coin:")
    print(rmse_per_coin)

    # Save RMSE results
    rmse_per_coin.to_csv(f"{results_folder}arimax_rmse_per_coin.csv", index=False)

    # Save actual vs predicted plot for each coin
    for coin in coins:
        coin_results = results_df[results_df['coin'] == coin]
        if not coin_results.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(coin_results['week'], coin_results['actual_ret_1w'], label='Actual', marker='o')
            plt.plot(coin_results['week'], coin_results['predicted_ret_1w'], label='Predicted', marker='x')
            plt.title(f'ARIMAX Forecast vs Actual for {coin}')
            plt.xlabel('Week')
            plt.ylabel('Return 1 Week')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{results_folder}{coin}_arimax_forecast.png")
            plt.close()

if __name__ == '__main__':
    main()