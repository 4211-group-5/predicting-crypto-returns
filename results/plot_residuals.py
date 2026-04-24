import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

def main():
    # current working directory
    results_folder = 'results/residuals'

    # get all model files in results folder
    model_files = list(Path(results_folder).glob('*residuals.csv'))

    for file in model_files:
        current_model_name = os.path.basename(file).replace('_residuals.csv', '')
        current_model_folder = f"results/plots/{current_model_name}"

        # Loading results from data_path
        df = pd.read_csv(file)

        # Ensure 'week' column is in datetime format
        try:
            df['week'] = pd.to_datetime(df['week'], format='%d/%m/%Y')
        except ValueError:
            pass

        # Get unique coins in the dataset
        coins = df['coin'].unique()

        # Save actual vs predicted plot for each coin
        for coin in coins:
            coin_results = df[df['coin'] == coin]
            print(coin_results.columns)
            if not coin_results.empty:
                plt.figure(figsize=(10, 5))
                plt.plot(coin_results['week'], coin_results['actual'], label='Actual', marker='o')
                plt.plot(coin_results['week'], coin_results['prediction'], label='Predicted', marker='x')
                plt.title(f'{current_model_name} Forecast vs Actual for {coin}')
                plt.xlabel('Week')
                plt.ylabel('Return 1 Week')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{current_model_folder}/{coin}_{current_model_name}_forecast.png")
                plt.close()

if __name__ == '__main__':
    main()