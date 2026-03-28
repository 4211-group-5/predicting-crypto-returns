import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

def standardise_pls_data():
    pls_folder = Path('../results/partial_least_squares/predictions')

    pls_data = []
    for file in pls_folder.iterdir():
        if file.is_file():
            df = pd.read_excel(file)
            pls_data.append(df)

    pls_df = pd.concat(pls_data, ignore_index=True)

    # rename prediction column
    pls_df.rename(columns={"pred": "prediction"}, inplace=True)

    # save to pls folder
    pls_df.to_csv(f"../results/partial_least_squares/pls_residuals.csv", index=False)

def main():
    standardise_pls_data()

    # current working directory
    results_folder = '../results/'

    # get all model folders in results folder
    model_folders = [f.path for f in os.scandir(results_folder) if f.is_dir()]

    for folder in model_folders:
        current_model_folder = folder
        current_model_name = os.path.basename(folder)

        # get file with residual in name
        data_path = list(Path(current_model_folder).glob('*residuals.csv'))[0]

        # Loading results from data_path
        df = pd.read_csv(data_path)

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