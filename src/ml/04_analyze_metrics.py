# src/ml/04_analyze_metrics.py
# Calculates VPH-based metrics for LSTM, GRU, and SimpleRNN prediction files.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def analyze_predictions(file_name: str, model_name: str, all_results: list):
    """Calculates and reports MAE and RMSE for a prediction file."""
    try:
        # Load the predictions CSV file
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"[WARN] Prediction file for {model_name.upper()} ('{file_name}') not found. Skipping analysis.")
        return

    # Column names are assumed to be consistent across prediction files
    y_true = df['y_true_vph']
    y_pred = df['y_pred_vph']

    # Clean data by removing any rows where true or predicted values are missing
    valid_data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()
    y_true = valid_data['y_true']
    y_pred = valid_data['y_pred']

    if y_true.empty:
        print(f"[ERROR] {model_name.upper()}: No valid data points found after cleaning. Cannot calculate metrics.")
        return

    # Calculate core metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Print results
    print(f"\n--- {model_name.upper()} Model Performance ---")
    print(f"Source File: {file_name}")
    print(f"Data Points Analyzed: {len(y_true):,}")
    print(f"Mean Absolute Error (VPH): {mae:,.2f}")
    print(f"Root Mean Squared Error (VPH): {rmse:,.2f}")
    print("-----------------------------------------")
    
    # Store results for final summary table
    all_results.append({
        'Model': model_name.upper(),
        'MAE (VPH)': mae,
        'RMSE (VPH)': rmse,
        'Data Points': len(y_true)
    })

def main():
    all_results = []
    
    # --- 1. Run LSTM Analysis ---
    lstm_file = os.path.join("artifacts", "lstm_predictions.csv")
    analyze_predictions(lstm_file, "LSTM", all_results)

    # --- 2. Run GRU Analysis ---
    gru_file = os.path.join("artifacts", "gru_predictions.csv")
    analyze_predictions(gru_file, "GRU", all_results)

    # --- 3. Run SimpleRNN Analysis (The Missing Step) ---
    simplernn_file = os.path.join("artifacts", "simplernn_predictions.csv")
    analyze_predictions(simplernn_file, "SimpleRNN", all_results)
    
    # --- 4. Print Final Summary Table ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(by='MAE (VPH)')
        
        print("\n\n=== Final Model Performance Summary (Sorted by MAE) ===")
        # Using to_markdown for a clean, aligned console output
        print(results_df.to_markdown(index=False, floatfmt=",.2f"))
        print("=======================================================")
    else:
        print("\n[INFO] No prediction files were successfully analyzed.")


if __name__ == "__main__":
    main()