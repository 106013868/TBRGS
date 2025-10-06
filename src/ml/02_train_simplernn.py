# src/ml/02_train_simplernn.py
# Trains a SimpleRNN model, predicts on the test set, and saves results.

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import joblib
import os
import sys

# Define constants
WINDOW_SIZE = 168  # 7 days of historical data
PREDICT_HORIZON = 24 # Predict the next 24 hours
EPOCHS = 20
MODEL_NAME = "simplernn"
ARTIFACTS_DIR = "artifacts"
TRAIN_FILE = os.path.join(ARTIFACTS_DIR, "train_sequences.joblib")
TEST_FILE = os.path.join(ARTIFACTS_DIR, "test_sequences.joblib")
SCALER_FILE = os.path.join(ARTIFACTS_DIR, "scalers.joblib")
PROCESSED_DATA_FILE = os.path.join(ARTIFACTS_DIR, "processed_data.csv")


def load_data(file_path):
    """Loads sequences from a joblib file."""
    if not os.path.exists(file_path):
        print(f"Error: Required file not found: {file_path}")
        sys.exit(1)
    return joblib.load(file_path)


def build_and_train_model(X_train, y_train, input_shape):
    """Builds and trains the SimpleRNN model."""
    print("Training SimpleRNN...")
    
    model = Sequential([
        SimpleRNN(
            units=50, 
            activation='relu', 
            input_shape=input_shape,
            return_sequences=False
        ),
        Dense(PREDICT_HORIZON)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train the model
    # Note: We use a check to see if the model has already been trained and saved 
    # to avoid re-running the training part if possible.
    model_path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_model.keras")
    if os.path.exists(model_path):
        print(f"Model already saved at {model_path}. Loading weights...")
        model = tf.keras.models.load_model(model_path)
    else:
        model.fit(
            X_train, 
            y_train, 
            epochs=EPOCHS, 
            batch_size=256, 
            verbose=1
        )
        model.save(model_path)
        print(f"Saved model to: {model_path}")
        
    return model


def predict_and_unscale(model, test_sequences, scalers_dict, processed_df):
    """Generates predictions and unscales VPH values. FIX APPLIED HERE."""
    all_predictions = []
    
    # Iterate through each site's test data
    for hf_id, sequences in test_sequences.items():
        if not sequences or hf_id not in scalers_dict:
            continue
            
        # y_test_3d is the true value sequence (N_samples, 24, 1)
        X_test, y_test_3d, date_index = sequences 
        scaler = scalers_dict[hf_id]

        # 1. Generate predictions (2D: N_samples x 24_hours)
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # 2. Extract the first hour of the TRUE and PREDICTED values (column 0)
        # Reshape the 2D prediction output for the scaler
        y_pred_scaled_1h = y_pred_scaled[:, 0].reshape(-1, 1) 
        
        # Reshape the 3D true value output for the scaler
        y_true_scaled_1h = y_test_3d[:, 0]
        
        # 3. Inverse transform (unscale) to VPH values
        # .flatten() converts the resulting (N_samples, 1) array to a 1D array
        y_pred_vph_unscaled = scaler.inverse_transform(y_pred_scaled_1h).flatten()
        y_true_vph_unscaled = scaler.inverse_transform(y_true_scaled_1h).flatten()
        
        # 4. Get metadata for the predicted point
        first_road = processed_df[processed_df['hf_id'] == hf_id]['First_Road'].iloc[0]
        
        # Construct prediction records
        for i in range(len(date_index)):
            all_predictions.append({
                'hf_id': hf_id,
                'First_Road': first_road,
                'datetime': date_index[i],
                'y_true_vph': y_true_vph_unscaled[i],
                'y_pred_vph': y_pred_vph_unscaled[i]
            })

    return pd.DataFrame(all_predictions)


if __name__ == "__main__":
    
    # --- 1. Load Data ---
    print("Loading features…")
    train_sequences = load_data(TRAIN_FILE)
    test_sequences = load_data(TEST_FILE)
    scalers_dict = load_data(SCALER_FILE)
    processed_df = pd.read_csv(PROCESSED_DATA_FILE)

    # Consolidate training data from all sites
    all_X_train = np.concatenate([seq[0] for seq in train_sequences.values()])
    all_y_train = np.concatenate([seq[1] for seq in train_sequences.values()])

    print(f"  Train sequences: {all_X_train.shape}")
    print(f"  Test sites: {len(test_sequences)}")

    # --- 2. Build and Train Model ---
    INPUT_SHAPE = (WINDOW_SIZE, all_X_train.shape[2]) # (168, 1)
    model = build_and_train_model(all_X_train, all_y_train, INPUT_SHAPE)
    
    # --- 3. Predict and Save Results ---
    print("\nPredicting on test…")
    predictions_df = predict_and_unscale(model, test_sequences, scalers_dict, processed_df)
    
    predictions_path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Wrote predictions: {predictions_path}")
    
    print("Done.")