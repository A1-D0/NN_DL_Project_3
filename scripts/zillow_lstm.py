# Bidirectional LSTM for Zillow ZHVI Forecasting:
# This script implements a Bidirectional LSTM model to forecast Zillow's ZHVI (Zillow Home Value Index) using historical data.
# The model is trained on a time series dataset, and the performance is evaluated using RMSE, MAE, and R^2 metrics.
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf


# =======================
# data preprocessing
# =======================
def load_and_preprocess_data(filepath, time_steps=12):
    df = pd.read_csv(filepath)

    # drop col non-informative col for time series
    cols_to_drop = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # transpose dataset such that each row is timestamp, each column is region ZHVI time series
    df_ts = df.set_index(df.columns[0]).T
    df_ts.index = pd.to_datetime(df_ts.index)

    # fill missing vals w forward fill
    df_ts.fillna(method='ffill', inplace=True)
    df_ts.dropna(axis=1, inplace=True)  # drop remaining NANs

    # normalize and stack region time series in one train/test set
    all_X, all_y = [], []
    scaler = MinMaxScaler()

    for col in df_ts.columns:
        series = df_ts[col].values.reshape(-1, 1)
        series_scaled = scaler.fit_transform(series)
        X, y = create_sequences(series_scaled, time_steps)
        all_X.append(X)
        all_y.append(y)

    X = np.vstack(all_X)
    y = np.vstack(all_y)
    split = int(len(X) * 0.8)

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    return X_train, y_train, X_test, y_test, scaler

# converts a time series into overlapping (X, y) pairs using a fixed-length sliding window
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# =======================
# model definition
# =======================
def build_bilstm_model(time_steps, num_features=1):
    # lightweight bi-lstm w 32 units
    model = Sequential()
    model.add(Bidirectional(LSTM(32), input_shape=(time_steps, num_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# =======================
# evaluation
# =======================
def evaluate_model(model, X_test, y_test, scaler):
    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)

    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.3f}")

    # save evaluation metrics to file
    output_dir = '/Users/anthonyroca/csc_375/NN_DL_Project_3/output/Zillow/zillow_hvi.csv'
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"R^2: {r2:.3f}\n")
    print(f"[INFO] Evaluation metrics saved to {metrics_path}")

    return y_test_inv, y_pred_inv

# =======================
# visualization
# =======================
def plot_predictions(y_test_inv, y_pred_inv):
    output_dir = '/Users/anthonyroca/csc_375/NN_DL_Project_3/output/Zillow/zillow_hvi.csv'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual')
    plt.plot(y_pred_inv, label='Predicted')
    plt.title('BiLSTM ZHVI Forecast')
    plt.xlabel('Time')
    plt.ylabel('ZHVI')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'predictions.png')
    plt.savefig(plot_path)
    print(f"[INFO] Predictions plot saved to {plot_path}")

# set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# define shared hyperparameters
TIME_STEPS = 12

# =======================
# main execution block
# =======================
if __name__ == '__main__':
    # abs path to data
    filepath = '/Users/anthonyroca/csc_375/NN_DL_Project_3/data/zillow_hvi.csv'
    print("[INFO] Loading and preprocessing data...")

    # load and preprocess data
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(filepath, time_steps=TIME_STEPS)
    print(f"[INFO] Data loaded. Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # construct model architecture
    print("[INFO] Building BiLSTM model...")
    model = build_bilstm_model(time_steps=TIME_STEPS)
    print("[INFO] Model built.")

    # begin training
    print("[INFO] Starting training...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1, verbose=1, callbacks=[early_stop])
    print("[INFO] Training complete.")

    # save model
    output_dir = '/Users/anthonyroca/csc_375/NN_DL_Project_3/output/Zillow'
    model_path = os.path.join(output_dir, 'bilstm_model.h5')
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

    # evaluate model performance
    print("[INFO] Evaluating model...")
    y_test_inv, y_pred_inv = evaluate_model(model, X_test, y_test, scaler)
    plot_predictions(y_test_inv, y_pred_inv)
