from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

# TensorFlow / Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths and filenames used by the pipeline
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH   = ROOT / "artifacts" / "processed_data.csv"
MODEL_PATH  = ROOT / "artifacts" / "lstm_model.keras"
OUTPUT_PATH = ROOT / "artifacts" / "lstm_predictions.csv"

# Windowing choices for sequence modeling
INPUT_STEPS   = 24 * 7   # 168 hours history per sample
FORECAST_HRS  = 1        # predict next 1 hour

# A loose notion of “nearly a month” so we can report dataset health
MIN_SERIES_HR = 24 * 26  # 624 hours (≈ 26 days)

# Time split sizes (from the end of the month)
TEST_DAYS = 7
VAL_DAYS  = 3

# Training setup
EPOCHS     = 20
BATCH_SIZE = 256

# Column names we expect to work with
VPH_COL = "volume_vph"
ROAD_COL = "First_Road"


def load_processed_data(path: Path) -> pd.DataFrame:
    """Load processed_data.csv and normalize column names we rely on."""
    if not path.exists():
        raise FileNotFoundError(f"processed_data.csv not found at: {path}")
    print("Loading features...")
    df = pd.read_csv(path, parse_dates=["datetime"])

    cols_lower = {c.lower(): c for c in df.columns}
    hf_col   = cols_lower.get("hf_id")
    road_col = cols_lower.get("first_road") or ("First_Road" if "First_Road" in df.columns else None)
    vph_col  = cols_lower.get("volume_vph") or cols_lower.get("vehicle_vph")

    if hf_col is None or road_col is None or vph_col is None or "datetime" not in df.columns:
        raise ValueError("Missing required columns (hf_id, First_Road, volume_vph, datetime).")

    df = df.rename(columns={hf_col: "hf_id", road_col: ROAD_COL, vph_col: VPH_COL})
    df = df.dropna(subset=["hf_id", VPH_COL, "datetime"]).copy()
    df["hf_id"] = df["hf_id"].astype(int)

    return (
        df[["hf_id", ROAD_COL, "datetime", VPH_COL]]
        .sort_values(["hf_id", "datetime"])
        .reset_index(drop=True)
    )


def make_continuous_hourly(df_site: pd.DataFrame) -> pd.DataFrame:
    """Fill hourly timeline per site, forward-fill labels, and interpolate gaps in flow."""
    df_site = df_site.sort_values("datetime").reset_index(drop=True)
    agg = (
        df_site.groupby("datetime", as_index=True)
               .agg({"hf_id": "first", ROAD_COL: "first", VPH_COL: "mean"})
               .reset_index()
    )
    start, end = agg["datetime"].min(), agg["datetime"].max()
    full_idx = pd.date_range(start=start, end=end, freq="h")
    out = (
        agg.set_index("datetime").reindex(full_idx)
           .rename_axis("datetime").reset_index()
    )
    out["hf_id"] = out["hf_id"].ffill().bfill()
    out[ROAD_COL] = out[ROAD_COL].ffill().bfill()
    out[VPH_COL] = out[VPH_COL].interpolate(limit_direction="both")
    return out


def build_sequences(values: np.ndarray,
                    timestamps: np.ndarray,
                    input_steps: int,
                    forecast_hrs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a 1D series into overlapping (X,y,t) windows for supervised learning."""
    xs, ys, ts = [], [], []
    total = len(values)
    end_idx = total - forecast_hrs
    for i in range(input_steps, end_idx):
        x = values[i - input_steps:i]
        y = values[i:i + forecast_hrs].sum()
        xs.append(x.reshape(-1, 1))
        ys.append(y)
        ts.append(timestamps[i + forecast_hrs - 1])
    if not xs:
        return np.zeros((0, input_steps, 1)), np.zeros((0,)), np.array([], dtype="datetime64[ns]")
    return np.stack(xs), np.array(ys), np.array(ts)


def time_split(df: pd.DataFrame):
    """Split into train/val/test using final days of the month for validation and test."""
    last_time = df["datetime"].max()
    test_start = last_time - pd.Timedelta(days=TEST_DAYS) + pd.Timedelta(hours=1)
    train_all = df[df["datetime"] < test_start].copy()
    test_df   = df[df["datetime"] >= test_start].copy()
    if train_all.empty or test_df.empty:
        raise RuntimeError("Train or Test split empty; adjust TEST_DAYS or check data range.")
    val_cut  = train_all["datetime"].max() - pd.Timedelta(days=VAL_DAYS)
    val_df   = train_all[train_all["datetime"] >= val_cut].copy()
    train_df = train_all[train_all["datetime"] <  val_cut].copy()
    return train_df, val_df, test_df, test_start


def make_lstm_model(input_steps: int) -> Sequential:
    """Build a small LSTM regressor for one-step-ahead flow prediction."""
    model = Sequential([
        LSTM(64, input_shape=(input_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    df = load_processed_data(DATA_PATH)

    # Print a friendly data-health summary without using strict filtering.
    counts = df.groupby("hf_id")["datetime"].count()
    approx_month_sites = set(counts[counts >= MIN_SERIES_HR].index.tolist())
    print(f"   Sites with >= {MIN_SERIES_HR} hours (approx month): {len(approx_month_sites)}")

    # Build time splits for the whole dataset.
    train_df, val_df, test_df, test_start = time_split(df)

    # Prepare per-site scalers from TRAIN only (mean/std per site).
    scalers: Dict[int, Tuple[float, float]] = {}
    road_map = df[["hf_id", ROAD_COL]].drop_duplicates().set_index("hf_id")[ROAD_COL].to_dict()

    # Make training windows across all sites.
    X_tr_list, y_tr_list = [], []
    kept_train_sites = 0

    for hf, g in train_df.groupby("hf_id", sort=True):
        g_full = make_continuous_hourly(g)
        values = g_full[VPH_COL].to_numpy(dtype=float)
        ts     = g_full["datetime"].to_numpy()

        mean = float(np.nanmean(values))
        std  = float(np.nanstd(values))
        if std < 1e-6:
            std = 1.0
        scalers[int(hf)] = (mean, std)

        values = (values - mean) / std
        X, y, _ = build_sequences(values, ts, INPUT_STEPS, FORECAST_HRS)
        if len(X) > 0:
            kept_train_sites += 1
            X_tr_list.append(X)
            y_tr_list.append(y)

    if not X_tr_list:
        raise RuntimeError("No training sequences built; check INPUT_STEPS and data coverage.")

    X_train = np.concatenate(X_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)
    print(f"   Train sequences: {X_train.shape} (sites contributing: {kept_train_sites})")

    # Make validation windows for early stopping if available.
    X_va_list, y_va_list = [], []
    for hf, g in val_df.groupby("hf_id", sort=True):
        if int(hf) not in scalers:
            continue
        g_full = make_continuous_hourly(g)
        values = g_full[VPH_COL].to_numpy(dtype=float)
        ts     = g_full["datetime"].to_numpy()
        mean, std = scalers[int(hf)]
        values = (values - mean) / std
        X, y, _ = build_sequences(values, ts, INPUT_STEPS, FORECAST_HRS)
        if len(X) > 0:
            X_va_list.append(X); y_va_list.append(y)

    if X_va_list:
        X_val = np.concatenate(X_va_list, axis=0)
        y_val = np.concatenate(y_va_list, axis=0)
        val_data = (X_val, y_val)
        print(f"   Val sequences:   {X_val.shape}")
    else:
        val_data = None
        print("   No validation sequences.")

    # Train the LSTM with early stopping and save the best model.
    model = make_lstm_model(INPUT_STEPS)
    callbacks = [
        EarlyStopping(monitor="val_loss" if val_data else "loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_loss" if val_data else "loss", save_best_only=True)
    ]
    print("Training LSTM...")
    model.fit(
        X_train, y_train,
        validation_data=val_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )
    if not MODEL_PATH.exists():
        model.save(MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

    # Build predictions for all sites that appear in TEST, using TRAIN scalers.
    print("Predicting on test...")
    pred_rows: List[dict] = []
    sites_in_test = sorted(test_df["hf_id"].unique().tolist())
    built_sites = 0

    for hf, g_all in df.groupby("hf_id", sort=True):
        if int(hf) not in sites_in_test or int(hf) not in scalers:
            continue

        g_full = make_continuous_hourly(g_all)
        values = g_full[VPH_COL].to_numpy(dtype=float)
        ts     = g_full["datetime"].to_numpy()

        mean, std = scalers[int(hf)]
        scale = std if std > 1e-6 else 1.0
        values_scaled = (values - mean) / scale

        X, y_scaled, tgt_ts = build_sequences(values_scaled, ts, INPUT_STEPS, FORECAST_HRS)
        if len(X) == 0:
            continue

        mask = tgt_ts >= pd.to_datetime(test_start).to_datetime64()
        if not mask.any():
            continue
        X, y_scaled, tgt_ts = X[mask], y_scaled[mask], tgt_ts[mask]

        built_sites += 1
        yhat_scaled = model.predict(X, batch_size=512, verbose=0).reshape(-1)

        y_true = y_scaled * scale + mean
        y_pred = yhat_scaled * scale + mean

        road = road_map.get(int(hf), "")
        for i in range(len(y_pred)):
            pred_rows.append({
                "hf_id": int(hf),
                ROAD_COL: road,
                "datetime": pd.to_datetime(tgt_ts[i]),
                "y_true_vph": float(max(0.0, y_true[i])),
                "y_pred_vph": float(max(0.0, y_pred[i])),
            })

    print(f"   Built predictions for sites: {built_sites}")

    if not pred_rows:
        print("WARNING: No prediction rows were generated.")
        empty = pd.DataFrame(columns=["hf_id", ROAD_COL, "datetime", "y_true_vph", "y_pred_vph"])
        empty.to_csv(OUTPUT_PATH, index=False)
        print(f"Wrote empty predictions file: {OUTPUT_PATH}")
        return

    preds = pd.DataFrame(pred_rows).sort_values(["hf_id", "datetime"]).reset_index(drop=True)
    preds.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote predictions: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
