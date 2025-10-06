from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

# TensorFlow / Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths / config
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH   = ROOT / "artifacts" / "processed_data.csv"
MODEL_PATH  = ROOT / "artifacts" / "gru_model.keras"
OUTPUT_PATH = ROOT / "artifacts" / "gru_predictions.csv"

# Windowing
INPUT_STEPS   = 24 * 7   # history window length (168 hours)
FORECAST_HRS  = 1        # predict next 1 hour
MIN_SERIES_HR = 24 * 30  # require >= 30 days total per site

# Time splits (by absolute time across the month)
TEST_DAYS = 7            # last 7 days are test
VAL_DAYS  = 3            # last 3 days of the training portion are val (optional)

# Training
EPOCHS     = 20
BATCH_SIZE = 256

VPH_COL = "volume_vph"
ROAD_COL = "First_Road"

# Methods
def load_processed_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"processed_data.csv not found at: {path}")
    print("Loading features…")
    df = pd.read_csv(path, parse_dates=["datetime"])

    # Flexible mapping (accepts different casings/aliases)
    cols_lower = {c.lower(): c for c in df.columns}
    hf_col   = cols_lower.get("hf_id")
    road_col = cols_lower.get("first_road") or ("First_Road" if "First_Road" in df.columns else None)
    vph_col  = cols_lower.get("volume_vph") or cols_lower.get("vehicle_vph")

    if hf_col is None or road_col is None or vph_col is None or "datetime" not in df.columns:
        raise ValueError(
            f"processed_data.csv missing required columns. "
            f"Found: {list(df.columns)}; need hf_id, First_Road/first_road, "
            f"datetime, and volume_vph/vehicle_vph."
        )

    df = df.rename(columns={hf_col: "hf_id", road_col: ROAD_COL, vph_col: VPH_COL})
    df = df.dropna(subset=["hf_id", VPH_COL, "datetime"]).copy()
    df["hf_id"] = df["hf_id"].astype(int)
    df = df[["hf_id", ROAD_COL, "datetime", VPH_COL]].sort_values(["hf_id", "datetime"]).reset_index(drop=True)
    return df

def make_continuous_hourly(df_site: pd.DataFrame) -> pd.DataFrame:
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
    last_time = df["datetime"].max()
    test_start = last_time - pd.Timedelta(days=TEST_DAYS) + pd.Timedelta(hours=1)
    train_all = df[df["datetime"] < test_start].copy()
    test_df   = df[df["datetime"] >= test_start].copy()
    if train_all.empty or test_df.empty:
        raise RuntimeError("Train or Test split empty. Adjust TEST_DAYS or check data range.")
    val_cut = train_all["datetime"].max() - pd.Timedelta(days=VAL_DAYS)
    val_df  = train_all[train_all["datetime"] >= val_cut].copy()
    train_df = train_all[train_all["datetime"] < val_cut].copy()
    return train_df, val_df, test_df, test_start

def make_gru_model(input_steps: int) -> Sequential:
    model = Sequential([
        GRU(64, input_shape=(input_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Main
def main():
    df = load_processed_data(DATA_PATH)

    # Keep sites with enough hours
    counts = df.groupby("hf_id")["datetime"].count()
    enough_sites = set(counts[counts >= MIN_SERIES_HR].index.tolist())
    df = df[df["hf_id"].isin(enough_sites)].copy()
    print(f"  Sites with >= {MIN_SERIES_HR} hours: {len(enough_sites)}")

    # Splits (by time)
    train_df, val_df, test_df, test_start = time_split(df)

    # Per-site scalers (mean, std) from TRAIN ONLY
    scalers: Dict[int, Tuple[float, float]] = {}
    road_map = df[["hf_id", ROAD_COL]].drop_duplicates().set_index("hf_id")[ROAD_COL].to_dict()

    # -------- Build TRAIN windows (fit scalers) --------
    X_tr_list, y_tr_list = [], []
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
            X_tr_list.append(X); y_tr_list.append(y)

    if not X_tr_list:
        raise RuntimeError("No training sequences built. Try reducing INPUT_STEPS or MIN_SERIES_HR.")

    X_train = np.concatenate(X_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)
    print(f"  Train sequences: {X_train.shape}")

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
        print(f"  Val sequences:   {X_val.shape}")
    else:
        val_data = None
        print("  No validation sequences (dataset/time split may be tight).")

    # -------- Build & Train GRU --------
    model = make_gru_model(INPUT_STEPS)
    callbacks = [
        EarlyStopping(monitor="val_loss" if val_data else "loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_loss" if val_data else "loss", save_best_only=True)
    ]
    print("Training…")
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

    print("Predicting on test…")
    pred_rows: List[dict] = []
    sites_seen_train = set(scalers.keys())
    sites_in_test = sorted(test_df["hf_id"].unique().tolist())
    print(f"  Test sites: {len(sites_in_test)} (overlap with train scalers: {len(sites_seen_train & set(sites_in_test))})")

    built_sites = 0
    skipped_no_scaler = 0
    skipped_no_sequences = 0

    # Build sequences from FULL site series but retain only targets in test window
    for hf, g_all in df.groupby("hf_id", sort=True):
        if int(hf) not in sites_in_test:
            continue
        if int(hf) not in scalers:
            skipped_no_scaler += 1
            continue

        g_full = make_continuous_hourly(g_all)
        values = g_full[VPH_COL].to_numpy(dtype=float)
        ts     = g_full["datetime"].to_numpy()

        mean, std = scalers[int(hf)]
        values_scaled = (values - mean) / (std if std > 1e-6 else 1.0)

        X, y_scaled, tgt_ts = build_sequences(values_scaled, ts, INPUT_STEPS, FORECAST_HRS)
        if len(X) == 0:
            skipped_no_sequences += 1
            continue

        # Keep only targets in the test period
        mask = tgt_ts >= pd.to_datetime(test_start).to_datetime64()
        if not mask.any():
            skipped_no_sequences += 1
            continue
        X, y_scaled, tgt_ts = X[mask], y_scaled[mask], tgt_ts[mask]

        built_sites += 1
        yhat_scaled = model.predict(X, batch_size=512, verbose=0).reshape(-1)

        # Unscale
        y_true = y_scaled * std + mean
        y_pred = yhat_scaled * std + mean

        road = road_map.get(int(hf), "")
        for i in range(len(y_pred)):
            pred_rows.append({
                "hf_id": int(hf),
                ROAD_COL: road,
                "datetime": pd.to_datetime(tgt_ts[i]),
                "y_true_vph": float(max(0.0, y_true[i])),
                "y_pred_vph": float(max(0.0, y_pred[i])),
            })

    print(f"  Built predictions for sites: {built_sites}")
    print(f"  Skipped (no scaler): {skipped_no_scaler}")
    print(f"  Skipped (no sequences): {skipped_no_sequences}")

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
