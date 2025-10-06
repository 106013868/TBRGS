# src/ml/02_train_gru.py
# Train a GRU on per-site hourly vehicle_vph using a dynamic eligibility rule
# and predict the last TEST_HOURS per site. Saves:
#   artifacts/gru_model.keras
#   artifacts/gru_predictions.csv
#
# Assumes artifacts/processed_data.csv with columns:
#   hf_id, First_Road, datetime, volume_vph
#
# Uses a single global GRU across all eligible sites, scaling per-site.

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- Paths ----------------
IS_FROZEN = getattr(sys, "frozen", False)
if IS_FROZEN:
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parents[2]  # repo root

ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

DATA_PATH  = ART / "processed_data.csv"     # <== renamed from features_hourly.csv
MODEL_PATH = ART / "gru_model.keras"
PRED_PATH  = ART / "gru_predictions.csv"

# ---------------- Windowing & eligibility params ----------------
LOOKBACK      = 168   # hours of history per training sample (1 week)
TEST_HOURS    = 168    # hours to predict per site
# (We train 1-step ahead; predictions are made for the last TEST_HOURS timestamps.)

# Eligibility rule (same as 01_make_windows.py)
HORIZON       = 24    # used only for MIN_HOURS calculation (kept consistent)
MIN_WINDOWS   = 300
MIN_WEEKS     = 3
MIN_COVERAGE  = 0.80

REQ_BY_WINDOWS = LOOKBACK + HORIZON + MIN_WINDOWS   # 168+24+300 = 492
REQ_BY_WEEKS   = MIN_WEEKS * 168                    # 3*168 = 504
MIN_HOURS_REQ  = max(REQ_BY_WINDOWS, REQ_BY_WEEKS)  # 504

# ---------------- Helpers ----------------
def expected_hours_between(min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> int:
    if pd.isna(min_ts) or pd.isna(max_ts):
        return 0
    return int((max_ts - min_ts).total_seconds() // 3600) + 1

def make_one_step_windows(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple 1-step-ahead supervised dataset:
    X[t] = series[t-lookback : t] (length = lookback)
    y[t] = series[t] (the next value)
    Returned shapes:
      X: (num_samples, lookback, 1)
      y: (num_samples,)
    """
    X, y = [], []
    for t in range(lookback, len(series)):
        X.append(series[t-lookback:t])
        y.append(series[t])
    X = np.array(X, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.array(y, dtype=np.float32).reshape(-1,)
    return X, y

def build_gru(input_len: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.GRU(64, return_sequences=True),
        layers.GRU(32),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="mse",
                  metrics=["mae"])
    return model

# ---------------- Main ----------------
def main():
    if not DATA_PATH.exists():
        print(f"[ERROR] Missing input file: {DATA_PATH}")
        sys.exit(1)

    print("Loading features…")
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    # basic cleaning & types
    need_cols = {"hf_id", "datetime", "volume_vph"}
    missing = need_cols - set(df.columns)
    if missing:
        print(f"[ERROR] processed_data.csv missing columns: {missing}")
        sys.exit(1)

    df = df.dropna(subset=["hf_id", "datetime", "volume_vph"]).copy()
    df["hf_id"] = pd.to_numeric(df["hf_id"], errors="coerce")
    df = df.dropna(subset=["hf_id"]).copy()
    df["hf_id"] = df["hf_id"].astype(int)

    # sort + dedup
    df = df.sort_values(["hf_id", "datetime"]).drop_duplicates(["hf_id", "datetime"])

    # ---------------- Eligibility check ----------------
    print(f"  Sites (unique hf_id): {df['hf_id'].nunique()}")
    print(f"[ELIGIBILITY] MIN_HOURS_REQ={MIN_HOURS_REQ}, MIN_COVERAGE={MIN_COVERAGE:.0%}")

    eligible = []
    for hf_id, g in df.groupby("hf_id", sort=False):
        n_hours = g["datetime"].nunique()
        t_min   = g["datetime"].min()
        t_max   = g["datetime"].max()
        exp     = expected_hours_between(t_min, t_max)
        cov     = (n_hours / exp) if exp > 0 else 0.0

        ok_hours    = n_hours >= MIN_HOURS_REQ
        ok_coverage = cov >= MIN_COVERAGE
        included    = ok_hours and ok_coverage

        if included:
            eligible.append(hf_id)
        else:
            if not ok_hours and not ok_coverage:
                print(f"[SKIP] HF {hf_id}: hours {n_hours} < {MIN_HOURS_REQ} AND coverage {cov:.1%} < {MIN_COVERAGE:.0%}")
            elif not ok_hours:
                print(f"[SKIP] HF {hf_id}: hours {n_hours} < {MIN_HOURS_REQ}")
            else:
                print(f"[SKIP] HF {hf_id}: coverage {cov:.1%} < {MIN_COVERAGE:.0%}")

    eligible = set(eligible)
    print(f"[ELIGIBILITY] Eligible sites: {len(eligible)}")

    # Keep only eligible sites
    df = df[df["hf_id"].isin(eligible)].copy()

    # ---------------- Train / Test split per site ----------------
    # Strategy: per site, reserve the last TEST_HOURS for prediction (test),
    # train on everything before that (still requires at least LOOKBACK+1 rows to form windows).
    site_train_arrays = []
    site_train_targets = []
    scalers: dict[int, MinMaxScaler] = {}
    test_slices: dict[int, pd.DataFrame] = {}

    # Count how many sites remain with enough data to form at least 1 window after the split
    sites_with_windows = 0

    for hf_id, g in df.groupby("hf_id", sort=False):
        g = g.sort_values("datetime").drop_duplicates("datetime")
        # Must have enough total to support split and at least 1 training window
        if len(g) < (LOOKBACK + 1):
            print(f"[SKIP no windows] HF {hf_id}: len={len(g)} < {LOOKBACK + 1}")
            continue

        # Split into train (all but last TEST_HOURS) and test (last TEST_HOURS)
        if len(g) <= TEST_HOURS + LOOKBACK:
            # Not enough to leave a test tail + create windows
            print(f"[SKIP short after split] HF {hf_id}: len={len(g)} (need > {TEST_HOURS + LOOKBACK})")
            continue

        train_g = g.iloc[: -TEST_HOURS].copy()
        test_g  = g.iloc[-TEST_HOURS:].copy()

        # Scale per-site using ONLY train data to avoid leakage
        scaler = MinMaxScaler()
        train_vals = train_g["volume_vph"].to_numpy(dtype=np.float32).reshape(-1, 1)
        scaled_train = scaler.fit_transform(train_vals).reshape(-1)
        scalers[hf_id] = scaler

        # Build windows from scaled train series
        X, y = make_one_step_windows(scaled_train, lookback=LOOKBACK)
        if len(X) == 0:
            print(f"[SKIP no train windows] HF {hf_id}: after lookback")
            del scalers[hf_id]
            continue

        site_train_arrays.append(X)
        site_train_targets.append(y)
        test_slices[hf_id] = test_g
        sites_with_windows += 1

    if sites_with_windows == 0:
        print("[WARN] No eligible sites produced windows after split. Nothing to train.")
        # Write empty preds to keep pipeline consistent
        pd.DataFrame(columns=["hf_id", "First_Road", "datetime", "pred_vph", "true_vph"]).to_csv(PRED_PATH, index=False)
        print(f"Wrote empty predictions: {PRED_PATH}")
        sys.exit(0)

    X_train = np.concatenate(site_train_arrays, axis=0)
    y_train = np.concatenate(site_train_targets, axis=0)
    print(f"  Train sequences: {X_train.shape}")

    # ---------------- Build & train GRU ----------------
    print("Building GRU…")
    model = build_gru(input_len=LOOKBACK)

    print("Training…")
    # No formal validation here (dataset is short and split per site). You can add a small val_split if desired.
    model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=20,
        verbose=1
    )

    model.save(MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

    # ---------------- Predict last TEST_HOURS per site ----------------
    print("Predicting on test…")
    pred_rows = []

    for hf_id, test_g in test_slices.items():
        road = (test_g["First_Road"].iloc[0] if "First_Road" in test_g.columns and len(test_g) > 0 else "")
        # For 1-step-ahead predictions over last TEST_HOURS:
        # We need the final LOOKBACK context just before the test block starts.
        full_g = df[df["hf_id"] == hf_id].sort_values("datetime").drop_duplicates("datetime").copy()
        # index where test starts
        test_start_idx = len(full_g) - TEST_HOURS
        # window context ends at test_start_idx - 1
        context_start = test_start_idx - LOOKBACK
        if context_start < 0:
            print(f"[PRED skip] HF {hf_id}: not enough context for LOOKBACK")
            continue

        # Use the trained scaler for this site (fit on train only)
        scaler = scalers.get(hf_id, None)
        if scaler is None:
            print(f"[PRED skip] HF {hf_id}: no scaler")
            continue

        series = full_g["volume_vph"].to_numpy(dtype=np.float32).reshape(-1, 1)
        # Scale using train scaler (applies to entire series; test may be out-of-range—ok)
        scaled = scaler.transform(series).reshape(-1)

        # Build windows that end exactly at each test point:
        # For each test t (absolute index idx), we predict series[idx] from scaled[idx-LOOKBACK:idx]
        # i.e., 1-step ahead aligned to the true test timestamps
        X_list = []
        dt_list = []
        true_list = []
        for k in range(TEST_HOURS):
            idx = test_start_idx + k
            left = idx - LOOKBACK
            right = idx
            if left < 0:
                continue
            X_list.append(scaled[left:right])
            dt_list.append(full_g.iloc[idx]["datetime"])
            true_list.append(float(full_g.iloc[idx]["volume_vph"]))

        if len(X_list) == 0:
            print(f"[PRED skip] HF {hf_id}: no test windows")
            continue

        X_test = np.array(X_list, dtype=np.float32).reshape(-1, LOOKBACK, 1)
        yhat_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
        yhat = scaler.inverse_transform(yhat_scaled).reshape(-1)

        for dt, pred, true in zip(dt_list, yhat, true_list):
            pred_rows.append({
                "hf_id": hf_id,
                "First_Road": road,
                "datetime": pd.to_datetime(dt),
                "pred_vph": float(pred),
                "true_vph": float(true)
            })

        print(f"[PRED] HF {hf_id} ({road}): ok ({len(yhat)} preds)")

    if len(pred_rows) == 0:
        print("WARNING: No prediction rows were generated.")
        pd.DataFrame(columns=["hf_id", "First_Road", "datetime", "pred_vph", "true_vph"]).to_csv(PRED_PATH, index=False)
        print(f"Wrote empty predictions file: {PRED_PATH}")
        sys.exit(0)

    preds = pd.DataFrame(pred_rows).sort_values(["hf_id", "datetime"]).reset_index(drop=True)
    preds.to_csv(PRED_PATH, index=False)
    print(f"Wrote predictions: {PRED_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()
