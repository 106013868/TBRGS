# src/ml/01_make_windows.py
# Builds training-window eligibility, scales data, and creates time-series sequences.
# Writes:
#   artifacts/scalers.joblib      (MinMaxScaler objects for each HF)
#   artifacts/train_sequences.joblib (X_train, y_train, date_index per site)
#   artifacts/test_sequences.joblib  (X_test, y_test, date_index per site)
#   artifacts/window_meta.csv        (metadata: per-HF hours, coverage, inclusion flag)

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# ---------------- Paths ----------------
IS_FROZEN = getattr(sys, "frozen", False)
if IS_FROZEN:
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parents[2]  # repository root

ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

INPUT_CSV = ART / "processed_data.csv"
META_OUT  = ART / "window_meta.csv"

# New output files needed by training scripts
SCALER_OUT = ART / "scalers.joblib"
TRAIN_SEQ_OUT = ART / "train_sequences.joblib"
TEST_SEQ_OUT = ART / "test_sequences.joblib"

# ---------------- Model/window parameters ----------------
LOOKBACK      = 168   # hours of history per sample (1 week)
HORIZON       = 24    # hours to predict (1 day)
MIN_WINDOWS   = 300   # at least this many training windows per HF
MIN_WEEKS     = 3     # at least 3 full weekly cycles

# The LSTM and GRU models were designed for a tight time split. 
TEST_SPLIT_FRACTION = 0.20 # 20% of data used for testing

MIN_COVERAGE  = 0.80  # 80% of expected hours across min..max datetime per HF

# Derived minimum hours for eligibility
REQ_BY_WINDOWS = LOOKBACK + HORIZON + MIN_WINDOWS
REQ_BY_WEEKS   = MIN_WEEKS * 168
MIN_HOURS_REQ  = max(REQ_BY_WINDOWS, REQ_BY_WEEKS)

def expected_hours_between(min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> int:
    """Inclusive number of hourly bins from min to max."""
    if pd.isna(min_ts) or pd.isna(max_ts):
        return 0
    total = int((max_ts - min_ts).total_seconds() // 3600) + 1
    return max(total, 0)

def create_windows(data: np.ndarray, index: np.ndarray, lookback: int, horizon: int):
    """Creates time-series windows (X) and labels (y) from a 1D array."""
    X, y, idx = [], [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback : i + lookback + horizon])
        # The index must be shifted by LOOKBACK to get the first predicted hour's timestamp
        idx.append(index[i + lookback]) 
    
    return np.array(X), np.array(y), np.array(idx)


def main():
    if not INPUT_CSV.exists():
        print(f"[ERROR] Missing input: {INPUT_CSV}")
        sys.exit(1)

    print("Loading processed_data.csv ...")
    df = pd.read_csv(INPUT_CSV, parse_dates=["datetime"])
    df = df.dropna(subset=["hf_id", "datetime", "volume_vph"]).copy()
    df["hf_id"] = pd.to_numeric(df["hf_id"], errors="coerce").astype(int)
    df = df.sort_values(["hf_id", "datetime"]).drop_duplicates(["hf_id", "datetime"])

    meta_rows = []
    eligible_hfs = []
    scalers_dict = {}
    train_sequences = {}
    test_sequences = {}

    print(f"  Sites (unique hf_id): {df['hf_id'].nunique()}")
    print(f"  Eligibility rule:")
    print(f"    MIN_HOURS_REQ = max(L+H+MIN_WINDOWS, MIN_WEEKS*168) = max({LOOKBACK}+{HORIZON}+{MIN_WINDOWS}, {MIN_WEEKS}*168) = {MIN_HOURS_REQ}")
    print(f"    MIN_COVERAGE  = {MIN_COVERAGE:.0%}\n")
    
    # Minimum required length for any window creation (Lookback + Horizon)
    min_len_req = LOOKBACK + HORIZON

    for hf_id, g in df.groupby("hf_id"):
        g = g.sort_values("datetime")
        n_hours = g["datetime"].nunique()
        t_min   = g["datetime"].min()
        t_max   = g["datetime"].max()
        exp_hours = expected_hours_between(t_min, t_max)
        cov = (n_hours / exp_hours) if exp_hours > 0 else 0.0

        ok_hours      = n_hours >= MIN_HOURS_REQ
        ok_coverage   = cov >= MIN_COVERAGE
        included      = bool(ok_hours and ok_coverage)
        
        # Get all relevant data for this site
        volume_data = g["volume_vph"].values.reshape(-1, 1) # Full 2D array
        full_index = g["datetime"].values # Full index array

        if not included:
            # Eligibility logging
            if not ok_hours and not ok_coverage:
                print(f"[SKIP] HF {hf_id}: hours {n_hours} < {MIN_HOURS_REQ} AND coverage {cov:.1%} < {MIN_COVERAGE:.0%}")
            elif not ok_hours:
                print(f"[SKIP] HF {hf_id}: hours {n_hours} < {MIN_HOURS_REQ}")
            else:
                print(f"[SKIP] HF {hf_id}: coverage {cov:.1%} < {MIN_COVERAGE:.0%}")
            
            # Append metadata regardless of inclusion
            meta_rows.append({
                "hf_id": hf_id, "First_Road": g["First_Road"].iloc[0], "n_hours": n_hours,
                "t_min": t_min, "t_max": t_max, "expected_hours": exp_hours,
                "coverage_ratio": round(cov, 4), "meets_hours": ok_hours,
                "meets_coverage": ok_coverage, "included": included
            })
            continue # Skip to next HF if not eligible

        # --- DATA PROCESSING FOR ELIGIBLE SITES ---
        eligible_hfs.append(hf_id)
        
        # 1. Scaling (fit on ALL data to prevent data leakage)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(volume_data)
        scalers_dict[hf_id] = scaler
        
        # 2. Train/Test Split (Time-based split)
        split_point = int(len(scaled_data) * (1 - TEST_SPLIT_FRACTION))
        
        # Data split: Train is 0 to split_point
        train_data = scaled_data[:split_point]
        train_index = full_index[:split_point]

        # Test data MUST include the lookback portion (168 hours) from the end of the train set
        # The index array for the test set MUST ALSO include the lookback portion.
        test_data_with_lookback = scaled_data[split_point - LOOKBACK:] 
        test_index_with_lookback = full_index[split_point - LOOKBACK:]
        
        # 3. Window Creation (Safe check applied)
        
        # Train sequences
        if len(train_data) >= min_len_req:
            X_train, y_train, idx_train = create_windows(train_data, train_index, LOOKBACK, HORIZON)
            if len(X_train) > 0:
                train_sequences[hf_id] = (X_train, y_train, idx_train)

        # Test sequences
        if len(test_data_with_lookback) >= min_len_req:
            # We use the full test_data_with_lookback array and index for window creation
            X_test, y_test, idx_test = create_windows(test_data_with_lookback, test_index_with_lookback, LOOKBACK, HORIZON)
            if len(X_test) > 0:
                test_sequences[hf_id] = (X_test, y_test, idx_test)
                
        # Append metadata
        meta_rows.append({
            "hf_id": hf_id, "First_Road": g["First_Road"].iloc[0], "n_hours": n_hours,
            "t_min": t_min, "t_max": t_max, "expected_hours": exp_hours,
            "coverage_ratio": round(cov, 4), "meets_hours": ok_hours,
            "meets_coverage": ok_coverage, "included": included
        })

    # --- FINAL WRITE ---
    
    # Write metadata
    meta_df = pd.DataFrame(meta_rows).sort_values(["included", "n_hours"], ascending=[False, False])
    meta_df.to_csv(META_OUT, index=False)
    print(f"\n[INFO] Eligibility summary written to: {META_OUT}")
    print(f"[INFO] Eligible sites: {len(eligible_hfs)}")

    # Write joblib files
    joblib.dump(scalers_dict, SCALER_OUT)
    joblib.dump(train_sequences, TRAIN_SEQ_OUT)
    joblib.dump(test_sequences, TEST_SEQ_OUT)
    
    print(f"[OK] Wrote: {SCALER_OUT}")
    print(f"[OK] Wrote: {TRAIN_SEQ_OUT}")
    print(f"[OK] Wrote: {TEST_SEQ_OUT}")
    
    print("Done.")

if __name__ == "__main__":
    main()