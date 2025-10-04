from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# -------------------
# Config / paths
# -------------------
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

FEATURES = ART / "processed_data.csv"
META_OUT = ART / "window_meta.csv"
TRAIN_NPZ = ART / "train.npz"
VAL_NPZ   = ART / "val.npz"
TEST_NPZ  = ART / "test.npz"

# windowing
INPUT_STEPS   = 24 * 7    # 7 days history
FORECAST_HRS  = 1         # 1-step ahead
MIN_SERIES_HR = 24 * 30   # at least 30 days per site

# splits by time (last N days are test; last M of train are val)
TEST_DAYS = 7
VAL_DAYS  = 3

# -------------------
# Helpers
# -------------------
def make_continuous_hourly(df_site: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure one row per hour per site.
    - Collapse duplicate datetimes (some sites have repeats).
    - Reindex to continuous hourly grid.
    - Fill small gaps by interpolation.
    """
    # 1) sort
    df_site = df_site.sort_values("datetime").reset_index(drop=True)

    # 2) collapse duplicates at the same hour
    #    choose aggregation that makes sense for your use (mean or sum):
    agg = (
        df_site.groupby("datetime", as_index=True)
               .agg({
                   "hf_id": "first",          # ID doesn't change within site
                   "First_Road": "first",     # keep label
                   "volume_vph": "mean"       # or "sum" if you prefer
               })
               .reset_index()
    )

    # 3) build continuous hourly index
    start = agg["datetime"].min()
    end   = agg["datetime"].max()
    full_idx = pd.date_range(start=start, end=end, freq="h")  # use "h" (lowercase)

    # 4) reindex to the continuous grid
    out = (
        agg.set_index("datetime")
           .reindex(full_idx)
           .rename_axis("datetime")
           .reset_index()
    )

    # 5) forward/backward fill static columns; interpolate volume
    out["hf_id"] = out["hf_id"].ffill().bfill()
    out["First_Road"] = out["First_Road"].ffill().bfill()
    out["volume_vph"] = out["volume_vph"].interpolate(limit_direction="both")

    return out

def build_sequences(values: np.ndarray,
                    timestamps: np.ndarray,
                    input_steps: int,
                    forecast_hrs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding windows:
      X: (n, T, 1) past T points
      y: (n,) next 'forecast_hrs' sum (1-hr ahead here)
      ts: (n,) timestamps of prediction target
    """
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

# -------------------
# Main
# -------------------
def main():
    print("Loading features_hourly.csv …")
    df = pd.read_csv(FEATURES, parse_dates=["datetime"])
    # Expect: ["hf_id","First_Road","datetime","volume_vph"]
    need = {"hf_id","First_Road","datetime","volume_vph"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in features file: {missing}")

    df = df.sort_values(["hf_id","datetime"]).reset_index(drop=True)

    # filter sites with enough data
    counts = df.groupby("hf_id")["datetime"].count()
    enough_sites = set(counts[counts >= MIN_SERIES_HR].index.tolist())
    df = df[df["hf_id"].isin(enough_sites)].copy()
    print(f"  Sites with >= {MIN_SERIES_HR} hours: {len(enough_sites)}")

    # split by time
    last_time = df["datetime"].max()
    test_start = last_time - pd.Timedelta(days=TEST_DAYS) + pd.Timedelta(hours=1)
    train_df = df[df["datetime"] < test_start].copy()
    test_df  = df[df["datetime"] >= test_start].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("Train or Test split is empty. Adjust TEST_DAYS.")

    val_cut = train_df["datetime"].max() - pd.Timedelta(days=VAL_DAYS)
    val_df = train_df[train_df["datetime"] >= val_cut].copy()

    # per-site scaler store (mean, std)
    scalers: Dict[int, Tuple[float,float]] = {}
    meta_rows: List[dict] = []

    # containers
    X_tr_list, y_tr_list = [], []
    X_va_list, y_va_list = [], []
    X_te_list, y_te_list, te_info = [], [], []  # te_info keeps (hf_id, First_Road, ts)

    # ---- build TRAIN (fit scaler), VALID and TEST (reuse scaler) per site ----
    # map for road names
    road_map = df[["hf_id","First_Road"]].drop_duplicates().set_index("hf_id")["First_Road"].to_dict()

    for split_name, subdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"Building {split_name} windows …")
        for hf, g in subdf.groupby("hf_id", sort=True):
            g_full = make_continuous_hourly(g)
            values = g_full["volume_vph"].to_numpy(dtype=float)
            ts     = g_full["datetime"].to_numpy()

            if split_name == "train":
                mean = float(np.nanmean(values))
                std  = float(np.nanstd(values))
                if std < 1e-6:
                    std = 1.0
                scalers[int(hf)] = (mean, std)
                # store meta once per site
                meta_rows.append({
                    "hf_id": int(hf),
                    "First_Road": road_map.get(int(hf), ""),
                    "mean": mean,
                    "std": std
                })
            else:
                if int(hf) not in scalers:
                    # site not seen in train (shouldn't happen with our split)
                    continue
                mean, std = scalers[int(hf)]

            # scale
            values = (values - mean) / std

            X, y, tgt_ts = build_sequences(values, ts, INPUT_STEPS, FORECAST_HRS)
            if len(X) == 0:
                continue

            if split_name == "train":
                X_tr_list.append(X); y_tr_list.append(y)
            elif split_name == "val":
                X_va_list.append(X); y_va_list.append(y)
            else:
                X_te_list.append(X); y_te_list.append(y)
                # keep info to reconstruct (hf_id, First_Road, datetime) later if needed
                road = road_map.get(int(hf), "")
                te_info.extend([(int(hf), road, t) for t in tgt_ts])

    # concat and save
    def concat_or_empty(lst, axis=0):
        return np.concatenate(lst, axis=axis) if lst else np.zeros((0, INPUT_STEPS, 1))

    X_train = concat_or_empty(X_tr_list)
    y_train = np.concatenate(y_tr_list, axis=0) if y_tr_list else np.zeros((0,))
    X_val   = concat_or_empty(X_va_list)
    y_val   = np.concatenate(y_va_list, axis=0) if y_va_list else np.zeros((0,))
    X_test  = concat_or_empty(X_te_list)
    y_test  = np.concatenate(y_te_list, axis=0) if y_te_list else np.zeros((0,))

    print(f"Shapes -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # save .npz
    np.savez_compressed(TRAIN_NPZ, X=X_train, y=y_train)
    np.savez_compressed(VAL_NPZ,   X=X_val,   y=y_val)
    np.savez_compressed(TEST_NPZ,  X=X_test,  y=y_test)

    # save meta with road names and scalers
    meta_df = pd.DataFrame(meta_rows).sort_values("hf_id").reset_index(drop=True)
    meta_df.to_csv(META_OUT, index=False)

    print("Wrote:")
    print(" ", TRAIN_NPZ)
    print(" ", VAL_NPZ)
    print(" ", TEST_NPZ)
    print(" ", META_OUT)

if __name__ == "__main__":
    main()
