# Build supervised windows for next-hour prediction.
# X = last 24 hourly volumes (scaled), y = next-hour volume (scaled).
# Output:
#   artifacts/ml/windows.npz  (X/y for train/val/test + site/time arrays)
#   artifacts/ml/scalers.csv  (per-site average/std to invert scaling later)

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
ART_ML = ART / "ml"
ART_ML.mkdir(parents=True, exist_ok=True)

FEATURES = ART / "features_hourly.csv"

WINDOW = 24

def make_site_windows(df_site: pd.DataFrame, site_id: int, w: int,
                      mean_: float, std_: float):
    x_list = []
    y_list = []
    y_time = []
    y_site = []
    vals = df_site["volume_vph"].to_numpy(dtype=float)

    # scale with given stats
    std_eff = std_ if std_ > 1e-6 else 1.0
    vals_scaled = (vals - mean_) / std_eff

    times = pd.to_datetime(df_site["datetime"]).to_numpy()

    # build rolling windows
    for t in range(w, len(vals_scaled)):
        x = vals_scaled[t-w:t]             # last w points
        y = vals_scaled[t]                 # next point
        x_list.append(x.reshape(w, 1))     # (w,1)
        y_list.append([y])                 # (1,)
        y_time.append(times[t])            # when the y occurs
        y_site.append(site_id)

    return (np.array(x_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            np.array(y_time),
            np.array(y_site, dtype=np.int32))

def split_indices(n: int, train_frac=0.7, val_frac=0.15):
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val
    return n_train, n_val, n_test

def main():
    if not FEATURES.exists():
        raise FileNotFoundError(f"Missing {FEATURES}")

    df = pd.read_csv(FEATURES)
    # clean & sort
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[pd.notna(df["datetime"])].copy()
    df = df.sort_values(["site_id", "datetime"]).reset_index(drop=True)

    # build per-site windows, but compute average/std on the TRAIN portion only
    X_train_all, y_train_all, t_train_all, s_train_all = [], [], [], []
    X_val_all,   y_val_all,   t_val_all,   s_val_all   = [], [], [], []
    X_test_all,  y_test_all,  t_test_all,  s_test_all  = [], [], [], []
    scaler_rows = []

    for site_id, g in df.groupby("site_id"):
        g = g.sort_values("datetime").reset_index(drop=True)

        # Need at least WINDOW+1 points to form one sample
        if len(g) < WINDOW + 10:
            # skip tiny series
            continue

        # preliminary scaling stats computed on TRAIN portion only
        # first split raw series into train/val/test to measure average/std on train values
        n = len(g)
        n_train, n_val, n_test = split_indices(n)
        g_train = g.iloc[:n_train]

        mean_ = float(g_train["volume_vph"].mean())
        std_  = float(g_train["volume_vph"].std(ddof=0))
        std_eff = std_ if std_ > 1e-6 else 1.0

        # save per-site scaler
        scaler_rows.append({"site_id": int(site_id), "mean": mean_, "std": std_eff})

        # Now build windows across the FULL series using those train stats
        X, y, y_time, y_site = make_site_windows(g, int(site_id), WINDOW, mean_, std_eff)

        # After windowing, split chronologically by index
        # Note: windows reduce the length by WINDOW; adjust n accordingly
        n_win = len(X)
        n_train_w, n_val_w, n_test_w = split_indices(n_win)

        X_tr = X[:n_train_w]; y_tr = y[:n_train_w]; t_tr = y_time[:n_train_w]; s_tr = y_site[:n_train_w]
        X_va = X[n_train_w:n_train_w+n_val_w]; y_va = y[n_train_w:n_train_w+n_val_w]
        t_va = y_time[n_train_w:n_train_w+n_val_w]; s_va = y_site[n_train_w:n_train_w+n_val_w]
        X_te = X[n_train_w+n_val_w:]; y_te = y[n_train_w+n_val_w:]
        t_te = y_time[n_train_w+n_val_w:]; s_te = y_site[n_train_w+n_val_w:]

        # collect
        X_train_all.append(X_tr); y_train_all.append(y_tr); t_train_all.append(t_tr); s_train_all.append(s_tr)
        X_val_all.append(X_va);   y_val_all.append(y_va);   t_val_all.append(t_va);   s_val_all.append(s_va)
        X_test_all.append(X_te);  y_test_all.append(y_te);  t_test_all.append(t_te);  s_test_all.append(s_te)

    if len(X_train_all) == 0:
        raise RuntimeError("No sites produced windows. Check features_hourly.csv contents.")

    # stack across sites
    def vcat(list_of_arrays):
        return np.concatenate(list_of_arrays, axis=0) if len(list_of_arrays) > 1 else list_of_arrays[0]

    X_train = vcat(X_train_all); y_train = vcat(y_train_all)
    X_val   = vcat(X_val_all);   y_val   = vcat(y_val_all)
    X_test  = vcat(X_test_all);  y_test  = vcat(y_test_all)

    t_train = vcat(t_train_all); s_train = vcat(s_train_all)
    t_val   = vcat(t_val_all);   s_val   = vcat(s_val_all)
    t_test  = vcat(t_test_all);  s_test  = vcat(s_test_all)

    # save arrays
    np.savez(ART_ML / "windows.npz",
             X_train=X_train, y_train=y_train,
             X_val=X_val,     y_val=y_val,
             X_test=X_test,   y_test=y_test,
             t_train=t_train.astype("datetime64[s]"),
             t_val=t_val.astype("datetime64[s]"),
             t_test=t_test.astype("datetime64[s]"),
             s_train=s_train, s_val=s_val, s_test=s_test,
             window=np.array([WINDOW], dtype=np.int32))

    # save scalers per site
    pd.DataFrame(scaler_rows).to_csv(ART_ML / "scalers.csv", index=False)

    print("Saved:", ART_ML / "windows.npz")
    print("Saved:", ART_ML / "scalers.csv")
    print("Shapes:",
          "\n X_train", X_train.shape,
          "\n y_train", y_train.shape,
          "\n X_val  ", X_val.shape,
          "\n y_val  ", y_val.shape,
          "\n X_test ", X_test.shape,
          "\n y_test ", y_test.shape)

if __name__ == "__main__":
    main()
