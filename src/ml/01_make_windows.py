# src/ml/01_make_windows.py
# Builds training windows eligibility using a dynamic rule and logs reasons for exclusion.
# Writes:
#   artifacts/windows_train.npz   (placeholder arrays – kept for compatibility/gating)
#   artifacts/window_sites.csv    (metadata: per-HF hours, coverage, included)

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ---------------- Paths ----------------
IS_FROZEN = getattr(sys, "frozen", False)
if IS_FROZEN:
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parents[2]  # <repo root>

ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

INPUT_CSV = ART / "processed_data.csv"          # <== renamed from features_hourly.csv
NPZ_OUT   = ART / "windows_train.npz"           # GUI Step 2 gating looks for this
META_OUT  = ART / "window_meta.csv"

# ---------------- Model/window parameters ----------------
LOOKBACK      = 168   # hours of history per sample (1 week)
HORIZON       = 24    # hours to predict (1 day) – not used for building here, but for eligibility math
MIN_WINDOWS   = 300   # want at least this many training windows per HF
MIN_WEEKS     = 3     # want at least 3 full weekly cycles
MIN_COVERAGE  = 0.80  # 80% of expected hours across min..max datetime per HF

# Derived minimum hours
REQ_BY_WINDOWS = LOOKBACK + HORIZON + MIN_WINDOWS     # L + H + K
REQ_BY_WEEKS   = MIN_WEEKS * 168
MIN_HOURS_REQ  = max(REQ_BY_WINDOWS, REQ_BY_WEEKS)

def expected_hours_between(min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> int:
    """Inclusive number of hourly bins from min to max."""
    if pd.isna(min_ts) or pd.isna(max_ts):
        return 0
    total = int((max_ts - min_ts).total_seconds() // 3600) + 1
    return max(total, 0)

def main():
    if not INPUT_CSV.exists():
        print(f"[ERROR] Missing input: {INPUT_CSV}")
        sys.exit(1)

    print("Loading processed_data.csv …")
    df = pd.read_csv(INPUT_CSV, parse_dates=["datetime"])
    # Basic cleaning
    df = df.dropna(subset=["hf_id", "datetime", "volume_vph"]).copy()
    # Enforce types
    df["hf_id"] = pd.to_numeric(df["hf_id"], errors="coerce")
    df = df.dropna(subset=["hf_id"]).copy()
    df["hf_id"] = df["hf_id"].astype(int)
    # Sort + de-dup on (hf_id, datetime)
    df = df.sort_values(["hf_id", "datetime"]).drop_duplicates(["hf_id", "datetime"])

    # Summaries per site
    meta_rows = []
    eligible_hfs = []

    print(f"  Sites (unique hf_id): {df['hf_id'].nunique()}")
    print(f"  Eligibility rule:")
    print(f"    MIN_HOURS_REQ = max(L+H+MIN_WINDOWS, MIN_WEEKS*168) = "
          f"max({LOOKBACK}+{HORIZON}+{MIN_WINDOWS}, {MIN_WEEKS}*168) = {MIN_HOURS_REQ}")
    print(f"    MIN_COVERAGE  = {MIN_COVERAGE:.0%}\n")

    for hf_id, g in df.groupby("hf_id"):
        g = g.sort_values("datetime")
        n_hours = g["datetime"].nunique()
        t_min   = g["datetime"].min()
        t_max   = g["datetime"].max()
        exp_hours = expected_hours_between(t_min, t_max)
        cov = (n_hours / exp_hours) if exp_hours > 0 else 0.0

        # Decide eligibility
        ok_hours    = n_hours >= MIN_HOURS_REQ
        ok_coverage = cov >= MIN_COVERAGE
        included    = bool(ok_hours and ok_coverage)

        meta_rows.append({
            "hf_id": hf_id,
            "First_Road": g["First_Road"].iloc[0] if "First_Road" in g.columns and len(g) > 0 else "",
            "n_hours": n_hours,
            "t_min": t_min,
            "t_max": t_max,
            "expected_hours": exp_hours,
            "coverage_ratio": round(cov, 4),
            "meets_hours": ok_hours,
            "meets_coverage": ok_coverage,
            "included": included
        })

        if not included:
            # Print reason for exclusion (clear + specific)
            if not ok_hours and not ok_coverage:
                print(f"[SKIP] HF {hf_id}: hours {n_hours} < {MIN_HOURS_REQ} AND coverage {cov:.1%} < {MIN_COVERAGE:.0%}")
            elif not ok_hours:
                print(f"[SKIP] HF {hf_id}: hours {n_hours} < {MIN_HOURS_REQ}")
            else:
                print(f"[SKIP] HF {hf_id}: coverage {cov:.1%} < {MIN_COVERAGE:.0%}")
        else:
            eligible_hfs.append(hf_id)

    # Save metadata for inspection
    meta_df = pd.DataFrame(meta_rows).sort_values(["included", "n_hours", "coverage_ratio", "hf_id"],
                                                   ascending=[False, False, False, True])
    meta_df.to_csv(META_OUT, index=False)
    print(f"\n[INFO] Eligibility summary written to: {META_OUT}")
    print(f"[INFO] Eligible sites: {len(eligible_hfs)}")

    # ---- Windows output (placeholder) ----
    # Your GRU trainer builds windows itself, but the GUI uses this file as Step 2 “done” signal.
    # We store minimal info so it’s harmless if not explicitly consumed by Step 3.
    # If you *do* want to consume it later, extend here to actually build X/y arrays.
    np.savez_compressed(
        NPZ_OUT,
        eligible_hf_ids=np.array(eligible_hfs, dtype=np.int32),
        lookback=np.int32(LOOKBACK),
        horizon=np.int32(HORIZON),
        min_hours_req=np.int32(MIN_HOURS_REQ),
        min_coverage=np.float32(MIN_COVERAGE),
    )
    print(f"[OK] Wrote: {NPZ_OUT}")

    # Friendly recap for the console
    if len(eligible_hfs) == 0:
        print("\n[WARN] No eligible sites under current thresholds.")
        print("       Consider relaxing MIN_WINDOWS or MIN_COVERAGE if your dataset is sparse.")
    else:
        # Show a couple of examples
        ex = meta_df[meta_df["included"]].head(5)
        print("\n[INFO] Example eligible sites:")
        for _, r in ex.iterrows():
            print(f"  HF {int(r['hf_id'])} ({r['First_Road']}): "
                  f"{int(r['n_hours'])}h, cov={r['coverage_ratio']:.1%}")

if __name__ == "__main__":
    main()
