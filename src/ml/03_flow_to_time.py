# src/ml/03_flow_to_time.py
# Convert predicted flow (vph) -> speed (km/h) for all prediction files

from __future__ import annotations
from pathlib import Path
import math
import pandas as pd

# ---------------------------
# Paths (two folders up from this file)
# ---------------------------
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_PATH = ROOT / "artifacts"
MODEL_NAMES = ["gru", "lstm", "custom"]

# ---------------------------
# Flow -> Speed constants (from assignment)
# ... [A, B, CAPACITY_FLOW, SPEED_LIMIT_KMH definitions remain here]
# ---------------------------
A = -1.4648375
B = 93.75
CAPACITY_FLOW = 1500.0
SPEED_LIMIT_KMH = 60.0

# [flow_to_speed_kmh function remains the same]

def flow_to_speed_kmh(flow_vph: float) -> float:
    # ... (function body remains the same as your original)
    if flow_vph is None or not math.isfinite(flow_vph) or flow_vph <= 0:
        return SPEED_LIMIT_KMH
    D = B*B - 4*A*(-flow_vph)
    if D < 0:
        return SPEED_LIMIT_KMH
    sqrtD = math.sqrt(D)
    s1 = (-B + sqrtD) / (2*A)
    s2 = (-B - sqrtD) / (2*A)
    roots = sorted([s for s in (s1, s2) if s > 0])
    if not roots:
        return SPEED_LIMIT_KMH
    if flow_vph <= CAPACITY_FLOW:
        speed = roots[-1]
    else:
        speed = roots[0]
    if speed > SPEED_LIMIT_KMH:
        speed = SPEED_LIMIT_KMH
    if speed <= 0:
        speed = 1e-6
    return float(speed)

def process_model_predictions(model_name: str):
    pred_file = ARTIFACTS_PATH / f"{model_name}_predictions.csv"
    out_file = ARTIFACTS_PATH / f"{model_name}_flow_speed_times.csv" # Separate output file
    
    if not pred_file.exists():
        print(f"Skipping {model_name}: Predictions file not found at {pred_file}")
        return

    print(f"\n--- Processing {model_name.upper()} Predictions ---")
    df = pd.read_csv(pred_file, parse_dates=["datetime"])

    # Flexible column mapping (same logic as before)
    cols_lower = {c.lower(): c for c in df.columns}
    hf_col   = cols_lower.get("hf_id")
    road_col = cols_lower.get("first_road") or ("First_Road" if "First_Road" in df.columns else None)
    flow_col = cols_lower.get("y_pred_vph") or cols_lower.get("predicted_vph")

    if hf_col is None or road_col is None or flow_col is None or "datetime" not in df.columns:
        print(f"WARNING: {model_name}_predictions.csv missing required columns. Skipping.")
        return

    df = df.rename(columns={hf_col: "hf_id", road_col: "First_Road", flow_col: "flow_vph"})
    df = df[["hf_id", "First_Road", "datetime", "flow_vph"]].copy()

    print("Converting flow -> speed and saving …")
    df["speed_kmh"] = df["flow_vph"].apply(flow_to_speed_kmh)

    df = df.sort_values(["hf_id", "datetime"]).reset_index(drop=True)
    df.to_csv(out_file, index=False)
    print(f"Wrote {len(df):,} rows to: {out_file.name}")


def main():
    for name in MODEL_NAMES:
        process_model_predictions(name)

    print("\nNext step: Update your routing program to load the desired '_flow_speed_times.csv' file.")

if __name__ == "__main__":
    main()