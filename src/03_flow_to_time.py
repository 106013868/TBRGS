# src/ml/03_flow_to_time.py
# Convert predicted flow (vph) -> speed (km/h)
# Input : artifacts/gru_predictions.csv
# Output: artifacts/flow_speed_times.csv

from __future__ import annotations
from pathlib import Path
import math
import pandas as pd

# ---------------------------
# Paths (two folders up from this file)
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "artifacts" / "gru_predictions.csv"
OUT_PATH  = ROOT / "artifacts" / "flow_speed_times.csv"

# ---------------------------
# Flow -> Speed constants (from assignment)
# flow = A*speed^2 + B*speed
# Capacity about 1500 vph at ~32 km/h
# ---------------------------
A = -1.4648375
B = 93.75
CAPACITY_FLOW = 1500.0
SPEED_LIMIT_KMH = 60.0

def flow_to_speed_kmh(flow_vph: float) -> float:
    """
    Convert flow (vehicles/hour) to speed (km/h) using the quadratic formula.
    Selects the higher-speed (green) branch for flow <= capacity,
    and the lower-speed (red) branch for flow > capacity.
    """
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

def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PRED_PATH}")

    print("Loading predictions…")
    df = pd.read_csv(PRED_PATH, parse_dates=["datetime"])

    # Flexible column mapping
    cols_lower = {c.lower(): c for c in df.columns}
    hf_col   = cols_lower.get("hf_id")
    road_col = cols_lower.get("first_road") or ("First_Road" if "First_Road" in df.columns else None)
    flow_col = cols_lower.get("y_pred_vph") or cols_lower.get("predicted_vph")

    if hf_col is None or road_col is None or flow_col is None or "datetime" not in df.columns:
        raise ValueError(
            "gru_predictions.csv missing required columns. "
            f"Found: {list(df.columns)}; need hf_id, First_Road/first_road, datetime, and y_pred_vph/predicted_vph."
        )

    df = df.rename(columns={hf_col: "hf_id", road_col: "First_Road", flow_col: "flow_vph"})
    df = df[["hf_id", "First_Road", "datetime", "flow_vph"]].copy()

    print("Converting flow -> speed …")
    df["speed_kmh"] = df["flow_vph"].apply(flow_to_speed_kmh)

    df = df.sort_values(["hf_id", "datetime"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("\n=== DONE ===")
    print(f"Rows written: {len(df):,}")
    print(f"Wrote: {OUT_PATH}")
    print("\nNext step:")
    print("  Once graph edges are ready, join on hf_id & datetime, then compute travel times per edge using distance and intersection delay.")

if __name__ == "__main__":
    main()
