from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import math

import pandas as pd
import numpy as np


# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data_raw"
ART_DIR = ROOT / "artifacts"

# make sure artifacts folder exists
ART_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Filenames
# -----------------------------
SITE_LISTING_CSV = "SCATSSiteListingSpreadsheet_VicRoads.csv"
SCATS_QH_CSV     = "Scats Data October 2006.csv"


# -----------------------------
# Quality rules
# -----------------------------
HOURS_PER_DAY = 24
QUARTERS_PER_HOUR = 4
MIN_QUARTERS_FOR_VALID_HOUR = 3     # 3 out of 4 quarters
MIN_VALID_HOURS_FOR_VALID_DAY = 19  # 19 out of 24 hours (~80%)


def make_quarter_columns() -> List[str]:
    cols = []
    for i in range(96):
        # zero-padded like V00, V01, ..., V95
        name = f"V{i:02d}"
        cols.append(name)
    return cols


# -----------------------------
# Normalize site type text
# We only keep "INT" sites (intersections).
# -----------------------------
def normalize_site_type(text: str | float | None) -> str:
    if text is None:
        return "UNKNOWN"
    if isinstance(text, float) and math.isnan(text):
        return "UNKNOWN"

    s = str(text)

    s = s.upper().strip()

    s = s.replace("ROUND A BOUT", "ROUNDABOUT")
    s = s.replace("FLASH PX", "POS")
    s = s.replace("FIRE WW", "FIRE")
    s = s.replace("FIRESIG", "FIRE")
    s = s.replace("AMBUSIG", "AMBULANCE")
    s = s.replace("BUSSIG", "BUS")
    s = s.replace("RAMPMTR", "RAMP")
    s = s.replace("O/H", "OH")
    s = s.replace("OHLANE", "HOV")

    return s


# -----------------------------
# Step 1: read site listing and keep INT sites only
# Expected columns in the CSV:
# "Site Number", "Site Type"
# -----------------------------
def load_int_site_ids(site_listing_path: Path) -> set[int]:
    if not site_listing_path.exists():
        raise FileNotFoundError(f"Missing site listing CSV at {site_listing_path}")

    # read without header because the file has intro lines above the real header
    raw = pd.read_csv(site_listing_path, header=None)

    # find the header row by looking for BOTH "Site Number" and "Site Type"
    header_row = None
    for i in range(min(40, len(raw))):
        row_vals = []
        for v in raw.iloc[i].tolist():
            # simple normalization: string, strip, collapse spaces, uppercase
            s = "" if pd.isna(v) else str(v)
            s = s.replace("\u00A0", " ").strip()
            s = " ".join(s.split())
            s = s.upper()
            row_vals.append(s)
        if ("SITE NUMBER" in row_vals) and ("SITE TYPE" in row_vals):
            header_row = i
            break

    if header_row is None:
        # show what we saw to help debug
        print("[DEBUG] Could not find a header row with 'Site Number' and 'Site Type'.")
        print("[DEBUG] First few rows:")
        print(raw.head(10).to_string(index=False, header=False))
        raise ValueError("Site listing CSV does not contain a clean header row.")

    # set that row as the header
    header_names = []
    for v in raw.iloc[header_row].tolist():
        s = "" if pd.isna(v) else str(v)
        s = s.replace("\u00A0", " ").strip()
        s = " ".join(s.split())
        header_names.append(s)
    df = raw.iloc[header_row + 1:].reset_index(drop=True)
    df.columns = header_names

    # now we expect these columns to exist
    if "Site Number" not in df.columns or "Site Type" not in df.columns:
        print("[DEBUG] Detected columns:", list(df.columns))
        raise ValueError("Site listing is missing 'Site Number' or 'Site Type' after header detection.")

    # normalize site type and keep INT only
    norm_types = []
    for val in df["Site Type"]:
        norm_types.append(normalize_site_type(val))
    df["Site Type Norm"] = norm_types
    df = df[df["Site Type Norm"] == "INT"]

    # collect integer site IDs
    int_ids: set[int] = set()
    for v in df["Site Number"]:
        try:
            num = int(pd.to_numeric(v))
            int_ids.add(num)
        except Exception:
            pass

    return int_ids



# -----------------------------
# Step 2: read SCATS 15-minute CSV
# We expect these columns:
# "SCATS Number", "Date", "V00" .. "V95"
# -----------------------------
def load_scats_qh(scats_path: Path) -> pd.DataFrame:
    if not scats_path.exists():
        raise FileNotFoundError(f"Missing SCATS CSV at {scats_path}")

    # Read raw with no header (since the real header may be lower in the file)
    raw = pd.read_csv(scats_path, header=None)

    # Simple string normaliser for comparing header cells
    def norm(x: object) -> str:
        s = "" if pd.isna(x) else str(x)
        s = s.replace("\u00A0", " ").strip()
        s = " ".join(s.split())
        return s

    # 1) Find the row that contains SITE/DATE labels
    site_keys = {"SCATS NUMBER", "SCATS SITE NUMBER", "SITE NUMBER", "SITE NO", "SCATS NO", "SCATS ID", "SID"}
    date_keys = {"DATE", "COUNT DATE", "DAY", "DATE/TIME", "DATETIME"}

    site_row_idx = None
    for i in range(min(80, len(raw))):
        vals = [norm(v).upper() for v in raw.iloc[i].tolist()]
        vals_set = set(vals)
        has_site = any(v in site_keys for v in vals_set)
        has_date = any((v in date_keys) or ("DATE" in v) for v in vals_set)
        if has_site and has_date:
            site_row_idx = i
            break

    if site_row_idx is None:
        print("[DEBUG] Could not find a header row with both site and date.")
        print("[DEBUG] First 12 rows:")
        print(raw.head(12).to_string(index=False, header=False))
        raise ValueError("SCATS CSV does not have a recognizable site/date header row.")

    # 2) Find the row that contains many TIME labels (0:00, 0.15, 00:30, etc.)
    # Accept times like "0:00", "00:00", "0.15", "12:45 AM", etc.
    import re
    time_re = re.compile(r"^\s*(\d{1,2})[:\.](\d{2})(?:\s*(AM|PM))?\s*$", re.IGNORECASE)

    def is_time_cell(txt: str) -> tuple[bool, int, int]:
        m = time_re.match(txt)
        if not m:
            return (False, -1, -1)
        hh = int(m.group(1))
        mm = int(m.group(2))
        mer = (m.group(3) or "").upper()
        # Basic validity
        if not (0 <= hh <= 23 and mm in (0, 15, 30, 45)):
            # If AM/PM present, we’ll convert 12h to 24h below
            if mm in (0, 15, 30, 45) and mer in {"AM", "PM"} and 1 <= hh <= 12:
                # convert to 24h
                hh24 = hh % 12
                if mer == "PM":
                    hh24 += 12
                return (True, hh24, mm)
            return (False, -1, -1)
        return (True, hh, mm)

    time_row_idx = None
    time_cells_positions: list[tuple[int, int, int]] = []  # (col_index, HH, MM)
    for i in range(site_row_idx, min(site_row_idx + 6, len(raw))):  # usually the next row or two
        row = [norm(v) for v in raw.iloc[i].tolist()]
        found = []
        for j, cell in enumerate(row):
            ok, hh, mm = is_time_cell(cell)
            if ok:
                found.append((j, hh, mm))
        if len(found) >= 24:  # "many" time labels on this row
            time_row_idx = i
            time_cells_positions = found
            break
    # If not found near site row, scan a bit wider
    if time_row_idx is None:
        for i in range(min(120, len(raw))):
            row = [norm(v) for v in raw.iloc[i].tolist()]
            found = []
            for j, cell in enumerate(row):
                ok, hh, mm = is_time_cell(cell)
                if ok:
                    found.append((j, hh, mm))
            if len(found) >= 24:
                time_row_idx = i
                time_cells_positions = found
                break

    if time_row_idx is None:
        print("[DEBUG] Could not find a time-label row with many quarter-hours.")
        print("[DEBUG] Site/Date header was at row index:", site_row_idx)
        raise ValueError("SCATS CSV time header row not found.")

    # 3) Build a merged header:
    # - For each column j:
    # * if time-row cell looks like a time -> map to V00..V95 (after sorting by HH:MM)
    # * else take the site/date header cell (from site_row_idx)
    site_header = [norm(v) for v in raw.iloc[site_row_idx].tolist()]
    time_header = [norm(v) for v in raw.iloc[time_row_idx].tolist()]

    # Create mapping from column index -> Vxx index
    # Sort by true time order (HH*60 + MM), then enumerate 0..95
    time_cells_positions.sort(key=lambda t: (t[1], t[2]))  # sort by (HH,MM)
    if len(time_cells_positions) != 96:
        print("[DEBUG] Detected time columns:", len(time_cells_positions), "(expected 96)")
        print("[DEBUG] Example of detected times (first 16):",
              [(c, f"{hh:02d}:{mm:02d}") for c, hh, mm in time_cells_positions[:16]])
        raise ValueError("SCATS CSV does not contain exactly 96 quarter-hour columns labeled as HH:MM.")

    idx_to_vname: dict[int, str] = {}
    for k, (col_idx, hh, mm) in enumerate(time_cells_positions):
        idx_to_vname[col_idx] = f"V{k:02d}"

    # Build final header names (one per column)
    max_cols = max(len(site_header), len(time_header))
    final_header: list[str] = []
    for j in range(max_cols):
        if j in idx_to_vname:
            final_header.append(idx_to_vname[j])
        else:
            # use site/date name (may be empty)
            name = site_header[j] if j < len(site_header) else ""
            final_header.append(name)

    # 4) Create the data frame with merged header.
    start_row = max(site_row_idx, time_row_idx) + 1
    df = raw.iloc[start_row:].reset_index(drop=True)
    # Some rows may be shorter; reindex columns to match header length
    df = df.reindex(columns=range(len(final_header)))
    df.columns = final_header

    # 5) Find site and date columns (case-insensitive)
    lower_map = {c.lower(): c for c in df.columns}

    site_col = None
    for key in ["scats number", "scats site number", "site number", "site no", "scats no", "scats id", "sid"]:
        if key in lower_map:
            site_col = lower_map[key]
            break
    if site_col is None:
        for c in df.columns:
            lc = c.lower()
            if "site" in lc and any(tok in lc for tok in ["number", "no", "id"]):
                site_col = c
                break

    date_col = None
    for key in ["date", "count date", "day", "date/time", "datetime"]:
        if key in lower_map:
            date_col = lower_map[key]
            break
    if date_col is None:
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break

    if site_col is None or date_col is None:
        print("[DEBUG] Columns after header merge (first 20):", list(df.columns)[:20])
        raise ValueError("Could not find site or date column after header merge.")

    # 6) Ensure we have all V00..V95 columns
    vcols = [f"V{i:02d}" for i in range(96)]
    missing_v = [c for c in vcols if c not in df.columns]
    if missing_v:
        print("[DEBUG] Missing V columns:", missing_v[:12], "…")
        raise ValueError("After header merge, not all V00..V95 columns are present.")

    # 7) Select and rename to canonical names
    df = df.rename(columns={site_col: "SCATS Number", date_col: "Date"})
    df = df[["SCATS Number", "Date"] + vcols]

    # 8) Drop rows without site/date
    df = df.dropna(subset=["SCATS Number", "Date"], how="any")

    return df




# -----------------------------
# Step 3: convert 15-minute to hourly for ONE row (one site, one day)
# Returns a list of tuples: (hour_datetime, hourly_total, is_valid_hour)
# -----------------------------
def hourify_day(row: pd.Series) -> List[Tuple[pd.Timestamp, float, bool]]:
    date_value = row["Date"]
    # make sure we get a datetime at midnight for the day
    day0 = pd.to_datetime(date_value).normalize()

    result = []

    # for each hour 0..23
    for hour in range(HOURS_PER_DAY):
        # quarter indices for this hour (0..95)
        start_q = hour * QUARTERS_PER_HOUR
        quarter_cols = [
            f"V{start_q:02d}",
            f"V{start_q + 1:02d}",
            f"V{start_q + 2:02d}",
            f"V{start_q + 3:02d}",
        ]

        # gather values and handle missing gracefully
        quarter_vals = []
        available_count = 0
        for c in quarter_cols:
            v = row.get(c, np.nan)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            if pd.notna(v):
                quarter_vals.append(v)
                available_count += 1

        # sum of available quarters
        if len(quarter_vals) > 0:
            hour_sum = float(np.nansum(quarter_vals))
        else:
            hour_sum = np.nan

        # decide if this hour is valid
        is_valid_hour = (available_count >= MIN_QUARTERS_FOR_VALID_HOUR)

        # the actual hour timestamp (e.g., 2006-10-01 13:00:00)
        hour_time = day0 + pd.Timedelta(hours=hour)

        # if hour is not valid, we will store NaN to make it obvious later
        if not is_valid_hour:
            hour_sum = np.nan

        result.append((hour_time, hour_sum, is_valid_hour))

    return result


# -----------------------------
# Step 4: main processing
#   - filter SCATS rows to INT sites
#   - per (site, date) aggregate to hourly
#   - apply day validity rule (>=19 valid hours)
#   - write outputs
# -----------------------------
def main():
    # 4.1 load INT site IDs
    site_listing_path = RAW_DIR / SITE_LISTING_CSV
    print("Reading site listing...")
    int_site_ids = load_int_site_ids(site_listing_path)
    print(f"  INT site IDs found: {len(int_site_ids)}")

    # 4.2 load SCATS quarter-hour table
    scats_path = RAW_DIR / SCATS_QH_CSV
    print("Reading SCATS 15-minute table...")
    scats_df = load_scats_qh(scats_path)

    # Parse dates explicitly as DD/MM/YYYY to avoid warnings & mis-parsing
    scats_df["Date"] = pd.to_datetime(scats_df["Date"], dayfirst=True, errors="coerce")
    # Drop rows where Date couldn’t parse (should be rare)
    scats_df = scats_df[pd.notna(scats_df["Date"])].copy()

    # 4.3 convert SCATS Number to integer site_id and filter to INT sites only
    site_ids = []
    for v in scats_df["SCATS Number"]:
        try:
            site_ids.append(int(pd.to_numeric(v)))
        except Exception:
            site_ids.append(np.nan)
    scats_df["site_id"] = site_ids
    scats_df = scats_df[pd.notna(scats_df["site_id"])]
    scats_df["site_id"] = scats_df["site_id"].astype(int)

    scats_df = scats_df[scats_df["site_id"].isin(int_site_ids)].copy()
    if len(scats_df) == 0:
        raise RuntimeError("After filtering to INT sites, there are no SCATS rows left. Check inputs.")

    # 4.4 aggregate to hourly with validity rules
    print("Aggregating to hourly...")
    records: List[Dict] = []   # for features_hourly.csv
    coverage: List[Dict] = []  # for coverage_report.csv

    # group by site and date so we handle one full day at a time
    # NOTE: SCATS file typically has one row per site per day already
    grouped = scats_df.groupby(["site_id", "Date"], sort=True)

    for (sid, date_val), group in grouped:
        # take the first row (there should be only one)
        row = group.iloc[0]

        # convert quarters to hourly
        hours_list = hourify_day(row)

        # count valid hours for this day
        valid_count = 0
        for (dt, sum_v, ok) in hours_list:
            if ok:
                valid_count += 1

        day_is_valid = (valid_count >= MIN_VALID_HOURS_FOR_VALID_DAY)

        # record day coverage info (even if invalid, we want to see it in the report)
        coverage.append({
            "site_id": sid,
            "date": pd.to_datetime(date_val).date(),
            "valid_hours": valid_count,
            "day_valid": day_is_valid
        })

        # only write hourly rows if the day is valid
        if day_is_valid:
            for (dt, sum_v, ok) in hours_list:
                if ok and pd.notna(sum_v):
                    records.append({
                        "site_id": sid,
                        "datetime": dt,
                        "volume_vph": float(sum_v)
                    })

    # turn lists into DataFrames
    features_df = pd.DataFrame(records)
    coverage_df = pd.DataFrame(coverage)

    # sort them nicely
    if len(features_df) > 0:
        features_df = features_df.sort_values(["site_id", "datetime"])
    if len(coverage_df) > 0:
        coverage_df = coverage_df.sort_values(["site_id", "date"])

    # 4.5 write outputs
    out_features = ART_DIR / "features_hourly.csv"
    out_coverage = ART_DIR / "coverage_report.csv"

    features_df.to_csv(out_features, index=False)
    coverage_df.to_csv(out_coverage, index=False)

    # little summary
    print("\n=== DONE ===")
    print(f"Sites in SCATS after INT filter: {scats_df['site_id'].nunique()}")
    print(f"Hourly rows written:            {len(features_df):,}")
    print(f"Coverage rows (site/day):       {len(coverage_df):,}")
    print(f"Wrote: {out_features}")
    print(f"Wrote: {out_coverage}")


if __name__ == "__main__":
    main()
