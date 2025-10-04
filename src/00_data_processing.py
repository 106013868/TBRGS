from pathlib import Path
import pandas as pd
import numpy as np
import re

# Paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

SITE_LISTING_CSV = RAW / "SCATSSiteListingSpreadsheet_VicRoads.csv"
SCATS_CSV        = RAW / "Scats Data October 2006.csv"

# Methods
def norm_cell(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\u00A0", " ").strip()
    return " ".join(s.split())

def normalize_site_type(s: object) -> str:
    if pd.isna(s):
        return ""
    t = str(s).upper().strip()
    t = t.replace("FLASH PX", "POS")
    return t

def filter_int(t: str) -> bool:
    return t == "INT"

def extract_first_road(text: object) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    s = text.upper().replace("_", " ").replace("-", " ").replace("\\", "/")
    s = s.replace(".", " ")
    s = " ".join(s.split())

    # Cut off at direction/location phrases (N OF, SW OF, etc.)
    cut = re.search(
        r"\b(?:AT|OPP|NEAR|BETWEEN|INTERSECTION|"
        r"(?:N|S|E|W|NE|NW|SE|SW)\s+OF)\b",
        s
    )
    if cut:
        s = s[:cut.start()].strip()

    # Also stop at segment separators (include AND)
    for sep in ["/", "&", " AND "]:
        if sep in s:
            s = s.split(sep)[0].strip()
            break

    tokens = re.findall(r"[A-Z0-9]+", s)
    if not tokens:
        return ""

    # Recognized road types
    TYPE_MAP = {
        "ROAD":"RD", "RD":"RD",
        "STREET":"ST", "ST":"ST",
        "HIGHWAY":"HWY", "HWY":"HWY",
        "FREEWAY":"FWY", "FWY":"FWY",
        "AVENUE":"AVE", "AVE":"AVE",
        "DRIVE":"DR", "DR":"DR",
        "CRESCENT":"CRES", "CRES":"CRES",
        "BOULEVARD":"BLVD", "BLVD":"BLVD",
        "LANE":"LN", "LN":"LN",
        "MOTORWAY":"MWY", "MWY":"MWY",
        "PARKWAY":"PKWY", "PKWY":"PKWY",
        "GROVE":"GV", "GV":"GV",
        "ARTERIAL":"ARTERIAL" 
    }
    TYPE_SET = set(TYPE_MAP.keys())

    if "ARTERIAL" in tokens[:6]:
        idx = tokens.index("ARTERIAL")
        base_tokens = tokens[:idx] or tokens[:1]
        # Collapse initials like ['S','E'] -> 'SE'
        if all(len(t) == 1 for t in base_tokens) and len(base_tokens) <= 3:
            base = "".join(base_tokens)
        else:
            base = " ".join(base_tokens)
        return f"{base} ARTERIAL".strip()

    type_idx = -1
    for i, t in enumerate(tokens[:6]):
        if t in TYPE_SET:
            type_idx = i
    if type_idx != -1:
        base_tokens = tokens[:type_idx] or tokens[:1]
        base = " ".join(base_tokens)  # keep multi-word bases intact
        final_type = TYPE_MAP[tokens[type_idx]]
        return f"{base} {final_type}".strip()

    # Defaults if no explicit type found
    DEFAULT = {
        "WARRIGAL":"RD","BURWOOD":"HWY","BURKE":"RD","CANTERBURY":"RD","BALWYN":"RD",
        "MAROONDAH":"HWY","EASTERN":"FWY","MONASH":"FWY","RIVERSDALE":"RD",
        "DONCASTER":"RD","CAMBERWELL":"RD","WHITEHORSE":"RD","GLENFERRIE":"RD",
        "BARKERS":"RD","COTHAM":"RD","TOORAK":"RD","STUD":"RD","STATION":"ST",
        "HIGH STREET":"RD"
    }
    if len(tokens) >= 2:
        two = f"{tokens[0]} {tokens[1]}"
        if two in DEFAULT:
            return f"{two} {DEFAULT[two]}"
    base = tokens[0]
    if base in DEFAULT:
        return f"{base} {DEFAULT[base]}"
    return base


def load_site_listing_int_scats() -> set[int]:
    raw = pd.read_csv(SITE_LISTING_CSV, header=None)

    # detect header row
    header_row = None
    for i in range(min(100, len(raw))):
        vals = [norm_cell(v).upper() for v in raw.iloc[i].tolist()]
        if "SITE NUMBER" in vals and "SITE TYPE" in vals and ("LOCATION" in vals or "LOCATION DESCRIPTION" in vals):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find header row in site listing CSV.")

    header = [norm_cell(v) for v in raw.iloc[header_row].tolist()]
    df = raw.iloc[header_row+1:].reset_index(drop=True)
    df.columns = header

    lower = {c.lower(): c for c in df.columns}
    site_col = lower.get("site number")
    type_col = lower.get("site type")

    if not site_col or not type_col:
        raise ValueError(f"Missing 'Site Number' or 'Site Type'. Found: {list(df.columns)}")

    df["Site Type"] = df[type_col].apply(normalize_site_type)
    df = df[df["Site Type"].apply(filter_int)].copy()

    df["SCATS Number"] = pd.to_numeric(df[site_col], errors="coerce")
    df = df[pd.notna(df["SCATS Number"])].copy()
    df["SCATS Number"] = df["SCATS Number"].astype(int)

    scats_set = set(df["SCATS Number"].tolist())
    print(f"  INT-only SCATS entries in listing: {len(scats_set)}")
    return scats_set

# Load SCATS Oct 2006 → HF Number + SCATS Number + Date + Location + V00..V95
def load_scats_qh_with_location() -> pd.DataFrame:
    if not SCATS_CSV.exists():
        raise FileNotFoundError(f"Missing {SCATS_CSV}")

    df = pd.read_csv(SCATS_CSV, header=1)

    required = ["SCATS Number", "HF VicRoads Internal", "Date", "Location"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Found: {list(df.columns)}")

    # rename
    df = df.rename(columns={"HF VicRoads Internal": "HF Number"})

    # vcols are V00..V95 in this file
    vcols = [f"V{i:02d}" for i in range(96)]
    miss = [c for c in vcols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing expected quarter-hour columns: {miss[:8]}...")

    # coerce types
    df["HF Number"] = pd.to_numeric(df["HF Number"], errors="coerce")
    df["SCATS Number"] = pd.to_numeric(df["SCATS Number"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    df = df[pd.notna(df["HF Number"]) & pd.notna(df["SCATS Number"]) & pd.notna(df["Date"])].copy()
    df = df[df["HF Number"] > 0].copy()
    df["HF Number"] = df["HF Number"].astype(int)
    df["SCATS Number"] = df["SCATS Number"].astype(int)

    # clean 15-min numeric
    df[vcols] = (
        df[vcols]
          .replace(r"[^\d\.]+", "", regex=True)
          .replace("", np.nan)
          .apply(pd.to_numeric, errors="coerce")
    )

    # extract First_Road directly from Location (per row)
    df["First_Road_raw"] = df["Location"].apply(extract_first_road)

    return df[["HF Number","SCATS Number","Date","First_Road_raw"] + vcols].reset_index(drop=True)

# Sum 96 quarters -> 24 hours
def hourify_day(row: pd.Series, vcols: list[str]) -> list[float]:
    vals = row[vcols].to_numpy(dtype=float)
    if vals.size < 96:
        vals = np.pad(vals, (0, 96 - vals.size), constant_values=np.nan)
    return [np.nansum(vals[i:i+4]) for i in range(0, 96, 4)]

# Main
def main():
    print("Loading SCATS Site Listing (INT only)…")
    int_scats_set = load_site_listing_int_scats()  # for filtering only

    print("Loading SCATS Oct-2006 (HF-based, with Location)…")
    scats = load_scats_qh_with_location()

    # keep only rows whose SCATS Number is INT
    before = len(scats)
    scats = scats[scats["SCATS Number"].isin(int_scats_set)].copy()
    print(f"  SCATS rows after INT filter: {len(scats)} (was {before})")

    # Build HF -> First_Road mapping from the Location column:
    # use the most frequent (mode) First_Road_raw per HF
    hf_first = (
        scats.groupby("HF Number")["First_Road_raw"]
             .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else (s.dropna().iloc[0] if not s.dropna().empty else ""))
             .reset_index()
             .rename(columns={"First_Road_raw": "First_Road"})
    )

    # Aggregate to hourly
    vcols = [f"V{i:02d}" for i in range(96)]
    hourly_rows = []
    coverage_rows = []

    for _, row in scats.iterrows():
        hf_id = int(row["HF Number"])
        day = row["Date"].normalize()
        hours = hourify_day(row, vcols)
        valid_hours = sum(1 for v in hours if pd.notna(v))

        for h, val in enumerate(hours):
            hourly_rows.append({
                "hf_id": hf_id,
                "datetime": day + pd.Timedelta(hours=h),
                "volume_vph": float(val) if pd.notna(val) else 0.0
            })

        coverage_rows.append({
            "hf_id": hf_id,
            "date": day.date(),
            "valid_hours": int(valid_hours),
            "day_valid": bool(valid_hours >= 19)
        })

    features_df = pd.DataFrame(hourly_rows)
    coverage_df = pd.DataFrame(coverage_rows)

    # site_labels: HF + First_Road
    site_labels = hf_first.rename(columns={"HF Number": "hf_id"})

    # attach First_Road to features via HF mapping
    features_df = features_df.merge(hf_first, left_on="hf_id", right_on="HF Number", how="left")
    features_df.drop(columns=["HF Number"], inplace=True)

    # --- NEW: put First_Road immediately after hf_id ---
    wanted = ["hf_id", "First_Road", "datetime", "volume_vph"]
    features_df = features_df[wanted]

    # write outputs
    features_out = ART / "processed_data.csv"
    coverage_out = ART / "coverage_report.csv"
    labels_out   = ART / "site_labels.csv"

    features_df.to_csv(features_out, index=False)
    coverage_df.to_csv(coverage_out, index=False)
    site_labels.to_csv(labels_out, index=False)

    # small summary
    total_labels = len(site_labels)
    missing_labels = int(site_labels["First_Road"].isna().sum() + (site_labels["First_Road"] == "").sum())
    print("\n=== DONE ===")
    print(f"HF sites in features: {features_df['hf_id'].nunique()}")
    print(f"Hourly rows written:  {len(features_df):,}")
    print(f"Coverage rows:        {len(coverage_df):,}")
    print(f"Site labels:          {total_labels} (blank First_Road: {missing_labels})")
    print("Wrote:", features_out)
    print("Wrote:", coverage_out)
    print("Wrote:", labels_out)

if __name__ == "__main__":
    main()
