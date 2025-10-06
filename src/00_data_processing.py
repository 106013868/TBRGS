from __future__ import annotations
from pathlib import Path
import re
import math
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data_raw"
ART  = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

SCATS_QH_CSV = RAW / "Scats Data October 2006.csv"

def norm_cell(x) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.replace("\u00A0", " ").strip()
    s = " ".join(s.split())
    return s

TIME_RE = re.compile(r"^\s*(\d{1,2})[:\.](\d{2})(?:\s*(AM|PM))?\s*$", re.IGNORECASE)

def parse_time_cell(txt: str):
    m = TIME_RE.match(txt or "")
    if not m: return (False, -1, -1)
    hh = int(m.group(1)); mm = int(m.group(2))
    mer = (m.group(3) or "").upper()
    if mer in {"AM","PM"}:
        if not (1 <= hh <= 12): return (False,-1,-1)
        hh24 = hh % 12
        if mer == "PM": hh24 += 12
        return ((mm in (0,15,30,45)), hh24, mm) if (mm in (0,15,30,45)) else (False,-1,-1)
    if 0 <= hh <= 23 and mm in (0,15,30,45):
        return (True, hh, mm)
    return (False, -1, -1)

def extract_first_road(text) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = str(text).upper()
    s = s.replace("\u00A0", " ")
    s = s.replace("_", " ").replace("-", " ")
    s = s.replace("\\", "/")
    for sep in ("/", "&"):
        if sep in s:
            s = s.split(sep)[0].strip()
            break
    tokens = re.findall(r"[A-Z0-9\.]+", s)
    tokens = [t.replace(".", "") for t in tokens]
    if not tokens: return ""
    NOISE = {"N","S","E","W","NE","NW","SE","SW","OF","NEAR","OPP","OUTSIDE","AT","BETWEEN",
             "JNC","JUNCTION","INTERCHANGE","BEFORE","AFTER"}
    tokens = [t for t in tokens if t not in NOISE]
    if not tokens: return ""
    TYPE_MAP = {
        "ROAD":"RD","RD":"RD","HIGHWAY":"HWY","HWY":"HWY","FREEWAY":"FWY","FWY":"FWY",
        "STREET":"ST","ST":"ST","AVENUE":"AVE","AVE":"AVE","DRIVE":"DR","DR":"DR",
        "CRESCENT":"CRES","CRES":"CRES","BOULEVARD":"BLVD","BLVD":"BLVD",
        "LANE":"LANE","LN":"LANE","MOTORWAY":"MWY","MWY":"MWY","PKWY":"PKWY","PARKWAY":"PKWY"
    }
    TYPE_SET = set(TYPE_MAP.keys())
    phrase = " ".join(tokens)
    if phrase.startswith("S E ARTERIAL"):
        return "S E ARTERIAL"
    if len(tokens) >= 2 and tokens[1] in TYPE_SET:
        return f"{tokens[0]} {TYPE_MAP[tokens[1]]}"
    for i in range(1, min(len(tokens), 6)):
        if tokens[i] in TYPE_SET:
            base = tokens[i-1]
            return f"{base} {TYPE_MAP[tokens[i]]}"
    DEFAULT_TYPE_BY_BASE = {
        "WARRIGAL":"RD","BURKE":"RD","COTHAM":"RD","DONCASTER":"RD","BALWYN":"RD","BARKERS":"RD",
        "CANTERBURY":"RD","GLENFERRIE":"RD","RIVERSDALE":"RD","WHITEHORSE":"RD","HIGH":"ST","SWAN":"ST",
        "MAROONDAH":"HWY","CHANDLER":"HWY","EASTERN":"FWY","MONASH":"FWY","BURWOOD":"RD","CAMBERWELL":"RD",
        "VICTORIA":"ST","TOORAK":"RD","DENMARK":"ST"
    }
    base = tokens[0]
    t = DEFAULT_TYPE_BY_BASE.get(base)
    if t: return f"{base} {t}"
    return base

def _fast_path_if_ready_header() -> pd.DataFrame | None:
    """
    Attempt reading with header=0. If HF/Date and all V00..V95 are present, return the frame; otherwise None.
    """
    try:
        df0 = pd.read_csv(SCATS_QH_CSV, header=0, dtype=str)
    except Exception:
        return None
    cols = [norm_cell(c) for c in df0.columns]
    have_hf   = any(c.lower().startswith("hf") for c in cols)
    have_date = any("date" in c.lower() for c in cols)
    have_v    = all((f"V{i:02d}" in cols) for i in range(96))
    if have_hf and have_date and have_v:
        # Standardise names
        colmap = {}
        for c in df0.columns:
            cl = c.lower()
            if cl.startswith("hf"):
                colmap[c] = "HF"
            elif "date" in cl:
                colmap[c] = "Date"
            elif c in {f"V{i:02d}" for i in range(96)}:
                colmap[c] = c
            elif "location" in cl:
                colmap[c] = "Location"
        df0 = df0.rename(columns=colmap)
        # Keep only relevant columns if present
        keep = ["HF","Date"] + (["Location"] if "Location" in df0.columns else []) + [f"V{i:02d}" for i in range(96)]
        keep = [k for k in keep if k in df0.columns]
        return df0[keep].copy()
    return None

def load_scats_qh_use_hf() -> pd.DataFrame:
    if not SCATS_QH_CSV.exists():
        raise FileNotFoundError(f"Missing {SCATS_QH_CSV}")

    # Fast path (already has header)
    ready = _fast_path_if_ready_header()
    if ready is not None:
        return ready

    # Fallback: scan to infer headers and quarter-hour columns
    raw = pd.read_csv(SCATS_QH_CSV, header=None, dtype=str)

    # Find a row that looks like the site row (HF + Date somewhere)
    site_row = None
    for i in range(min(200, len(raw))):
        vals = [norm_cell(v).upper() for v in raw.iloc[i].tolist()]
        if any(v.startswith("HF") for v in vals) and any("DATE" in v for v in vals):
            site_row = i
            break

    # Find a row that looks like the time row (either many 00:15 labels OR many Vxx)
    time_row = None
    time_cells = []
    vnames_row = None
    for i in range(min(200, len(raw))):
        row = [norm_cell(v) for v in raw.iloc[i].tolist()]
        # Option A: raw times such as 0:00, 0:15…
        found = []
        for j, cell in enumerate(row):
            ok, hh, mm = parse_time_cell(cell)
            if ok: found.append((j, hh, mm))
        if len(found) >= 24:
            time_row = i
            time_cells = found
            break
        # Option B: row already has V00..V95 labels
        vcount = sum(1 for c in row if re.fullmatch(r"V\d{2}", c))
        if vcount >= 24:
            vnames_row = i
            break

    if time_row is None and vnames_row is None:
        print("[DEBUG] First 15 rows (normed) to help diagnose time header presence:")
        head = raw.head(15).fillna("").astype(str)
        print(head.to_string(index=True, header=False))
        raise ValueError("Could not find time header row with quarter-hour labels in SCATS CSV.")

    if vnames_row is not None:
        # Build header from that row of Vxx names plus a site header row for identifiers
        vrow = [norm_cell(v) for v in raw.iloc[vnames_row].tolist()]
        # Collect which columns are Vxx
        v_indices = [j for j, c in enumerate(vrow) if re.fullmatch(r"V\d{2}", c)]
        # Accept even if not all V00..V95 are explicitly named (will remap later if needed)

        # Choose site header row for non-V columns
        if site_row is None:
            site_row = max(0, vnames_row - 1)
        site_header = [norm_cell(v) for v in raw.iloc[site_row].tolist()]
        max_cols = max(len(site_header), len(vrow))
        final_header = []
        for j in range(max_cols):
            name = vrow[j] if j < len(vrow) and re.fullmatch(r"V\d{2}", vrow[j]) else (site_header[j] if j < len(site_header) else "")
            final_header.append(name)

        start_row = max(site_row, vnames_row) + 1
        df = raw.iloc[start_row:].reset_index(drop=True)
        df = df.reindex(columns=range(len(final_header)))
        df.columns = final_header
    else:
        # Found raw time labels; convert to V00..V95 by order
        time_cells.sort(key=lambda t: (t[1], t[2]))
        if len(time_cells) != 96:
            raise ValueError("SCATS CSV does not have exactly 96 quarter-hour columns.")
        idx_to_v = {col_idx: f"V{k:02d}" for k, (col_idx, _, _) in enumerate(time_cells)}

        # Use a site header row for identifiers
        if site_row is None:
            site_row = max(0, time_row - 1)
        site_header = [norm_cell(v) for v in raw.iloc[site_row].tolist()]
        time_header = [norm_cell(v) for v in raw.iloc[time_row].tolist()]
        max_cols = max(len(site_header), len(time_header))
        final_header = []
        for j in range(max_cols):
            if j in idx_to_v:
                final_header.append(idx_to_v[j])
            else:
                name = site_header[j] if j < len(site_header) else ""
                final_header.append(name)

        start_row = max(site_row, time_row) + 1
        df = raw.iloc[start_row:].reset_index(drop=True)
        df = df.reindex(columns=range(len(final_header)))
        df.columns = final_header

    # Identify essential columns
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    hf_col = None
    for key in ["hf", "hf number", "hf vicroads internal", "hf no", "hf_site_no", "hf site no"]:
        if key in lower:
            hf_col = lower[key]; break
    date_col = None
    for key in ["date", "count date", "day", "date/time", "datetime"]:
        if key in lower:
            date_col = lower[key]; break
    if date_col is None:
        for c in cols:
            if "date" in c.lower():
                date_col = c; break
    loc_col = None
    for key in ["location", "location description"]:
        if key in lower:
            loc_col = lower[key]; break

    if hf_col is None or date_col is None:
        raise ValueError(f"Missing HF/Date columns after header merge. Found: {list(df.columns)[:20]}")

    # Ensure all V00..V95 exist (some formats already had them as names)
    vcols = [f"V{i:02d}" for i in range(96)]
    missing_v = [v for v in vcols if v not in df.columns]
    if missing_v:
        raise ValueError(f"Missing time columns after header merge: {missing_v[:10]}... (total {len(missing_v)})")

    # Canonicalise
    ren = {hf_col: "HF", date_col: "Date"}
    if loc_col: ren[loc_col] = "Location"
    df = df.rename(columns=ren)

    # Types & drop NA
    df = df.dropna(subset=["HF","Date"]).copy()
    df["HF"] = pd.to_numeric(df["HF"], errors="coerce")
    df = df[pd.notna(df["HF"])].copy()
    df["HF"] = df["HF"].astype(int)

    # Keep only essentials
    keep = ["HF","Date"] + (["Location"] if "Location" in df.columns else []) + vcols
    return df[keep].reset_index(drop=True)

def hourify_row_to_list(row: pd.Series) -> list[float]:
    vals = []
    for i in range(96):
        v = row[f"V{i:02d}"]
        try:
            vals.append(float(v))
        except:
            vals.append(0.0)
    hours = []
    for h in range(24):
        hours.append(sum(vals[h*4:(h+1)*4]))
    return hours

def main():
    print("Loading SCATS Oct-2006 (HF-based)...")
    scats = load_scats_qh_use_hf()

    scats["Date"] = pd.to_datetime(scats["Date"], dayfirst=True, errors="coerce")
    scats = scats[pd.notna(scats["Date"])].copy()

    if "Location" in scats.columns:
        scats["First_Road"] = scats["Location"].apply(extract_first_road)
    else:
        scats["First_Road"] = ""

    scats = scats[scats["HF"] != -1].copy()

    print("Aggregating 15-min to hourly...")
    vcols = [f"V{i:02d}" for i in range(96)]
    rows = []
    for _, r in scats.iterrows():
        hf_id = int(r["HF"])
        day0  = pd.to_datetime(r["Date"]).normalize()
        first_road = r.get("First_Road", "")
        hours = hourify_row_to_list(r[vcols])
        for h in range(24):
            rows.append({
                "hf_id": hf_id,
                "First_Road": first_road,
                "datetime": day0 + pd.Timedelta(hours=h),
                "volume_vph": float(hours[h])
            })

    out = pd.DataFrame(rows).sort_values(["hf_id","datetime"]).reset_index(drop=True)
    out_path = ART / "processed_data.csv"
    out.to_csv(out_path, index=False)
    print("Done.")
    print(f"Wrote: {out_path}")
    print(f"Rows: {len(out):,} | Sites: {out['hf_id'].nunique()}")

if __name__ == "__main__":
    main()
