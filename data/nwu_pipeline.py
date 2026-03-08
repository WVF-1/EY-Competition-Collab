"""
NWU Water Chemistry Data Pipeline
===================================
Processes all 5 NWU source files + Sample Stations into a single clean CSV
that matches the EY competition training data schema exactly:

    Latitude | Longitude | Sample Date | Total Alkalinity |
    Electrical Conductance | Dissolved Reactive Phosphorus

Unit conversions applied:
  - EC:  mS/m × 10     →  µS/cm       (matches training data range ~15–1,506)
  - TAL: mg/L as CaCO₃ →  no change   (already matches training data)
  - PO4: mg/L × 1,000  →  µg/L        (NWU stores as mg/L; training data is in µg/L.
                                        Percentile alignment confirms this conversion:
                                        NWU p50 × 1000 = 26  vs  training p50 = 20  ✓)

Coordinates:
  - Latitude stored as positive in NWU (South Africa convention) → negated to WGS84
  - Longitude already positive (eastern hemisphere) → unchanged

Files expected (place in BASE_DIR or update paths below):
  - Dams_and_lakes_1999-2012__A-X_.xlsx
  - Dams_and_lakes_up_to_1998__A-X_.xlsx
  - Rivers_1999-2012__A-X_.xlsx
  - Rivers_up_to_1998__A-D_.xlsx
  - Rivers_up_to_1998__E-X_.xlsx
  - Sample_stations.xlsx

Usage:
  python nwu_pipeline.py
  Output: nwu_water_quality_clean.csv  (ready to concatenate with training data)

Citation (required under CC BY 4.0):
  Huizenga, J.M. et al. (2013). A national dataset of inorganic chemical data
  of surface waters in South Africa. Water SA, 39(2).
  waterscience.co.za/waterchemistry/data.html
"""

import os
import re
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS  —  update BASE_DIR to the folder containing your NWU downloads
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = "."

NWU_FILES = [
    os.path.join(BASE_DIR, "Dams_and_lakes_1999-2012__A-X_.xlsx"),
    os.path.join(BASE_DIR, "Dams_and_lakes_up_to_1998__A-X_.xlsx"),
    os.path.join(BASE_DIR, "Rivers_1999-2012__A-X_.xlsx"),
    os.path.join(BASE_DIR, "Rivers_up_to_1998__A-D_.xlsx"),
    os.path.join(BASE_DIR, "Rivers_up_to_1998__E-X_.xlsx"),
]

STATIONS_FILE = os.path.join(BASE_DIR, "Sample_stations.xlsx")
OUTPUT_FILE   = os.path.join(BASE_DIR, "nwu_water_quality_clean.csv")

MISSING_SENTINEL = -9999   # NWU's placeholder for missing data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & standardise one NWU measurement file
# ─────────────────────────────────────────────────────────────────────────────
def load_nwu_file(path: str) -> pd.DataFrame:
    """
    Reads one NWU Excel file.

    Header layout:
      Row 0 = section headers  (skipped)
      Row 1 = unit labels      (skipped)
      Row 2 = column names     <- used as header
      Row 3+ = data

    Extracts: STATION | DATE | EC (mS/m) | TAL (mg/L) | PO4 (mg/L)
    Applies unit conversions and returns a clean DataFrame.
    """
    print(f"  Loading: {os.path.basename(path)}")

    df = pd.read_excel(path, header=2)

    # Col 4 is EC in mS/m; pandas suffixes the duplicate EC col as 'EC.1'
    # We rename explicitly so the logic is clear regardless of pandas version
    cols = df.columns.tolist()
    col_rename = {cols[4]: "EC_mSm"}
    df = df.rename(columns=col_rename)

    # Different NWU files use different station column names — normalise all to STATION
    station_aliases = ["STATION", "SAMPLE\nSTATION ID", "SAMPLE STATION ID", "SAMPLE_STATION_ID"]
    for alias in station_aliases:
        if alias in df.columns:
            df = df.rename(columns={alias: "STATION"})
            break

    needed = ["STATION", "DATE", "EC_mSm", "TAL", "PO4"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {os.path.basename(path)}")

    df = df[needed].copy()

    # Replace missing sentinel with NaN
    df.replace(MISSING_SENTINEL, np.nan, inplace=True)

    # ── Unit conversions ──────────────────────────────────────────────────
    # EC: mS/m -> µS/cm  (x 10)
    df["Electrical Conductance"] = df["EC_mSm"] * 10

    # TAL: mg/L CaCO3 — no change needed
    df["Total Alkalinity"] = df["TAL"]

    # PO4: mg/L -> µg/L  (x 1000)
    # NWU reports in mg/L; training data DRP is in µg/L.
    # Percentile check confirms alignment after this conversion.
    df["Dissolved Reactive Phosphorus"] = df["PO4"] * 1000

    # ── Parse dates (already datetime objects from Excel; coerce errors) ──
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # Drop intermediate columns
    df.drop(columns=["EC_mSm", "TAL", "PO4"], inplace=True)

    return df[["STATION", "DATE",
               "Electrical Conductance",
               "Total Alkalinity",
               "Dissolved Reactive Phosphorus"]]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load & clean Sample Stations (coordinates)
# ─────────────────────────────────────────────────────────────────────────────
def load_stations(path: str) -> pd.DataFrame:
    """
    Returns: STATION | Latitude | Longitude

    NWU stores latitudes as positive decimal degrees (South African convention).
    WGS84 requires negative values for the southern hemisphere -> negate here.
    Longitude is already positive (eastern hemisphere) -> unchanged.
    """
    print(f"  Loading stations: {os.path.basename(path)}")

    ss = pd.read_excel(path)
    ss = ss.rename(columns={
        "SAMPLE\nSTATION ID":       "STATION",
        "SOUTH LAT.\nDECIMAL DEGR": "LAT_POS",
        "SOUTH LON.\nDECIMAL DEGR": "LON",
    })

    ss = ss[["STATION", "LAT_POS", "LON"]].copy()

    # Drop rows with missing or zero coordinates (data entry errors)
    ss.dropna(subset=["LAT_POS", "LON"], inplace=True)
    ss = ss[(ss["LAT_POS"] != 0) & (ss["LON"] != 0)]

    # Negate latitude for WGS84
    ss["Latitude"]  = -ss["LAT_POS"].abs()
    ss["Longitude"] =  ss["LON"]

    return ss[["STATION", "Latitude", "Longitude"]].drop_duplicates(subset="STATION")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline() -> pd.DataFrame:
    print("\n" + "=" * 62)
    print("  NWU WATER QUALITY PIPELINE")
    print("=" * 62)

    # ── 3a. Load all available NWU measurement files ─────────────────────
    frames = []
    for fpath in NWU_FILES:
        if not os.path.exists(fpath):
            print(f"  ⚠️  Not found — skipping: {os.path.basename(fpath)}")
            continue
        try:
            frames.append(load_nwu_file(fpath))
        except Exception as exc:
            print(f"  ✗ Error in {os.path.basename(fpath)}: {exc}")

    if not frames:
        raise RuntimeError("No NWU files were loaded. Check BASE_DIR and file names.")

    raw = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined raw rows:               {len(raw):>10,}")

    # ── 3b. Load station coordinates ─────────────────────────────────────
    stations = load_stations(STATIONS_FILE)
    print(f"  Station records with coords:    {len(stations):>10,}")

    # ── 3c. Join coordinates (inner join — only stations with known coords)
    merged = raw.merge(stations, on="STATION", how="inner")
    print(f"  Rows after coord join:          {len(merged):>10,}")

    # ── 3d. Drop rows missing any target variable, coordinate, or date ───
    before = len(merged)
    merged.dropna(
        subset=["Electrical Conductance", "Total Alkalinity",
                "Dissolved Reactive Phosphorus", "Latitude", "Longitude", "DATE"],
        inplace=True,
    )
    print(f"  Rows dropped (nulls):           {before - len(merged):>10,}")

    # ── 3e. Deduplicate on STATION + DATE (same instrument, same day) ────
    before = len(merged)
    merged.sort_values("DATE", inplace=True)
    merged.drop_duplicates(subset=["STATION", "DATE"], keep="first", inplace=True)
    print(f"  Rows dropped (station+date dup):{before - len(merged):>10,}")

    # ── 3f. Average rows sharing the same Lat + Lon + Date ───────────────
    # Handles two distinct stations registered at identical coordinates
    # on the same calendar day — average measurements rather than drop.
    before = len(merged)
    merged = (
        merged
        .groupby(["Latitude", "Longitude", "DATE"], as_index=False)
        .agg({
            "Total Alkalinity":              "mean",
            "Electrical Conductance":        "mean",
            "Dissolved Reactive Phosphorus": "mean",
        })
    )
    averaged = before - len(merged)
    if averaged > 0:
        print(f"  Rows merged (coord+date dup):   {averaged:>10,}")
    print(f"  Final clean rows:               {len(merged):>10,}")

    # ── 3g. Format date to match training data: DD-MM-YYYY ───────────────
    merged["Sample Date"] = merged["DATE"].dt.strftime("%d-%m-%Y")

    # ── 3h. Select and order final columns to match training schema ───────
    final_cols = [
        "Latitude",
        "Longitude",
        "Sample Date",
        "Total Alkalinity",
        "Electrical Conductance",
        "Dissolved Reactive Phosphorus",
    ]
    output = merged[final_cols].reset_index(drop=True)

    # ── 3i. Save ──────────────────────────────────────────────────────────
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  ✅ Saved: {OUTPUT_FILE}")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
def sanity_check(df: pd.DataFrame):
    print("\n" + "=" * 62)
    print("  SANITY CHECKS")
    print("=" * 62)

    passed = 0
    failed = 0

    def check(label: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✅ PASS  {label}")
        else:
            failed += 1
            print(f"  ❌ FAIL  {label}")
        if detail:
            print(f"           -> {detail}")

    # ── Schema ────────────────────────────────────────────────────────────
    expected_cols = [
        "Latitude", "Longitude", "Sample Date",
        "Total Alkalinity", "Electrical Conductance",
        "Dissolved Reactive Phosphorus",
    ]
    check(
        "All 6 required columns present",
        all(c in df.columns for c in expected_cols),
        f"Found: {df.columns.tolist()}",
    )

    check(
        "Dataset is non-empty",
        len(df) > 0,
        f"Total rows: {len(df):,}",
    )

    # ── Nulls ─────────────────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    check(
        "No null values in any column",
        null_counts.sum() == 0,
        f"Null counts: {null_counts.to_dict()}",
    )

    # ── Coordinates — South Africa bounding box ───────────────────────────
    lat_ok = df["Latitude"].between(-35.0, -22.0).all()
    check(
        "Latitude within South Africa  (-35.0 to -22.0)",
        lat_ok,
        f"Actual range: {df['Latitude'].min():.4f}  to  {df['Latitude'].max():.4f}",
    )

    lon_ok = df["Longitude"].between(16.0, 33.5).all()
    check(
        "Longitude within South Africa  (16.0 to 33.5)",
        lon_ok,
        f"Actual range: {df['Longitude'].min():.4f}  to  {df['Longitude'].max():.4f}",
    )

    # ── Electrical Conductance ────────────────────────────────────────────
    ec = df["Electrical Conductance"]
    check(
        "EC values are positive",
        (ec > 0).all(),
        f"Min: {ec.min():.1f} µS/cm",
    )
    check(
        "EC plausible upper bound  (< 50,000 µS/cm)",
        (ec < 50_000).all(),
        f"Max observed: {ec.max():.1f}  |  Training max: ~1,506 µS/cm",
    )

    # ── Total Alkalinity ──────────────────────────────────────────────────
    ta = df["Total Alkalinity"]
    check(
        "Total Alkalinity values are positive",
        (ta > 0).all(),
        f"Range: {ta.min():.2f}  to  {ta.max():.2f} mg/L CaCO3",
    )

    # ── DRP ───────────────────────────────────────────────────────────────
    drp = df["Dissolved Reactive Phosphorus"]
    check(
        "DRP values are non-negative",
        (drp >= 0).all(),
        f"Range: {drp.min():.2f}  to  {drp.max():.2f} µg/L",
    )

    # ── Date format  DD-MM-YYYY ───────────────────────────────────────────
    date_pattern = re.compile(r"^\d{2}-\d{2}-\d{4}$")
    sample_check = df["Sample Date"].dropna().head(300)
    dates_ok = sample_check.apply(lambda d: bool(date_pattern.match(str(d)))).all()
    check(
        "Sample Date format matches DD-MM-YYYY",
        dates_ok,
        f"Sample: {df['Sample Date'].head(3).tolist()}",
    )

    # ── No duplicate Lat + Lon + Date ─────────────────────────────────────
    dupes = df.duplicated(subset=["Latitude", "Longitude", "Sample Date"]).sum()
    check(
        "No duplicate Lat + Lon + Date combinations",
        dupes == 0,
        f"Duplicates found: {dupes:,}",
    )

    # ── Distribution comparison vs EY training data ───────────────────────
    print()
    print("  Percentile distribution vs EY training data:")
    print(f"  {'Variable':<35} {'NWU p25':>9} {'NWU p50':>9} {'NWU p75':>9}  "
          f"{'Train p25':>10} {'Train p50':>10} {'Train p75':>10}")
    print(f"  {'─' * 96}")

    # Reference training percentiles (computed from water_quality_training_dataset.csv)
    ref_percentiles = {
        "Total Alkalinity":               (33.3,   92.5,  182.6),
        "Electrical Conductance":         (198.2,  381.6, 678.6),
        "Dissolved Reactive Phosphorus":  (10.0,   20.0,   48.0),
    }
    for col, (tp25, tp50, tp75) in ref_percentiles.items():
        p25 = df[col].quantile(0.25)
        p50 = df[col].quantile(0.50)
        p75 = df[col].quantile(0.75)
        print(f"  {col:<35} {p25:>9.2f} {p50:>9.2f} {p75:>9.2f}  "
              f"{tp25:>10.2f} {tp50:>10.2f} {tp75:>10.2f}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  {'─' * 44}")
    print(f"  Results:  {passed} passed  |  {failed} failed")
    if failed == 0:
        print("  🎉 All checks passed — data is ready for modelling!")
    else:
        print("  ⚠️  Some checks failed — review details above before proceeding.")
    print("=" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    clean_df = run_pipeline()
    sanity_check(clean_df)
