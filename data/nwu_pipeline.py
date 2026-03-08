# ============================================================
# NWU WATER CHEMISTRY PIPELINE — GOOGLE COLAB VERSION
# (FILES ALREADY UPLOADED)
#
# Fixes vs previous version:
#   ✓ Outlier cap restored: 3× training maximum per variable
#     (was accidentally dropped in the Colab rewrite)
#   ✓ Zero-floor filter added: TA=0 and EC=0 are physically
#     impossible — those rows are instrument errors or
#     sentinel values that slipped through -9999 replacement
#   ✓ Sanity check now reports counts removed at each stage
#     and flags if any variable range looks implausible
#
# Cap values derived from EY training data maxima × 3:
#   Total Alkalinity:              cap =  1,085 mg/L CaCO₃
#   Electrical Conductance:        cap =  4,518 µS/cm
#   Dissolved Reactive Phosphorus: cap =    585 µg/L
#
# These preserve genuine extreme South African waterbodies
# (mineralised springs, seasonal pans) while removing
# instrument errors, mine drainage spikes, and sewage
# outflows that would corrupt the KD-tree spatial baselines.
# ============================================================

import pandas as pd
import numpy as np
import re

MISSING_SENTINEL = -9999

# Outlier caps: 3× the maximum observed in the EY training data.
# Adjust these if your training data has a different range.
TA_CAP  = 1085.0    # mg/L CaCO₃
EC_CAP  = 4518.0    # µS/cm
DRP_CAP = 585.0     # µg/L

# Physical floor: values at or below zero are impossible
# for surface water TA and EC (DRP can legitimately be ~0)
TA_FLOOR  = 0.0
EC_FLOOR  = 0.0

# ------------------------------------------------------------
# FILE NAMES (must match uploaded files)
# ------------------------------------------------------------
NWU_FILES = [
    'Rivers up to 1998 (A-D).xlsx',
    'Dams and lakes 1999-2012 (A-X).xlsx',
    'Rivers 1999-2012 (A-X).xlsx',
    'Dams and lakes up to 1998 (A-X).xlsx',
    'Rivers up to 1998 (E-X).xlsx',
]

STATIONS_FILE = "Sample stations.xlsx"
OUTPUT_FILE   = "nwu_water_quality_clean.csv"


# ------------------------------------------------------------
# LOAD NWU MEASUREMENT FILE
# ------------------------------------------------------------
def load_nwu_file(path):
    print(f"  Loading: {path}")

    df = pd.read_excel(path, header=2)

    # Col 4 is EC in mS/m — rename before anything else
    cols = df.columns.tolist()
    df = df.rename(columns={cols[4]: "EC_mSm"})

    # Station column has different names across files
    station_aliases = [
        "STATION",
        "SAMPLE\nSTATION ID",
        "SAMPLE STATION ID",
        "SAMPLE_STATION_ID",
    ]
    for alias in station_aliases:
        if alias in df.columns:
            df = df.rename(columns={alias: "STATION"})
            break

    needed = ["STATION", "DATE", "EC_mSm", "TAL", "PO4"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} — missing columns: {missing}")

    df = df[needed].copy()

    # Replace NWU missing sentinel with NaN
    df.replace(MISSING_SENTINEL, np.nan, inplace=True)

    # ── Unit conversions ──────────────────────────────────────
    # EC:  mS/m  × 10   → µS/cm
    # TAL: mg/L CaCO₃   → no change
    # PO4: mg/L  × 1000 → µg/L  (matches EY training units)
    df["Electrical Conductance"]        = df["EC_mSm"] * 10
    df["Total Alkalinity"]              = df["TAL"]
    df["Dissolved Reactive Phosphorus"] = df["PO4"] * 1000

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    df.drop(columns=["EC_mSm", "TAL", "PO4"], inplace=True)

    return df[["STATION", "DATE",
               "Total Alkalinity",
               "Electrical Conductance",
               "Dissolved Reactive Phosphorus"]]


# ------------------------------------------------------------
# LOAD STATION COORDINATES
# ------------------------------------------------------------
def load_stations(path):
    print(f"  Loading stations: {path}")

    ss = pd.read_excel(path)
    ss = ss.rename(columns={
        "SAMPLE\nSTATION ID":       "STATION",
        "SOUTH LAT.\nDECIMAL DEGR": "LAT_POS",
        "SOUTH LON.\nDECIMAL DEGR": "LON",
    })

    ss = ss[["STATION", "LAT_POS", "LON"]].dropna()
    ss = ss[(ss["LAT_POS"] != 0) & (ss["LON"] != 0)]

    # NWU stores latitude as positive — negate for WGS84
    ss["Latitude"]  = -ss["LAT_POS"].abs()
    ss["Longitude"] =  ss["LON"]

    return ss[["STATION", "Latitude", "Longitude"]].drop_duplicates()


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def run_pipeline():
    print("\n" + "=" * 56)
    print("  NWU WATER CHEMISTRY PIPELINE")
    print("=" * 56)

    # ── Load all measurement files ───────────────────────────
    frames = []
    for path in NWU_FILES:
        try:
            frames.append(load_nwu_file(path))
        except Exception as exc:
            print(f"  ⚠ Skipping {path}: {exc}")

    if not frames:
        raise RuntimeError("No NWU files loaded — check filenames.")

    raw = pd.concat(frames, ignore_index=True)
    print(f"\n  Raw rows combined:              {len(raw):>10,}")

    # ── Join station coordinates ─────────────────────────────
    stations = load_stations(STATIONS_FILE)
    merged   = raw.merge(stations, on="STATION", how="inner")
    print(f"  Rows after coordinate join:     {len(merged):>10,}")

    # ── Drop rows with any missing target or coordinate ──────
    before = len(merged)
    merged.dropna(subset=[
        "Total Alkalinity", "Electrical Conductance",
        "Dissolved Reactive Phosphorus",
        "Latitude", "Longitude", "DATE",
    ], inplace=True)
    print(f"  Rows dropped (nulls):           {before - len(merged):>10,}")

    # ── Physical floor filter ────────────────────────────────
    # TA = 0 and EC = 0 are physically impossible for surface
    # water — these are instrument errors or missed sentinels.
    before = len(merged)
    merged = merged[
        (merged["Total Alkalinity"]       > TA_FLOOR) &
        (merged["Electrical Conductance"] > EC_FLOOR)
    ]
    print(f"  Rows dropped (zero floor):      {before - len(merged):>10,}")

    # ── Outlier cap: 3× EY training maximum ─────────────────
    # Removes mine drainage spikes, sewage outflows, and brine
    # pans that would corrupt the KD-tree spatial baselines.
    before = len(merged)
    merged = merged[
        (merged["Total Alkalinity"]              <= TA_CAP)  &
        (merged["Electrical Conductance"]        <= EC_CAP)  &
        (merged["Dissolved Reactive Phosphorus"] <= DRP_CAP)
    ]
    print(f"  Rows dropped (outlier cap 3×):  {before - len(merged):>10,}")

    # ── Deduplicate: same station + same date ────────────────
    before = len(merged)
    merged.sort_values("DATE", inplace=True)
    merged.drop_duplicates(subset=["STATION", "DATE"], keep="first", inplace=True)
    print(f"  Rows dropped (station+date dup):{before - len(merged):>10,}")

    # ── Average rows at identical coordinates on same date ───
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
    if before - len(merged) > 0:
        print(f"  Rows merged (coord+date dup):   {before - len(merged):>10,}")

    print(f"\n  Final clean rows:               {len(merged):>10,}")

    # ── Format date to DD-MM-YYYY ────────────────────────────
    merged["Sample Date"] = merged["DATE"].dt.strftime("%d-%m-%Y")

    output = merged[[
        "Latitude", "Longitude", "Sample Date",
        "Total Alkalinity", "Electrical Conductance",
        "Dissolved Reactive Phosphorus",
    ]].reset_index(drop=True)

    output.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE}")

    return output


# ------------------------------------------------------------
# SANITY CHECKS
# ------------------------------------------------------------
def sanity_check(df):
    print("\n" + "=" * 56)
    print("  SANITY CHECKS")
    print("=" * 56)

    passed = failed = 0

    def check(label, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✅ PASS  {label}")
        else:
            failed += 1
            print(f"  ❌ FAIL  {label}")
        if detail:
            print(f"           → {detail}")

    check("Row count > 0",
          len(df) > 0, f"{len(df):,} rows")

    check("Zero null values",
          df.isnull().sum().sum() == 0,
          str(df.isnull().sum().to_dict()))

    check("Latitude within SA  (-35 to -22)",
          df["Latitude"].between(-35, -22).all(),
          f"{df['Latitude'].min():.4f} to {df['Latitude'].max():.4f}")

    check("Longitude within SA  (16 to 33.5)",
          df["Longitude"].between(16, 33.5).all(),
          f"{df['Longitude'].min():.4f} to {df['Longitude'].max():.4f}")

    check(f"Total Alkalinity within cap  (0 – {TA_CAP})",
          df["Total Alkalinity"].between(0, TA_CAP, inclusive="right").all(),
          f"Range: {df['Total Alkalinity'].min():.2f} to {df['Total Alkalinity'].max():.2f}")

    check(f"Electrical Conductance within cap  (0 – {EC_CAP})",
          df["Electrical Conductance"].between(0, EC_CAP, inclusive="right").all(),
          f"Range: {df['Electrical Conductance'].min():.2f} to {df['Electrical Conductance'].max():.2f}")

    check(f"DRP within cap  (0 – {DRP_CAP})",
          df["Dissolved Reactive Phosphorus"].between(0, DRP_CAP, inclusive="both").all(),
          f"Range: {df['Dissolved Reactive Phosphorus'].min():.2f} to {df['Dissolved Reactive Phosphorus'].max():.2f}")

    pat = re.compile(r"^\d{2}-\d{2}-\d{4}$")
    check("Date format DD-MM-YYYY",
          df["Sample Date"].head(300).apply(
              lambda x: bool(pat.match(str(x)))
          ).all(),
          f"Sample: {df['Sample Date'].head(3).tolist()}")

    check("No duplicate Lat + Lon + Date",
          df.duplicated(subset=["Latitude", "Longitude", "Sample Date"]).sum() == 0)

    # Distribution comparison vs EY training data
    print()
    print("  Percentile check vs EY training data:")
    print(f"  {'Variable':<35} {'NWU p25':>9} {'NWU p50':>9} {'NWU p75':>9} "
          f"{'Train p25':>10} {'Train p50':>10} {'Train p75':>10}")
    print(f"  {'─'*96}")

    ref = {
        "Total Alkalinity":               (33.3,  92.5, 182.6),
        "Electrical Conductance":         (198.2, 381.6, 678.6),
        "Dissolved Reactive Phosphorus":  (10.0,  20.0,  48.0),
    }
    for col, (tp25, tp50, tp75) in ref.items():
        p25 = df[col].quantile(0.25)
        p50 = df[col].quantile(0.50)
        p75 = df[col].quantile(0.75)
        print(f"  {col:<35} {p25:>9.1f} {p50:>9.1f} {p75:>9.1f} "
              f"{tp25:>10.1f} {tp50:>10.1f} {tp75:>10.1f}")

    print(f"\n  {'─'*44}")
    print(f"  {passed} passed | {failed} failed")
    if failed == 0:
        print("  🎉 All checks passed — data is ready for modelling!")
    else:
        print("  ⚠️  Review failures above before proceeding.")
    print("=" * 56)


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
clean_df = run_pipeline()
sanity_check(clean_df)
