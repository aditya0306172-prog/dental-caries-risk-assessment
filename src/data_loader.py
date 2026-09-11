"""
NHANES 2017-2018 data loader for dental caries & periodontal disease
risk assessment.

Downloads CDC NHANES 2017-2018 survey datasets:
  - DEMO_J.XPT  : Demographics (age, gender)
  - OHXDEN_J.XPT: Oral health dental examination (decay/status)
  - OHQ_J.XPT   : Oral health questionnaire (self-rated oral health,
                   bleeding gums, pain frequency)
  - DIQ_J.XPT   : Diabetes questionnaire (diabetes status)

Merges on SEQN, cleans, and saves to  data/raw/nhanes_real_dental.csv .
"""

from __future__ import annotations

import os
from io import BytesIO
from urllib.request import urlopen, Request

import gzip

import numpy as np
import pandas as pd


# ── NHANES 2017-2018 URLs ─────────────────────────────────
NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
URLS = {
    "demo":   f"{NHANES_BASE}/DEMO_J.xpt",
    "ohxden": f"{NHANES_BASE}/OHXDEN_J.xpt",
    "ohq":    f"{NHANES_BASE}/OHQ_J.xpt",
    "diq":    f"{NHANES_BASE}/DIQ_J.xpt",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CSV_PATH = os.path.join(RAW_DIR, "nhanes_real_dental.csv")


def _fetch_xpt(url: str) -> pd.DataFrame:
    """Download a SAS XPORT (.XPT) file from CDC and return as DataFrame."""
    name = url.rsplit("/", 1)[-1]
    print(f"  Downloading {name} ...")

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        raw = resp.read()

    # The CDC server may silently gzip the payload.
    # XPORT files always start with "HEADER RECORD*******"
    # Gzip data starts with bytes 0x1f 0x8b.
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    buf = BytesIO(raw)
    return pd.read_sas(buf, format="xport")


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name found in *df* from *candidates*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_data(force_download: bool = False) -> pd.DataFrame:
    """Load the cleaned NHANES dental dataset.

    On first call the raw XPT files are downloaded from the CDC,
    merged on SEQN, cleaned, and cached as a CSV.  Subsequent calls
    read from the cache unless *force_download* is True.

    Features produced
    -----------------
    age              : int   - patient age in years
    gender           : int   - 0 = Male, 1 = Female
    bleeding_gums    : int   - 0 = No, 1 = Yes
    mouth_pain_freq  : int   - 0 (never) .. 4 (very often)
    diabetes_status  : int   - 0 = No, 1 = Yes

    Target
    ------
    caries_or_perio_risk : int  - 1 if self-rated oral health >= 4
                                  (Fair / Poor), else 0
    """
    if not force_download and os.path.exists(CSV_PATH):
        print(f"Loading cached dataset: {CSV_PATH}")
        return pd.read_csv(CSV_PATH)

    print("=" * 60)
    print("Downloading NHANES 2017-2018 datasets from CDC ...")
    print("=" * 60)

    demo   = _fetch_xpt(URLS["demo"])
    ohxden = _fetch_xpt(URLS["ohxden"])
    ohq    = _fetch_xpt(URLS["ohq"])
    diq    = _fetch_xpt(URLS["diq"])

    # ── Diagnostics ───────────────────────────────────────
    print(f"\n  DEMO  : {demo.shape[0]:,} rows, {demo.shape[1]} cols")
    print(f"  OHXDEN: {ohxden.shape[0]:,} rows, {ohxden.shape[1]} cols")
    print(f"  OHQ   : {ohq.shape[0]:,} rows, {ohq.shape[1]} cols")
    print(f"  OHQ columns: {list(ohq.columns)}")
    print(f"  DIQ   : {diq.shape[0]:,} rows, {diq.shape[1]} cols")

    # ── Demographics ──────────────────────────────────────
    demo_sub = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
    demo_sub.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "gender"}, inplace=True)
    demo_sub["gender"] = demo_sub["gender"].map({1.0: 0, 2.0: 1})

    # ── Oral Health Questionnaire ─────────────────────────
    ohq_keep = ["SEQN", "OHQ030"]

    # Bleeding gums (try several known NHANES variable names)
    bleed_var = _find_column(ohq, ["OHQ860", "OHQ620", "OHQ835"])
    if bleed_var:
        ohq_keep.append(bleed_var)
        print(f"\n  Bleeding gums variable : {bleed_var}")
    else:
        print("\n  [WARN] No bleeding-gums variable found in OHQ_J; "
              "feature will be omitted.")

    # Mouth pain frequency
    pain_var = _find_column(ohq, ["OHQ033", "OHQ770", "OHQ850"])
    if pain_var:
        ohq_keep.append(pain_var)
        print(f"  Mouth pain variable    : {pain_var}")
    else:
        print("  [WARN] No mouth-pain variable found in OHQ_J; "
              "feature will be omitted.")

    ohq_sub = ohq[[c for c in ohq_keep if c in ohq.columns]].copy()

    # Map bleeding gums to binary 0/1
    if bleed_var and bleed_var in ohq_sub.columns:
        ohq_sub["bleeding_gums"] = ohq_sub[bleed_var].map({1.0: 1, 2.0: 0})
        ohq_sub.drop(columns=[bleed_var], inplace=True)

    # Map mouth pain frequency (invert: 1=Very often..5=Never -> 4..0)
    if pain_var and pain_var in ohq_sub.columns:
        ohq_sub["mouth_pain_freq"] = (5 - ohq_sub[pain_var]).clip(0, 4)
        ohq_sub.drop(columns=[pain_var], inplace=True)

    # ── Diabetes ──────────────────────────────────────────
    diq_sub = diq[["SEQN", "DIQ010"]].copy()
    # 1 = Yes diabetes, 2 = No, 3 = Borderline -> treat as No
    diq_sub["diabetes_status"] = diq_sub["DIQ010"].map({1.0: 1, 2.0: 0, 3.0: 0})
    diq_sub.drop(columns=["DIQ010"], inplace=True)

    # ── Merge on SEQN ────────────────────────────────────
    merged = demo_sub.merge(ohq_sub, on="SEQN", how="inner")
    merged = merged.merge(diq_sub, on="SEQN", how="inner")
    # Keep only respondents who also had a dental exam
    merged = merged.merge(ohxden[["SEQN"]].drop_duplicates(),
                          on="SEQN", how="inner")
    print(f"\n  Merged rows (before dropna): {merged.shape[0]:,}")

    # ── Target ────────────────────────────────────────────
    # Self-rated oral health: 1=Excellent .. 5=Poor
    # >= 4  (Fair / Poor) => high risk
    merged["caries_or_perio_risk"] = (merged["OHQ030"] >= 4).astype(int)

    # ── Final cleanup ─────────────────────────────────────
    merged.drop(columns=["SEQN", "OHQ030"], inplace=True)
    merged.dropna(inplace=True)

    # Cast numeric columns to int
    for col in merged.columns:
        merged[col] = merged[col].astype(int)

    # ── Save ──────────────────────────────────────────────
    os.makedirs(RAW_DIR, exist_ok=True)
    merged.to_csv(CSV_PATH, index=False)

    features = [c for c in merged.columns if c != "caries_or_perio_risk"]
    print(f"\n  Cleaned dataset saved -> {CSV_PATH}")
    print(f"  Shape   : {merged.shape}")
    print(f"  Features: {features}")
    print(f"  Class distribution:")
    for label, count in merged["caries_or_perio_risk"].value_counts().sort_index().items():
        pct = count / len(merged) * 100
        tag = "low risk" if label == 0 else "high risk"
        print(f"    {label} ({tag}): {count:,}  ({pct:.1f}%)")

    return merged


if __name__ == "__main__":
    df = load_data()
    print(f"\n{df.head(10).to_string()}")
    print(f"\nDataset shape: {df.shape}")
