"""
Pre-compute all website data and save to results/data.pkl

Run once (or whenever new results land):
    python generate_pickle.py

Load the pickle on the website with import pickle
    with open("results/data.pkl", "rb") as f:
        data = pickle.load(f)

What's inside data
------------------
data["pipeline"]          pd.DataFrame  — LOSO accuracy per feat/classifier/subject
data["sliding_window"]    pd.DataFrame  — accuracy per window size/method/subject
data["itr"]               pd.DataFrame  — ITR (bits/min) per window size/method/subject
data["augmentation"]      pd.DataFrame  — window count and aug factor per window size
data["summary"]           dict          — headline numbers for scorecards
data["constants"]         dict          — FS, STIM_FREQS, ITI, etc.
data["generated_at"]      str           — ISO timestamp

All heavy computation (LOSO, feature extraction) is already stored in the CSVs.
This script reads those CSVs, enriches them, and packages everything cleanly.
Augmentation stats are recomputed live from one .mat file (takes ~1 s).
"""

import pickle
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
RESULTS       = ROOT / "results"
DATA_RAW      = ROOT / "data" / "raw" / "ssvep"
OUT_PICKLE    = RESULTS / "data.pkl"

PIPELINE_CSV  = RESULTS / "full_pipeline_results.csv"
WINDOW_CSV    = RESULTS / "sliding_window_results.csv"

# Reference .mat file used only for augmentation stats (any subject/session works)
REF_MAT = DATA_RAW / "subject_1_fvep_led_training_1.mat"

# ── Constants (mirrored here so the pickle is self-contained) ──────────────
FS         = 256
STIM_FREQS = [9, 10, 12, 15]
ITI_SEC    = 3.145          # inter-trial interval from dataset
N_CLASSES  = 4
CHANCE     = 0.25
WIN_SIZES  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.85]
STEP_SEC   = 0.5
PRE_SEC    = 0.5


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _itr(P, T_win, N=N_CLASSES, ITI=ITI_SEC):
    """Wolpaw (2000): ITR in bits/min."""
    if P <= 0 or P < 1.0 / N:
        return 0.0
    B = (np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))
         if P < 1.0 else np.log2(N))
    return max(0.0, B) * 60.0 / (T_win + ITI)


# ══════════════════════════════════════════════════════════════════════════
# 1. Pipeline results
# ══════════════════════════════════════════════════════════════════════════

def build_pipeline(path=PIPELINE_CSV):
    """
    Load LOSO pipeline results and average over LOSO directions.

    Returns a DataFrame with one row per (feat, classifier, subject):
        feat, classifier, subject, accuracy_mean, accuracy_std
    """
    df = pd.read_csv(path)

    # Average over the two LOSO directions (train→test A→B and B→A)
    avg = (
        df.groupby(["feat", "classifier", "subject"])["accuracy"]
        .agg(accuracy_mean="mean", accuracy_std="std")
        .reset_index()
    )
    avg["accuracy_std"] = avg["accuracy_std"].fillna(0.0)

    # Also keep the raw fold-level data for detailed views
    return df, avg


# ══════════════════════════════════════════════════════════════════════════
# 2. Sliding window + ITR
# ══════════════════════════════════════════════════════════════════════════

def build_sliding_window(path=WINDOW_CSV):
    """
    Load sliding window results and derive ITR.

    Returns:
        sw_raw  — raw fold-level accuracy (win_sec, method, subject, test_sess, accuracy)
        sw_avg  — averaged over LOSO folds
        itr_df  — ITR (bits/min) for each (win_sec, method, subject)
    """
    sw = pd.read_csv(path)

    sw_avg = (
        sw.groupby(["win_sec", "method", "subject"])["accuracy"]
        .agg(accuracy_mean="mean", accuracy_std="std")
        .reset_index()
    )
    sw_avg["accuracy_std"] = sw_avg["accuracy_std"].fillna(0.0)

    # Derive ITR row-by-row
    itr_rows = []
    for _, row in sw_avg.iterrows():
        itr_rows.append({
            "win_sec":   row["win_sec"],
            "method":    row["method"],
            "subject":   row["subject"],
            "itr":       _itr(row["accuracy_mean"], row["win_sec"]),
            "accuracy":  row["accuracy_mean"],
        })
    itr_df = pd.DataFrame(itr_rows)

    return sw, sw_avg, itr_df


# ══════════════════════════════════════════════════════════════════════════
# 3. Augmentation stats
# ══════════════════════════════════════════════════════════════════════════

def build_augmentation(mat_path=REF_MAT):
    """
    Compute window counts and augmentation factors for each window size.

    Uses one .mat file as a reference (all sessions/subjects have the same
    trial length: 7.355 s → 6.855 s usable after 0.5 s pre-stimulus skip).
    """
    from src import load_ssvep_data, preprocess, augmentation_stats, EEG_COLS

    df    = preprocess(load_ssvep_data(str(mat_path)))
    stats = augmentation_stats(
        df, EEG_COLS,
        win_sizes=WIN_SIZES,
        step_sec=STEP_SEC,
        pre_sec=PRE_SEC,
        fs=FS,
    )
    return stats


# ══════════════════════════════════════════════════════════════════════════
# 4. Summary / scorecard numbers
# ══════════════════════════════════════════════════════════════════════════

def build_summary(pipeline_avg, itr_df):
    """
    Headline numbers for the website scorecards.

    Returns a plain dict — easy to access by key.
    """
    # Best accuracy per subject (FBCCA, SVM, averaged LOSO)
    fbcca_svm = pipeline_avg[
        (pipeline_avg["feat"] == "FBCCA") & (pipeline_avg["classifier"] == "SVM")
    ].set_index("subject")["accuracy_mean"]

    # Best ITR per subject (peak across all window sizes)
    best_itr = itr_df.groupby("subject")["itr"].max()

    # Best window for ITR
    best_win_row = itr_df.loc[itr_df.groupby("subject")["itr"].idxmax()]

    summary = {
        # Accuracy
        "fbcca_acc_sub1": float(fbcca_svm.get(1, 0.0)),
        "fbcca_acc_sub2": float(fbcca_svm.get(2, 0.0)),

        # ITR
        "best_itr_sub1":        float(best_itr.get(1, 0.0)),
        "best_itr_sub2":        float(best_itr.get(2, 0.0)),
        "best_itr_win_sub1":    float(
            best_win_row[best_win_row["subject"] == 1]["win_sec"].values[0]
            if (best_win_row["subject"] == 1).any() else 0.0
        ),
        "best_itr_win_sub2":    float(
            best_win_row[best_win_row["subject"] == 2]["win_sec"].values[0]
            if (best_win_row["subject"] == 2).any() else 0.0
        ),

        # Dataset
        "n_subjects":             2,
        "n_classes":              N_CLASSES,
        "stim_freqs":             STIM_FREQS,
        "n_trials_per_session":   20,
        "trial_duration_sec":     7.355,
        "usable_duration_sec":    6.855,
        "chance_level":           CHANCE,
    }
    return summary


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("Generating results/data.pkl ...")

    print("  [1/4] Loading pipeline results ...")
    pipeline_raw, pipeline_avg = build_pipeline()

    print("  [2/4] Loading sliding window results + computing ITR ...")
    sw_raw, sw_avg, itr_df = build_sliding_window()

    print("  [3/4] Computing augmentation stats (loading 1 .mat file) ...")
    aug_df = build_augmentation()

    print("  [4/4] Building summary ...")
    summary = build_summary(pipeline_avg, itr_df)

    # ── Bundle ────────────────────────────────────────────────────────────
    data = {
        # ── DataFrames ────────────────────────────────────────────────────
        # LOSO pipeline: accuracy per feat/classifier/subject/fold
        "pipeline":           pipeline_raw,
        # LOSO pipeline: averaged over folds
        "pipeline_avg":       pipeline_avg,

        # Sliding window: accuracy per win_sec/method/subject/fold
        "sliding_window":     sw_raw,
        # Sliding window: averaged over folds
        "sliding_window_avg": sw_avg,

        # ITR table (bits/min) per win_sec/method/subject
        "itr":                itr_df,

        # Augmentation table: window counts & aug factor
        "augmentation":       aug_df,

        # ── Summary / scorecard ───────────────────────────────────────────
        "summary":            summary,

        # ── Metadata ──────────────────────────────────────────────────────
        "constants": {
            "FS":         FS,
            "STIM_FREQS": STIM_FREQS,
            "ITI_SEC":    ITI_SEC,
            "N_CLASSES":  N_CLASSES,
            "CHANCE":     CHANCE,
            "WIN_SIZES":  WIN_SIZES,
            "STEP_SEC":   STEP_SEC,
            "PRE_SEC":    PRE_SEC,
        },
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    RESULTS.mkdir(exist_ok=True)
    with open(OUT_PICKLE, "wb") as f:
        pickle.dump(data, f, protocol=4)

    # ── Print summary ─────────────────────────────────────────────────────
    size_kb = OUT_PICKLE.stat().st_size / 1024
    print(f"\nSaved to {OUT_PICKLE}  ({size_kb:.1f} KB)")
    print()
    print("Contents:")
    for key, val in data.items():
        if isinstance(val, pd.DataFrame):
            print(f"  data['{key}']  → DataFrame {val.shape}")
        elif isinstance(val, dict):
            print(f"  data['{key}']  → dict ({len(val)} keys)")
        else:
            print(f"  data['{key}']  → {val!r}")

    print()
    print("Quick-load on the website:")
    print("  import pickle")
    print("  with open('results/data.pkl', 'rb') as f:")
    print("      data = pickle.load(f)")


if __name__ == "__main__":
    main()
