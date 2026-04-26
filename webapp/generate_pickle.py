"""
Pre-compute all website data and save to results/data.pkl

Run once (or whenever new results land):
    python3 generate_pickle.py

Load the pickle on the website with import pickle
    with open("results/data.pkl", "rb") as f:
        data = pickle.load(f)

What's inside data
------------------
data["pipeline"]          pd.DataFrame  — LOSO accuracy per feat/classifier/subject
data["sliding_window"]    pd.DataFrame  — accuracy per window size/method/subject
data["itr"]               pd.DataFrame  — ITR (bits/min) per window size/method/subject
data["augmentation"]      pd.DataFrame  — window count and aug factor per window size
data["snr_vs_success"]    pd.DataFrame  — per-trial PSD LOSO correctness + Oz SNR
data["cca_features_sub1"] pd.DataFrame  — raw CCA feature matrix for Subject 1, session 1
data["cca_features_sub2"] pd.DataFrame  — raw CCA feature matrix for Subject 2, session 1
data["psd_features_sub1"] pd.DataFrame  — raw PSD feature matrix for Subject 1, session 1
data["psd_features_sub2"] pd.DataFrame  — raw PSD feature matrix for Subject 2, session 1
data["summary"]           dict          — headline numbers for scorecards
data["story"]             dict          — derived storytelling metrics / callouts
data["constants"]         dict          — FS, STIM_FREQS, ITI, etc.
data["generated_at"]      str           — ISO timestamp

The website summaries still come from the CSVs, but feature-level story blocks are
computed from the current `src` extraction methods so they stay aligned with the
real preprocessing / feature APIs used elsewhere in the repo.
"""

import pickle
import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── Paths ──────────────────────────────────────────────────────────────────
APP_DIR       = Path(__file__).resolve().parent
ROOT          = APP_DIR.parent
sys.path.insert(0, str(ROOT))
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


def load_preprocessed_sessions(data_dir=DATA_RAW):
    """
    Load and preprocess all subject/session `.mat` files once.

    Returns
    -------
    dict[(subject, session)] -> preprocessed trial DataFrame
    """
    from src import load_ssvep_data, preprocess

    sessions = {}
    for subject in [1, 2]:
        for session in [1, 2]:
            path = data_dir / f"subject_{subject}_fvep_led_training_{session}.mat"
            sessions[(subject, session)] = preprocess(load_ssvep_data(str(path)))
    return sessions


def _extract_features(df, method, eeg_cols, win_sec=6.85):
    """Dispatch to the canonical extractor in `src`."""
    from src import extract_cca, extract_fbcca, extract_psd

    if method == "psd":
        return extract_psd(df, eeg_cols, STIM_FREQS, win_sec=win_sec, pre_sec=PRE_SEC, fs=FS)
    if method == "cca":
        return extract_cca(df, eeg_cols, STIM_FREQS, win_sec=win_sec, pre_sec=PRE_SEC, fs=FS)
    if method == "fbcca":
        return extract_fbcca(df, eeg_cols, STIM_FREQS, win_sec=win_sec, pre_sec=PRE_SEC, fs=FS)
    raise ValueError(f"Unsupported method: {method}")


def build_feature_snapshots(sessions, feature_session=1, win_sec=6.85):
    """
    Build ready-to-pickle feature matrices for story / PCA plots.

    One DataFrame per subject, using a single fixed session so the plot input is
    deterministic and easy for Streamlit to consume.
    """
    from src import EEG_COLS

    payload = {}
    for subject in [1, 2]:
        base = sessions[(subject, feature_session)]
        cca_df = _extract_features(base, "cca", EEG_COLS, win_sec=win_sec).copy()
        psd_df = _extract_features(base, "psd", EEG_COLS, win_sec=win_sec).copy()

        cca_df.insert(0, "session", feature_session)
        cca_df.insert(0, "subject", subject)
        psd_df.insert(0, "session", feature_session)
        psd_df.insert(0, "subject", subject)

        payload[f"cca_features_sub{subject}"] = cca_df
        payload[f"psd_features_sub{subject}"] = psd_df

    return payload


def build_snr_vs_success(sessions, win_sec=6.85):
    """
    Compute per-trial PSD LOSO correctness plus the Oz-channel fundamental SNR.

    This mirrors the current repo evaluation style:
    - train on one session
    - test on the held-out session
    - use PSD features + SVM

    Returns columns suited for scatter plots in Streamlit:
        subject, train_sess, test_sess, trial, target, predicted, correct, snr
    """
    from src import EEG_COLS

    oz_feature_idx = EEG_COLS.index("eeg_7") + 1  # feature columns are 1-indexed: ch7_* = Oz
    rows = []

    for subject in [1, 2]:
        for test_sess in [1, 2]:
            train_sess = 2 if test_sess == 1 else 1
            train_df = _extract_features(sessions[(subject, train_sess)], "psd", EEG_COLS, win_sec=win_sec)
            test_df = _extract_features(sessions[(subject, test_sess)], "psd", EEG_COLS, win_sec=win_sec)

            feature_cols = [c for c in train_df.columns if c.startswith("ch")]
            clf = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=1.0, gamma="scale"),
            )
            clf.fit(train_df[feature_cols], train_df["target"])
            preds = clf.predict(test_df[feature_cols])

            for row, pred in zip(test_df.to_dict("records"), preds):
                target = int(row["target"])
                snr_col = f"ch{oz_feature_idx}_snr_{target}Hz_h1"
                rows.append({
                    "subject": subject,
                    "train_sess": train_sess,
                    "test_sess": test_sess,
                    "trial": int(row["trial"]),
                    "target": float(row["target"]),
                    "predicted": float(pred),
                    "correct": bool(pred == row["target"]),
                    "snr": float(row[snr_col]),
                    "snr_col": snr_col,
                })

    return pd.DataFrame(rows).sort_values(
        ["subject", "test_sess", "trial"]
    ).reset_index(drop=True)


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


def build_story_metrics(pipeline_avg, sw_avg, itr_df, snr_vs_success):
    """
    Compute narrative-ready metrics so Streamlit can focus on presentation rather
    than re-deriving the same comparisons in the UI layer.
    """
    svm = pipeline_avg[pipeline_avg["classifier"] == "SVM"].copy()

    def _acc(feat, subject):
        values = svm[(svm["feat"] == feat) & (svm["subject"] == subject)]["accuracy_mean"]
        return float(values.iloc[0]) if not values.empty else 0.0

    def _first_window_at_or_above(subject, threshold):
        sub = sw_avg[(sw_avg["method"] == "FBCCA") & (sw_avg["subject"] == subject)]
        sub = sub.sort_values("win_sec")
        hits = sub[sub["accuracy_mean"] >= threshold]
        return float(hits.iloc[0]["win_sec"]) if not hits.empty else float(sub.iloc[-1]["win_sec"])

    def _first_window_at_pct_of_max(subject, pct=0.95):
        sub = sw_avg[(sw_avg["method"] == "FBCCA") & (sw_avg["subject"] == subject)]
        max_acc = float(sub["accuracy_mean"].max())
        return _first_window_at_or_above(subject, max_acc * pct)

    def _peak_itr(method, subject):
        sub = itr_df[(itr_df["method"] == method) & (itr_df["subject"] == subject)]
        peak = sub.loc[sub["itr"].idxmax()]
        return {
            "win_sec": float(peak["win_sec"]),
            "itr": float(peak["itr"]),
            "accuracy": float(peak["accuracy"]),
        }

    def _snr_stats(subject):
        sub = snr_vs_success[snr_vs_success["subject"] == subject]
        incorrect = sub[~sub["correct"]]
        correct = sub[sub["correct"]]
        return {
            "trial_accuracy": float(sub["correct"].mean()),
            "incorrect_rate": float((~sub["correct"]).mean()),
            "median_snr": float(sub["snr"].median()),
            "mean_snr": float(sub["snr"].mean()),
            "correct_mean_snr": float(correct["snr"].mean()) if not correct.empty else 0.0,
            "incorrect_mean_snr": float(incorrect["snr"].mean()) if not incorrect.empty else 0.0,
        }

    peak_itr_sub1 = _peak_itr("FBCCA", 1)
    peak_itr_sub2 = _peak_itr("FBCCA", 2)
    snr_sub1 = _snr_stats(1)
    snr_sub2 = _snr_stats(2)

    return {
        "baseline": {
            "psd_svm_sub1": _acc("PSD", 1),
            "psd_svm_sub2": _acc("PSD", 2),
            "fbcca_svm_sub1": _acc("FBCCA", 1),
            "fbcca_svm_sub2": _acc("FBCCA", 2),
            "fbcca_gain_vs_psd_sub1": _acc("FBCCA", 1) - _acc("PSD", 1),
            "fbcca_gain_vs_psd_sub2": _acc("FBCCA", 2) - _acc("PSD", 2),
        },
        "channel_penalty": {
            "cca_sub1": _acc("CCA", 1) - _acc("CCA-3ch", 1),
            "cca_sub2": _acc("CCA", 2) - _acc("CCA-3ch", 2),
            "fbcca_sub1": _acc("FBCCA", 1) - _acc("FBCCA-3ch", 1),
            "fbcca_sub2": _acc("FBCCA", 2) - _acc("FBCCA-3ch", 2),
        },
        "timing": {
            "fbcca_95pct_window_sub1": _first_window_at_pct_of_max(1, pct=0.95),
            "fbcca_95pct_window_sub2": _first_window_at_pct_of_max(2, pct=0.95),
            "fbcca_full_window_sub1": _first_window_at_or_above(1, 1.0),
            "fbcca_full_window_sub2": _first_window_at_or_above(2, 1.0),
            "peak_itr_sub1": peak_itr_sub1,
            "peak_itr_sub2": peak_itr_sub2,
        },
        "snr": {
            "subject_1": snr_sub1,
            "subject_2": snr_sub2,
            "incorrect_rate_gap_sub2_minus_sub1": snr_sub2["incorrect_rate"] - snr_sub1["incorrect_rate"],
        },
        "callouts": {
            "headline": (
                "FBCCA reaches perfect cross-session accuracy for both subjects, but "
                "Subject 2 needs more time and broader spatial coverage to get there."
            ),
            "tab1": (
                "Under PSD, Subject 2 behaves like the hard-responder case: "
                f"{snr_sub2['trial_accuracy']:.0%} trial accuracy vs {snr_sub1['trial_accuracy']:.0%} for Subject 1."
            ),
            "tab2": (
                "The biggest methodological rescue is Subject 2: PSD rises from "
                f"{_acc('PSD', 2):.0%} to {_acc('FBCCA', 2):.0%} with FBCCA + SVM."
            ),
            "tab3": (
                "Peak communication speed is subject-specific: Subject 1 peaks at "
                f"{peak_itr_sub1['win_sec']:.2f}s, Subject 2 at {peak_itr_sub2['win_sec']:.2f}s."
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("Generating results/data.pkl ...")

    print("  [1/6] Loading pipeline results ...")
    pipeline_raw, pipeline_avg = build_pipeline()

    print("  [2/6] Loading sliding window results + computing ITR ...")
    sw_raw, sw_avg, itr_df = build_sliding_window()

    print("  [3/6] Computing augmentation stats (loading 1 .mat file) ...")
    aug_df = build_augmentation()

    print("  [4/6] Loading preprocessed sessions ...")
    sessions = load_preprocessed_sessions()

    print("  [5/6] Building feature snapshots + SNR trial table ...")
    feature_snapshots = build_feature_snapshots(sessions)
    snr_vs_success = build_snr_vs_success(sessions)

    print("  [6/6] Building summary ...")
    summary = build_summary(pipeline_avg, itr_df)
    story = build_story_metrics(pipeline_avg, sw_avg, itr_df, snr_vs_success)

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

        # Story / Streamlit helpers based on current `src` feature extractors
        "snr_vs_success":     snr_vs_success,
        **feature_snapshots,

        # ── Summary / scorecard ───────────────────────────────────────────
        "summary":            summary,
        "story":              story,

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
            "FEATURE_SNAPSHOT_SESSION": 1,
            "SNR_CHANNEL": "Oz",
            "SNR_CHANNEL_FEATURE_PREFIX": "ch7",
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
