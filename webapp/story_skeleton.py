"""
story_skeleton.py — Narrative data map for the Streamlit board based on the document

One source of truth: results/data.pkl
Load it once at the top of app.py and pass `data` everywhere.

    import pickle
    with open("results/data.pkl", "rb") as f:
        data = pickle.load(f)

Sections follow the PDF draft in order.
Each block shows exactly what to access and what shape/values to expect.
This file is aligned to the current `webapp/generate_pickle.py` output.
"""

# ══════════════════════════════════════════════════════════════════════════════
# HEADER BLOCK
# ══════════════════════════════════════════════════════════════════════════════
#
# Four scorecard chips across the top.

HEADER = {
    # "Best Accuracy: 100%"
    "best_accuracy": {
        "sub1": "data['summary']['fbcca_acc_sub1']",   # → 1.0
        "sub2": "data['summary']['fbcca_acc_sub2']",   # → 1.0
        "label": "FBCCA + SVM · LOSO",
    },

    # "Peak ITR: 21.7 bits/min"
    "peak_itr": {
        "value":   "data['summary']['best_itr_sub1']",     # → 21.67 bits/min
        "at_window": "data['summary']['best_itr_win_sub1']",  # → 1.0 s
        "label": "Subject 1 · 1 s window",
    },

    # "Biggest Rescue: +65 pts"
    "largest_gain": {
        "value": "data['story']['baseline']['fbcca_gain_vs_psd_sub2']",  # → 0.65
        "label": "Subject 2 · PSD → FBCCA",
        "note": "The weakest baseline shows the strongest algorithmic rescue.",
    },

    # "p < 0.001 ★★★"
    # ⚠ NEEDS COMPUTE — binomial test is not yet in the pickle.
    # Formula: from scipy.stats import binom_test (or binomtest)
    # binom_test(k=20, n=20, p=0.25, alternative='greater') → p ≈ 2.8e-12
    # Hardcode is fine: all FBCCA/CCA conditions are p < 0.001.
    "stat_sig": {
        "p_value": "< 0.001",
        "stars":   "★★★",
        "note":    "Binomial test vs 25% chance, n=20 trials, k=20 correct",
    },

    # "Adaptive Need: 2s vs 6s"
    "adaptive_need": {
        "sub1_95pct": "data['story']['timing']['fbcca_95pct_window_sub1']",  # → 2.0
        "sub2_95pct": "data['story']['timing']['fbcca_95pct_window_sub2']",  # → 6.0
        "note":   "95% of max accuracy arrives fast for Subject 1 and much later for Subject 2.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SNR & Variability  ("The BCI-Illiteracy Spectrum")
# ══════════════════════════════════════════════════════════════════════════════

TAB1 = {

    # ── Plot A: SNR vs. PSD Success (scatter) ─────────────────────────────
    # Show per-trial SNR on x-axis, binary correct/incorrect on y-axis.
    # Colour by subject. Jitter or facet by subject works best.
    "snr_vs_success": {
        "source":  "data['snr_vs_success']",
        "x":       "snr  (Oz channel, fundamental harmonic, Welch; feature prefix ch7_*)",
        "y":       "correct  (bool: PSD prediction == target)",
        "colour":  "subject",
        "shape":   "(80, 9)",
        "insight": (
            "This is the fragility plot, not a simple correlation plot: "
            "Subject 2 fails much more often under PSD even before you get to advanced methods."
        ),
        "pull_quotes": {
            "sub1_trial_accuracy": "data['story']['snr']['subject_1']['trial_accuracy']",  # → 0.875
            "sub2_trial_accuracy": "data['story']['snr']['subject_2']['trial_accuracy']",  # → 0.375
            "incorrect_rate_gap": "data['story']['snr']['incorrect_rate_gap_sub2_minus_sub1']",  # → 0.50
        },
    },

    # ── Plot B: Lateralization — per-channel accuracy heatmap ─────────────
    # Bar chart or small heatmap: 8-channel accuracy vs 3-channel (O1/Oz/O2).
    # Shows Sub2 is PO7/PO8-dominant, Sub1 purely occipital.
    #
    # Data IS in pickle — use channel-comparison rows from pipeline_avg.
    "lateralization": {
        "source": "data['pipeline_avg']",
        "filter": """
            import pandas as pd
            pa = data['pipeline_avg']

            # 8-ch (PO7…O2)
            full_ch = pa[
                pa['feat'].isin(['CCA', 'FBCCA']) & (pa['classifier'] == 'SVM')
            ][['feat', 'subject', 'accuracy_mean']]

            # 3-ch occipital (O1/Oz/O2 = eeg_6/7/8)
            occ_ch = pa[
                pa['feat'].isin(['CCA-3ch', 'FBCCA-3ch']) & (pa['classifier'] == 'SVM')
            ][['feat', 'subject', 'accuracy_mean']]
        """,
        "chart":   "Grouped bars: Subject 1 vs Subject 2 · 8ch vs 3ch · FBCCA & CCA",
        "insight": (
            "Sub1 3ch ≈ 8ch → purely occipital. "
            "Sub2 3ch < 8ch → lateral channels (PO7/PO8) carry signal. "
            "Standard 3-ch montage missed Subject 2."
        ),
        "penalties": {
            "fbcca_sub1": "data['story']['channel_penalty']['fbcca_sub1']",  # → 0.0
            "fbcca_sub2": "data['story']['channel_penalty']['fbcca_sub2']",  # → 0.35
        },
    },

    # ── Neuroscience callout (text card) ──────────────────────────────────
    "bci_illiteracy_callout": {
        "header": "The BCI-Illiteracy Spectrum",
        "body": (
            "~15–30% of users show weak or absent SSVEP responses under standard montages. "
            "Subject 2 exemplifies this: PSD reached only 35% accuracy (chance = 25%) while "
            "FBCCA — which requires no SNR assumptions — achieved 100%. "
            "SNR-robust, training-free methods are not optional for clinical deployment."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Comparative Methodology
# ══════════════════════════════════════════════════════════════════════════════

TAB2 = {

    # ── Plot A: PCA projection — CCA vs PSD feature space ─────────────────
    # 2-D scatter after PCA(n=2). One plot per method.
    # CCA: clean 4-cluster separation. PSD: overlapping blobs.
    "pca_feature_space": {
        "source":    "data['cca_features_sub{1,2}'] / data['psd_features_sub{1,2}']",
        "cca_cols":  "['cca_9Hz', 'cca_10Hz', 'cca_12Hz', 'cca_15Hz']",
        "psd_cols":  "[c for c in df.columns if c.startswith('ch') and 'psd' in c and 'h1' in c]",
        "colour":    "target  (9/10/12/15 Hz)",
        "chart":     "Side-by-side scatter: PCA(CCA) vs PCA(PSD), one panel per subject",
        "insight":   "CCA maps to 4 clean clusters; PSD overlaps — explains the accuracy gap",
        "shapes": {
            "cca_sub1": "(20, 8)",
            "cca_sub2": "(20, 8)",
            "psd_sub1": "(20, 228)",
            "psd_sub2": "(20, 228)",
        },
    },

    # ── Plot B: Algorithm benchmarking table ──────────────────────────────
    # Simple table: rows = methods, cols = Sub1 acc / Sub2 acc.
    # All from pickle except eTRCA which we hardcode from the notebook run.
    "benchmark_table": {
        "source": "data['pipeline_avg']",
        "build": """
            import pandas as pd
            pa = data['pipeline_avg']
            svm = pa[pa['classifier'] == 'SVM'].copy()

            # Pivot to wide: index=feat, columns=subject
            tbl = svm.pivot_table(
                index='feat', columns='subject',
                values='accuracy_mean'
            ).rename(columns={1: 'Subject 1', 2: 'Subject 2'})

            # Add eTRCA row (from notebook run — hardcoded)
            etrca_row = pd.DataFrame(
                [{'feat': 'eTRCA', 'Subject 1': 0.775, 'Subject 2': 0.200}]
            ).set_index('feat')
            tbl = pd.concat([tbl, etrca_row])

            # Display order
            order = ['PSD', 'CCA', 'CCA-3ch', 'FBCCA', 'FBCCA-3ch', 'eTRCA']
            tbl = tbl.reindex([o for o in order if o in tbl.index])
        """,
        "format":  "Highlight FBCCA row. Color scale green (high) → red (low).",
        "insight":  "FBCCA wins on both subjects; eTRCA collapses on Subject 2 (data-hungry).",
        "story_hook": "data['story']['callouts']['tab2']",
    },

    # ── Text card: Explainable AI — why eTRCA failed ──────────────────────
    "etrca_explanation": {
        "header": "Why eTRCA Underperformed",
        "bullets": [
            "Data-hungry: needs ≥15 training trials/freq. We have 5 under LOSO.",
            "Inter-trial covariance (S matrix) is near-singular at N=5 — spatial filter is noise.",
            "Non-stationary sessions: filter learned on session A doesn't generalise to session B.",
            "FBCCA uses a mathematical reference (sine/cosine) — zero training data needed.",
        ],
        "numbers": {
            "etrca_sub1": 0.775,   # mean LOSO accuracy
            "etrca_sub2": 0.200,
            "fbcca_sub1": "data['summary']['fbcca_acc_sub1']",  # 1.0
            "fbcca_sub2": "data['summary']['fbcca_acc_sub2']",  # 1.0
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — The Neuroscience of Utility
# ══════════════════════════════════════════════════════════════════════════════

TAB3 = {

    # ── Plot A: Accuracy vs Window Length (line chart) ────────────────────
    # x = win_sec, y = accuracy_mean, lines = Subject 1 / Subject 2.
    # Filter to FBCCA only. Annotate saturation points.
    "accuracy_vs_window": {
        "source": "data['sliding_window_avg']",
        "filter": """
            sw = data['sliding_window_avg']
            fbcca = sw[sw['method'] == 'FBCCA']

            sub1 = fbcca[fbcca['subject'] == 1][['win_sec', 'accuracy_mean', 'accuracy_std']]
            sub2 = fbcca[fbcca['subject'] == 2][['win_sec', 'accuracy_mean', 'accuracy_std']]
        """,
        "annotations": {
            "sub1_saturates": "win_sec == 3.0  → accuracy_mean == 1.0  (annotate '3s: 100%')",
            "sub2_peak":      "win_sec == 6.85 → accuracy_mean == 1.0  (annotate '6.85s: 100%')",
        },
        "chart":   "Line chart with shaded std band. Dashed horizontal at 100%. Dashed vertical at 3s.",
        "insight": "Sub1 saturates at 3s — visual cortex is highly entrained. Sub2 needs full epoch.",
        "95pct_windows": {
            "sub1": "data['story']['timing']['fbcca_95pct_window_sub1']",  # → 2.0
            "sub2": "data['story']['timing']['fbcca_95pct_window_sub2']",  # → 6.0
        },
    },

    # ── Plot B: ITR vs Window Length (line chart) ─────────────────────────
    # Same x-axis, y = ITR (bits/min). Highlight peak per subject.
    "itr_vs_window": {
        "source": "data['itr']",
        "filter": """
            itr = data['itr']
            fbcca_itr = itr[itr['method'] == 'FBCCA']

            sub1_itr = fbcca_itr[fbcca_itr['subject'] == 1][['win_sec', 'itr', 'accuracy']]
            sub2_itr = fbcca_itr[fbcca_itr['subject'] == 2][['win_sec', 'itr', 'accuracy']]

            # Peak rows
            sub1_peak = sub1_itr.loc[sub1_itr['itr'].idxmax()]
            # → win_sec=1.0, itr=21.67, accuracy=0.925

            sub2_peak = sub2_itr.loc[sub2_itr['itr'].idxmax()]
            # → win_sec=6.85, itr=12.0, accuracy=1.0
        """,
        "annotations": {
            "sub1_peak": "Star marker at (1s, 21.7 bits/min) — '92.5% acc but peak speed'",
            "sub2_peak": "Star marker at (6.85s, 12.0 bits/min) — '100% acc, longer window'",
        },
        "chart":   "Line chart. Star markers at peaks. Reference line at 12 bits/min (standard BCI benchmark).",
        "insight": (
            "100% accuracy ≠ optimal communication speed. "
            "Subject 1 peaks at 1s/92.5% — 21.7 bits/min. "
            "Subject 2 peaks at 6.85s/100% — 12.0 bits/min. "
            "Optimal window is subject-specific."
        ),
        "story_hook": "data['story']['callouts']['tab3']",
    },

    # ── Text card: Call to Action ─────────────────────────────────────────
    "adaptive_bci_cta": {
        "header": "Toward Adaptive BCIs",
        "body": (
            "A fixed decision window ignores inter-subject variability. "
            "The next step: monitor real-time SNR and dynamically shrink the window "
            "when the signal is strong (high-SNR trial → 1s decision) "
            "and extend it when it is weak (low-SNR → 6.85s). "
            "This closes the gap between Subject 1 and Subject 2 without sacrificing accuracy."
        ),
        "summary_numbers": {
            "best_itr_sub1": "data['summary']['best_itr_sub1']",   # 21.67
            "best_itr_sub2": "data['summary']['best_itr_sub2']",   # 12.0
            "best_win_sub1": "data['summary']['best_itr_win_sub1']",  # 1.0 s
            "best_win_sub2": "data['summary']['best_itr_win_sub2']",  # 6.85 s
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Interactive Simulator
# ══════════════════════════════════════════════════════════════════════════════

TAB4 = {
    # Embed the pre-built Plotly animation HTML.
    # No pickle data needed — the HTML is self-contained.
    "simulator": {
        "source": "results/simulator.html",
        "embed": """
            import streamlit as st
            import streamlit.components.v1 as components
            html = open("results/simulator.html", encoding="utf-8").read()
            components.html(html, height=620, scrolling=False)
        """,
        "description": (
            "Animated bar chart. Slider steps through window sizes 1→2→3→4→5→6→6.85s. "
            "4 bars: CCA Sub1, CCA Sub2, FBCCA Sub1, FBCCA Sub2. "
            "ITR value displayed above bars. Accuracy % inside bars. "
            "Play/Pause buttons included."
        ),
        "aesthetic_note": (
            "Dark theme already set (paper_bgcolor='#0e1117'). "
            "Teammate can restyle fonts/colours without regenerating data."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PICKLE COVERAGE — what is already present vs still hardcoded
# ══════════════════════════════════════════════════════════════════════════════

MISSING = {
    "already_in_pickle": (
        "data['snr_vs_success'], data['cca_features_sub1'], data['cca_features_sub2'], "
        "data['psd_features_sub1'], data['psd_features_sub2'], and data['story'] "
        "are now generated in webapp/generate_pickle.py."
    ),
    "etrca_results": (
        "Tab 2 / Table. "
        "Currently hardcoded: Sub1=77.5%, Sub2=20.0%. "
        "Source: notebooks/explo.ipynb cells 50-52. "
        "Could add data['etrca_results'] if table needs to be dynamic."
    ),
}
