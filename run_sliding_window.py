"""Standalone sliding window analysis."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import svd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

BASE     = "/Users/binivazquez/CodeWorkspace/hacks/brainwave-riders-ssvep"
DATA_DIR = f"{BASE}/data/raw/ssvep"
FIG_DIR  = f"{BASE}/results/figures"
FS = 256
STIM_FREQS = [9, 10, 12, 15]
EEG_COLS   = [f"eeg_{i}" for i in range(1, 9)]
PRE_SEC    = 0.5
WIN_SIZES  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.85]


def load_ssvep_data(filepath, stim_freqs):
    mat  = sio.loadmat(filepath)
    data = mat["y"].T  # (samples, 11)
    df = pd.DataFrame(data, columns=[
        "timestamp","eeg_1","eeg_2","eeg_3","eeg_4",
        "eeg_5","eeg_6","eeg_7","eeg_8","stimulus","classifier_output"])
    trigger = df["stimulus"].to_numpy()
    # Stimulus encodes frequency directly (0=rest, 9/10/12/15=active)
    onsets  = np.where((trigger[:-1] == 0) & (trigger[1:] != 0))[0] + 1
    offsets = np.where((trigger[:-1] != 0) & (trigger[1:] == 0))[0] + 1
    df["target Hz"] = 0
    df["trial"]     = 0
    for trial_n, (on, off) in enumerate(zip(onsets, offsets)):
        df.loc[on:off-1, "target Hz"] = trigger[on]   # label from stimulus value
        df.loc[on:off-1, "trial"]     = trial_n + 1
    return df[df["trial"] > 0].copy()


def preprocess(df):
    df  = df.copy()
    nyq = FS / 2
    sos = butter(4, [8/nyq, 50/nyq], btype="band", output="sos")
    for col in EEG_COLS:
        df[col] = sosfiltfilt(sos, df[col].to_numpy())
    return df


def create_reference(freq, n_samples, n_harmonics=3):
    t   = np.arange(n_samples) / FS
    ref = []
    for h in range(1, n_harmonics + 1):
        ref += [np.sin(2*np.pi*h*freq*t), np.cos(2*np.pi*h*freq*t)]
    return np.column_stack(ref)


def cca_score(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    Qx, sx, _ = svd(X, full_matrices=False)
    Qy, sy, _ = svd(Y, full_matrices=False)
    tol_x = X.shape[0] * np.finfo(float).eps * sx[0]
    tol_y = Y.shape[0] * np.finfo(float).eps * sy[0]
    qx = Qx[:, sx > tol_x]
    qy = Qy[:, sy > tol_y]
    c  = svd(qx.T @ qy, compute_uv=False)
    return float(np.clip(c[0], 0, 1)) if c.size else 0.0


def extract_cca(df, win_sec):
    pre   = int(PRE_SEC * FS)
    nsamp = int(win_sec * FS)
    nyq   = FS / 2
    sos   = butter(4, [8/nyq, 50/nyq], btype="band", output="sos")
    rows  = []
    for t in range(1, df["trial"].max() + 1):
        td  = df[df["trial"] == t]
        seg = td[EEG_COLS].values[pre:pre+nsamp]
        if len(seg) < nsamp:
            continue
        filt = sosfiltfilt(sos, seg, axis=0)
        feat = {"trial": t, "target": td["target Hz"].iloc[0]}
        for sf in STIM_FREQS:
            feat[f"cca_{sf}"] = cca_score(filt, create_reference(sf, nsamp))
        rows.append(feat)
    return pd.DataFrame(rows)


def extract_fbcca(df, win_sec):
    FB    = [(8,50),(14,50),(22,50),(30,50),(38,50)]
    wts   = np.array([(k+1)**-1.25 + 0.25 for k in range(5)])
    pre   = int(PRE_SEC * FS)
    nsamp = int(win_sec * FS)
    rows  = []
    for t in range(1, df["trial"].max() + 1):
        td  = df[df["trial"] == t]
        seg = td[EEG_COLS].values[pre:pre+nsamp]
        if len(seg) < nsamp:
            continue
        feat = {"trial": t, "target": td["target Hz"].iloc[0]}
        for sf in STIM_FREQS:
            ref   = create_reference(sf, nsamp)
            score = 0.0
            for k, (lo, hi) in enumerate(FB):
                nyq = FS / 2
                sos = butter(4, [lo/nyq, hi/nyq], btype="band", output="sos")
                score += wts[k] * cca_score(sosfiltfilt(sos, seg, axis=0), ref)**2
            feat[f"fbcca_{sf}"] = score
        rows.append(feat)
    return pd.DataFrame(rows)


# Load sessions
print("Loading & preprocessing data...")
FILES = {
    (1,1): f"{DATA_DIR}/subject_1_fvep_led_training_1.mat",
    (1,2): f"{DATA_DIR}/subject_1_fvep_led_training_2.mat",
    (2,1): f"{DATA_DIR}/subject_2_fvep_led_training_1.mat",
    (2,2): f"{DATA_DIR}/subject_2_fvep_led_training_2.mat",
}
sessions = {k: preprocess(load_ssvep_data(v, STIM_FREQS)) for k, v in FILES.items()}
print({k: sessions[k]["trial"].nunique() for k in sessions}, "trials per session\n")

# Main loop
print(f"Running {len(WIN_SIZES)} windows x 2 methods x 4 LOSO folds...")
results = []
for win_sec in WIN_SIZES:
    print(f"  win={win_sec:.2f}s", end="  ", flush=True)
    for method, fn, prefix in [
        ("CCA",   extract_cca,   "cca_"),
        ("FBCCA", extract_fbcca, "fbcca_"),
    ]:
        feats = {k: fn(v, win_sec) for k, v in sessions.items()}
        for subj in [1, 2]:
            for test_s in [1, 2]:
                train_s = 2 if test_s == 1 else 1
                tr   = feats[(subj, train_s)]
                te   = feats[(subj, test_s)]
                cols = [c for c in tr.columns if c.startswith(prefix)]
                sc   = StandardScaler()
                Xtr  = sc.fit_transform(tr[cols])
                Xte  = sc.transform(te[cols])
                clf  = SVC(kernel="rbf", C=1.0, gamma="scale")
                clf.fit(Xtr, tr["target"])
                acc  = clf.score(Xte, te["target"])
                results.append(dict(win_sec=win_sec, method=method,
                                    subject=subj, test_sess=test_s, accuracy=acc))
    print("done")

from scipy.stats import binomtest

df_res = pd.DataFrame(results)
df_res.to_csv(f"{BASE}/results/sliding_window_results.csv", index=False)
print(f"\nSaved CSV -> {BASE}/results/sliding_window_results.csv")

# ── Binomial significance per condition ───────────────────────────────────
N_TRIALS  = 20          # test trials per LOSO fold
N_CLASSES = 4           # chance = 1/4 = 25%
CHANCE    = 1 / N_CLASSES

def stars(p):
    if p < 0.001: return "★★★"
    if p < 0.01:  return "★★"
    if p < 0.05:  return "★"
    return "ns"

def binom_p(acc, n=N_TRIALS, p0=CHANCE):
    k = round(acc * n)
    return binomtest(k, n=n, p=p0, alternative="greater").pvalue

# Build significance table: mean acc → p-value → stars, per (method, subject, win_sec)
sig = {}
for method in ["CCA", "FBCCA"]:
    for subj in [1, 2]:
        agg = (df_res[(df_res["method"] == method) & (df_res["subject"] == subj)]
               .groupby("win_sec")["accuracy"].mean())
        for win, acc in agg.items():
            p = binom_p(acc)
            sig[(method, subj, win)] = (p, stars(p))

# ── Plot with significance annotations ───────────────────────────────────
COLORS  = {"CCA": "#4878CF", "FBCCA": "#E24A33"}
MARKERS = {"CCA": "o",       "FBCCA": "s"}
OFFSETS = {"CCA": -0.08,     "FBCCA": 0.08}   # vertical nudge so stars don't overlap

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
fig.suptitle("Sliding Window Analysis: Accuracy vs Window Length\n"
             "(★ p<0.05  ★★ p<0.01  ★★★ p<0.001  vs. 25% chance, binomial test)",
             fontsize=13, fontweight="bold")

for ax, subj in zip(axes, [1, 2]):
    for method in ["CCA", "FBCCA"]:
        agg = (df_res[(df_res["subject"] == subj) & (df_res["method"] == method)]
               .groupby("win_sec")["accuracy"]
               .agg(["mean", "std"]).reset_index())

        ax.plot(agg["win_sec"], agg["mean"],
                color=COLORS[method], marker=MARKERS[method],
                linewidth=2.5, markersize=7, label=method, zorder=3)
        ax.fill_between(agg["win_sec"],
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        alpha=0.15, color=COLORS[method])

        # Annotate each point with significance stars
        for _, row in agg.iterrows():
            win  = row["win_sec"]
            mean = row["mean"]
            p, s = sig[(method, subj, win)]
            if s != "ns":
                y_pos = min(mean + OFFSETS[method] + 0.03, 1.03)
                ax.text(win, y_pos, s, ha="center", va="bottom",
                        fontsize=7.5, color=COLORS[method], fontweight="bold")

    ax.set_title(f"Subject {subj}", fontsize=12)
    ax.set_xlabel("Window Length (s)", fontsize=11)
    if subj == 1:
        ax.set_ylabel("Classification Accuracy", fontsize=11)
    ax.set_xticks(WIN_SIZES)
    ax.set_xticklabels([str(w) for w in WIN_SIZES], rotation=30, ha="right")
    ax.set_ylim(0, 1.12)
    ax.axhline(CHANCE, color="gray", ls="--", lw=1, alpha=0.6, label="Chance (25%)")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
for p in [f"{FIG_DIR}/sliding_window.png",
          f"{FIG_DIR}/discord/08_sliding_window.png"]:
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved -> {p}")

# ── Summary table with p-values ───────────────────────────────────────────
print("\n--- Significance table (binomial test vs 25% chance) ---")
print(f"{'Method':<8} {'Win':>5} {'Sub1 acc':>9} {'p':>10} {'sig':>5}  "
      f"{'Sub2 acc':>9} {'p':>10} {'sig':>5}")
print("-" * 68)
for method in ["CCA", "FBCCA"]:
    for win in WIN_SIZES:
        row_parts = [f"{method:<8}", f"{win:>5.2f}"]
        for subj in [1, 2]:
            agg = (df_res[(df_res["method"] == method) &
                          (df_res["subject"] == subj) &
                          (df_res["win_sec"] == win)]["accuracy"].mean())
            p, s = sig[(method, subj, win)]
            row_parts += [f"{agg:>9.1%}", f"{p:>10.2e}", f"{s:>5}"]
        print("  ".join(row_parts))

print("\n--- Saturation (first win reaching 95% of max) ---")
for method in ["CCA", "FBCCA"]:
    for subj in [1, 2]:
        row = (df_res[(df_res["method"] == method) & (df_res["subject"] == subj)]
               .groupby("win_sec")["accuracy"].mean().sort_index())
        mx  = row.max()
        sat = next((w for w, a in row.items() if a >= 0.95 * mx), row.index[-1])
        _, s = sig[(method, subj, sat)]
        print(f"  {method} Sub{subj}: max={mx:.1%}  95%-sat @ {sat:.1f}s  [{s}]")
