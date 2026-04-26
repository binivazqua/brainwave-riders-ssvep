# FINAL2 CLabel + FBCCA/CCA SSVEP pipeline

from pathlib import Path
import numpy as np
import scipy.io
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

# =========================
# CONFIG
# =========================

DATA = Path("ssvep")

SFREQ = 256
CH_NAMES = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

STIM_FREQS = [9, 10, 12, 15]
N_HARM = 3

CLASS_TO_FREQ = {1: 15, 2: 12, 3: 10, 4: 9}
FREQ_TO_CLASS = {v: k for k, v in CLASS_TO_FREQ.items()}

EPOCH_TMIN = 0.5
EPOCH_TMAX = 7.0

KEEP_LABELS = {"brain", "other"}
N_SSVEP_PROTECTED = 3

FBCCA_BANDS = [(6, 90), (14, 90), (22, 90), (30, 90), (38, 90)]
FBCCA_WEIGHTS = [(m + 1) ** (-1.25) + 0.25 for m in range(len(FBCCA_BANDS))]
ENS_FB_WEIGHT = 2.0


# =========================
# DATA
# =========================

def load_raw(path):
    y = scipy.io.loadmat(path)["y"]

    raw = mne.io.RawArray(
        y[1:9] * 1e-6,
        mne.create_info(CH_NAMES, SFREQ, ch_types="eeg"),
        verbose=False
    )

    raw.set_montage("standard_1020", verbose=False)
    trigger = y[9]

    return raw, trigger


def preprocess(raw):
    raw = raw.copy()
    raw.notch_filter(50.0, method="fir", verbose=False)
    raw.filter(1.0, 100.0, method="fir", verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    return raw


def make_epochs(raw, trigger):
    starts = np.where(np.diff((trigger > 0).astype(int)) == 1)[0] + 1

    events = [
        [s, 0, FREQ_TO_CLASS[int(trigger[s])]]
        for s in starts
        if int(trigger[s]) in FREQ_TO_CLASS
    ]

    if not events:
        return None

    event_id = {f"{f}Hz": FREQ_TO_CLASS[f] for f in STIM_FREQS}

    return mne.Epochs(
        raw,
        np.array(events),
        event_id=event_id,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=None,
        preload=True,
        verbose=False
    )


# =========================
# ICA + ICLABEL
# =========================

def harmonic_sum_index(sources):
    psd = np.abs(np.fft.rfft(sources, axis=1)) ** 2
    freqs = np.fft.rfftfreq(sources.shape[1], 1.0 / SFREQ)

    band = (freqs >= 1) & (freqs <= 100)
    total_power = psd[:, band].sum(axis=1) + 1e-30

    hsi = np.zeros(sources.shape[0])

    for f in STIM_FREQS:
        harmonic_power = np.zeros(sources.shape[0])

        for h in range(1, N_HARM + 1):
            target = f * h
            if target > 100:
                break

            idx = np.argmin(np.abs(freqs - target))
            harmonic_power += psd[:, max(0, idx - 1):idx + 2].sum(axis=1)

        hsi = np.maximum(hsi, harmonic_power / total_power)

    return hsi


def clean_with_icalabel(raw):
    ica = ICA(
        n_components=len(CH_NAMES),
        method="infomax",
        fit_params={"extended": True},
        max_iter="auto",
        random_state=42
    )

    ica.fit(raw, verbose=False)

    sources = ica.get_sources(raw).get_data()
    hsi = harmonic_sum_index(sources)

    labels_info = label_components(raw, ica, method="iclabel")
    labels = labels_info["labels"]
    probs = labels_info["y_pred_proba"]

    protected = set(np.argsort(hsi)[-N_SSVEP_PROTECTED:])

    excluded = sorted(
        {i for i, label in enumerate(labels) if label not in KEEP_LABELS}
        - protected
    )

    for i, label in enumerate(labels):
        tag = ""
        if i in protected:
            tag = " [prot]"
        elif i in excluded:
            tag = " [EXCL]"

        print(
            f"    IC {i + 1:<2d} {label:<10s} "
            f"p={probs[i].max():.2f}  HSI={hsi[i]:.3f}{tag}"
        )

    print(f"  Excluded: {excluded}")

    if excluded:
        raw_clean = raw.copy()
        ica.exclude = excluded
        ica.apply(raw_clean, verbose=False)
        return raw_clean

    return raw


# =========================
# CLASSIFICATION
# =========================

def cca_score(data, freq):
    n_t = data.shape[1]
    t = np.arange(n_t) / SFREQ

    ref = np.column_stack([
        fn(2 * np.pi * freq * h * t)
        for h in range(1, N_HARM + 1)
        for fn in (np.sin, np.cos)
    ])

    x = data.T - data.T.mean(axis=0)
    y = ref - ref.mean(axis=0)

    def whiten(m):
        _, s, vt = np.linalg.svd(m, full_matrices=False)
        return (m @ vt.T) / (s + 1e-10)

    return np.linalg.svd(whiten(x).T @ whiten(y), compute_uv=False)[0]


def fbcca_score(data, freq):
    score = 0.0

    for band, weight in zip(FBCCA_BANDS, FBCCA_WEIGHTS):
        low, high = band
        high = min(high, SFREQ / 2 - 1)

        if low < high:
            filtered = mne.filter.filter_data(
                data.astype(float),
                SFREQ,
                low,
                high,
                method="fir",
                verbose=False
            )
        else:
            filtered = data

        score += weight * cca_score(filtered, freq) ** 2

    return score


def predict_epoch(epoch):
    fbcca = np.array([fbcca_score(epoch, f) for f in STIM_FREQS])
    cca = np.array([cca_score(epoch, f) for f in STIM_FREQS])

    combined = (
        ENS_FB_WEIGHT * fbcca / (fbcca.sum() + 1e-10)
        + cca / (cca.sum() + 1e-10)
    )

    best_freq = STIM_FREQS[np.argmax(combined)]
    return FREQ_TO_CLASS[best_freq]


def evaluate(epochs):
    y_true = []
    y_pred = []

    for freq in STIM_FREQS:
        key = f"{freq}Hz"

        if key not in epochs.event_id:
            continue

        for epoch in epochs[key].get_data():
            y_true.append(FREQ_TO_CLASS[freq])
            y_pred.append(predict_epoch(epoch))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred) if len(y_true) else 0.0

    return accuracy, y_true, y_pred


# =========================
# SNR
# =========================

def oz_snr(epochs, n_nb=4):
    oz_idx = CH_NAMES.index("Oz")
    snrs = {}

    for freq in STIM_FREQS:
        key = f"{freq}Hz"

        if key not in epochs.event_id:
            continue

        data = epochs[key].get_data()[:, oz_idx, :]
        spectrum = np.abs(np.fft.rfft(data, axis=1)).mean(axis=0)
        freqs = np.fft.rfftfreq(data.shape[1], 1.0 / SFREQ)

        idx = np.argmin(np.abs(freqs - freq))

        left = [i for i in range(idx - n_nb - 1, idx - 1) if i >= 0]
        right = [i for i in range(idx + 2, idx + n_nb + 2) if i < len(freqs)]

        noise = np.mean(spectrum[left + right]) if left + right else 1e-30
        snrs[freq] = 10 * np.log10(max(spectrum[idx], 1e-30) / noise)

    return snrs


# =========================
# MAIN
# =========================

results = []

for mat_file in sorted(DATA.glob("*.mat")):
    print("\n" + "=" * 60)
    print(f"  {mat_file.stem}")
    print("=" * 60)

    raw, trigger = load_raw(mat_file)
    raw = preprocess(raw)
    raw = clean_with_icalabel(raw)

    epochs = make_epochs(raw, trigger)

    if epochs is None:
        print("  No valid epochs, skipped.")
        continue

    acc, y_true, y_pred = evaluate(epochs)
    snr = oz_snr(epochs)

    n_epochs = len(y_true)

    print(f"  Epochs  : {n_epochs}")
    print(f"  Accuracy: {acc:.1%}  {'✓' if acc >= 0.95 else '✗'}")
    print(f"  SNR(Oz) : { {f: f'{v:+.1f} dB' for f, v in snr.items()} }")

    results.append({
        "file": mat_file.stem,
        "acc": acc,
        "snr": snr,
        "n_epochs": n_epochs
    })


# =========================
# SUMMARY
# =========================

print("\n" + "=" * 60)
print("  GLOBAL SUMMARY")
print("=" * 60)

print(f"  {'File':<45} {'Acc':>7} {'9Hz':>7} {'10Hz':>7} {'12Hz':>7} {'15Hz':>7}")
print("  " + "-" * 82)

for r in results:
    ok = "✓" if r["acc"] >= 0.95 else "✗"

    snr_values = " ".join(
        f"{r['snr'].get(f, float('nan')):>7.1f}"
        for f in STIM_FREQS
    )

    print(f"  {ok} {r['file']:<43} {r['acc']:>7.1%} {snr_values}")

if results:
    print("  " + "-" * 82)

    mean_acc = np.mean([r["acc"] for r in results])
    passed = sum(r["acc"] >= 0.95 for r in results)

    print(f"  Mean: {mean_acc:.1%}  |  ≥95% in {passed}/{len(results)} files")
