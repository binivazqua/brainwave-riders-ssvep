# ICA + CCA SSVEP pipeline 

from pathlib import Path
import numpy as np
import scipy.io
from scipy.stats import kurtosis
import mne
from mne.preprocessing import ICA

mne.set_log_level('WARNING')

DATA = Path('ssvep')
OUT = DATA / 'ica_output_final'
OUT.mkdir(exist_ok=True)

SFREQ = 256
CH = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
STIM = [9, 10, 12, 15]
FREQ_TO_CLASS = {15: 1, 12: 2, 10: 3, 9: 4}
CLASS_TO_FREQ = {v: k for k, v in FREQ_TO_CLASS.items()}
OZ = CH.index('Oz')

# Conservative ICA settings
KURT_THR = 6.0
HF_THR = 0.35
N_PROTECT = 4
N_HARM = 3
TMIN, TMAX = 0.5, 7.0
SNR_NEIGH = 4


def clean_int_list(values):
    return [int(v) for v in values]


def load_raw(path):
    y = scipy.io.loadmat(path)['y']
    raw = mne.io.RawArray(
        y[1:9] * 1e-6,
        mne.create_info(CH, SFREQ, 'eeg'),
        verbose=False,
    )
    raw.set_montage('standard_1020', verbose=False)
    return raw, y[9]


def preprocess(raw):
    raw = raw.copy()
    raw.notch_filter(50, method='fir', verbose=False)
    raw.filter(1, 40, method='fir', verbose=False)
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

    return mne.Epochs(
        raw,
        np.asarray(events),
        event_id={f'{f}Hz': FREQ_TO_CLASS[f] for f in STIM},
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )


def psd_fft(x):
    p = np.abs(np.fft.rfft(x, axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1], 1 / SFREQ)
    return p, f


def hsi(sources):
    """
    Harmonic SSVEP index.
    Components with strong power at stimulation frequencies and harmonics
    are protected from removal.
    """
    p, f = psd_fft(sources)
    total = p[:, (f >= 1) & (f <= 40)].sum(1) + 1e-30
    score = np.zeros(sources.shape[0])

    for stim in STIM:
        hp = np.zeros(sources.shape[0])
        for h in range(1, N_HARM + 1):
            target = stim * h
            if target > 40:
                break
            i = int(np.argmin(abs(f - target)))
            hp += p[:, max(0, i - 1): i + 2].sum(1)
        score = np.maximum(score, hp / total)

    return score


def hf_ratio(sources):
    p, f = psd_fft(sources)
    total = p[:, (f >= 1) & (f <= 40)].sum(1) + 1e-30
    return p[:, (f > 20) & (f <= 40)].sum(1) / total


def choose_bad_ics(ica, raw, sources):
    k = kurtosis(sources, axis=1, fisher=True)
    hf = hf_ratio(sources)
    hs = hsi(sources)

    bad = set(np.where((k > KURT_THR) & (hf > HF_THR))[0])

    try:
        muscle, _ = ica.find_bads_muscle(raw, verbose=False)
        bad |= {i for i in muscle if hf[i] > HF_THR}
    except Exception:
        pass

    protected = set(np.argsort(hs)[-N_PROTECT:])
    excluded = sorted(bad - protected)

    return clean_int_list(excluded), clean_int_list(sorted(bad)), clean_int_list(sorted(protected))


def apply_conservative_ica(raw):
    ica = ICA(
        n_components=len(CH),
        method='infomax',
        fit_params={'extended': True},
        max_iter='auto',
        random_state=42,
    )
    ica.fit(raw, verbose=False)

    sources = ica.get_sources(raw).get_data()
    excluded, bad, protected = choose_bad_ics(ica, raw, sources)

    raw_clean = raw.copy()
    if excluded:
        ica.exclude = excluded
        ica.apply(raw_clean, verbose=False)

    return raw_clean, excluded, bad, protected


def snr_at(x_fft, freqs, target):
    i = int(np.argmin(abs(freqs - target)))
    nb = [j for j in range(i - SNR_NEIGH - 1, i - 1) if j >= 0]
    nb += [j for j in range(i + 2, i + SNR_NEIGH + 2) if j < len(freqs)]
    noise = np.mean(x_fft[nb]) if nb else 1e-30
    return 10 * np.log10(max(x_fft[i], 1e-30) / noise)


def snr_by_freq(epochs):
    out = {}

    for f0 in STIM:
        key = f'{f0}Hz'
        if key not in epochs.event_id or len(epochs[key]) == 0:
            out[f0] = np.nan
            continue

        d = epochs[key].get_data()[:, OZ, :]
        mag = np.abs(np.fft.rfft(d, axis=1)).mean(0)
        ff = np.fft.rfftfreq(d.shape[1], 1 / SFREQ)
        out[f0] = snr_at(mag, ff, f0)

    return out


def cca_score(epoch, freq):
    t = np.arange(epoch.shape[1]) / SFREQ
    y = np.column_stack([
        fn(2 * np.pi * freq * h * t)
        for h in range(1, N_HARM + 1)
        for fn in (np.sin, np.cos)
    ])
    x = epoch.T

    def white(m):
        m = m - m.mean(0)
        _, s, vt = np.linalg.svd(m, full_matrices=False)
        return (m @ vt.T) / (s + 1e-10)

    return float(np.linalg.svd(white(x).T @ white(y), compute_uv=False)[0])


def cca_metrics(epochs):
    by_freq = {f: [] for f in STIM}
    y_true, y_pred = [], []

    for f0 in STIM:
        key = f'{f0}Hz'
        if key not in epochs.event_id:
            continue

        for ep in epochs[key].get_data():
            scores = {f: cca_score(ep, f) for f in STIM}
            predicted_freq = max(scores, key=scores.get)

            by_freq[f0].append(scores[f0])
            y_true.append(FREQ_TO_CLASS[f0])
            y_pred.append(FREQ_TO_CLASS[predicted_freq])

    acc = np.mean(np.asarray(y_true) == np.asarray(y_pred)) if y_true else np.nan
    mean_cca = {f: float(np.mean(v)) if v else np.nan for f, v in by_freq.items()}

    return acc, mean_cca


def count_epochs(epochs):
    return sum(len(epochs[k]) for k in epochs.event_id) if epochs else 0


def fmt_freqs(d, digits=2):
    return ' '.join(f'{f}:{d[f]:.{digits}f}' for f in STIM)


def mean_dict(list_of_dicts):
    return {f: float(np.nanmean([d[f] for d in list_of_dicts])) for f in STIM}


rows = []

for mat in sorted(DATA.glob('*.mat')):
    raw, trigger = load_raw(mat)
    raw_f = preprocess(raw)

    raw_clean, excluded, bad, protected = apply_conservative_ica(raw_f)
    epochs = make_epochs(raw_clean, trigger)

    if epochs is None:
        print(f'{mat.stem}: no epochs')
        continue

    acc, cca = cca_metrics(epochs)
    snr = snr_by_freq(epochs)
    n_epochs = count_epochs(epochs)

    rows.append({
        'stem': mat.stem,
        'n': n_epochs,
        'excluded': excluded,
        'bad': bad,
        'protected': protected,
        'acc': acc,
        'cca': cca,
        'snr': snr,
    })

    print(f'\n{mat.stem}')
    print(f'  IC bad:       {bad}')
    print(f'  IC protected: {protected}')
    print(f'  IC excluded:  {excluded}')
    print(f'  Epochs kept:  {n_epochs}/20')
    print(f'  Accuracy CCA: {acc:.1%}')
    print(f'  Mean CCA:     {np.nanmean(list(cca.values())):.3f}')
    print(f'  Oz SNR, dB:   {fmt_freqs(snr)}')

if rows:
    print('\n' + '=' * 72)
    print('FINAL SUMMARY')
    print('=' * 72)
    print(f"{'File':<34} {'Excl':<12} {'Epochs':>8} {'Accuracy':>10} {'Mean CCA':>10}")
    print('-' * 72)

    for r in rows:
        print(
            f"{r['stem']:<34} "
            f"{str(r['excluded']):<12} "
            f"{r['n']:>3}/20 "
            f"{r['acc']:>10.1%} "
            f"{np.nanmean(list(r['cca'].values())):>10.3f}"
        )

    acc_final = np.nanmean([r['acc'] for r in rows])
    cca_final = mean_dict([r['cca'] for r in rows])
    snr_final = mean_dict([r['snr'] for r in rows])
    total_epochs = sum(r['n'] for r in rows)
    expected_epochs = 20 * len(rows)

    print('-' * 72)
    print(f'Final mean accuracy: {acc_final:.1%}')
    print(f'Epochs kept:         {total_epochs}/{expected_epochs}')

    print('\nMean CCA by frequency:')
    for f in STIM:
        print(f'  {f:>2} Hz: {cca_final[f]:.3f}')

    print('\nMean Oz SNR by frequency, dB:')
    for f in STIM:
        print(f'  {f:>2} Hz: {snr_final[f]:+.2f}')
