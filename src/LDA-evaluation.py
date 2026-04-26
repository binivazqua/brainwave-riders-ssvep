from pathlib import Path
from collections import Counter

import numpy as np
import scipy.io


DATA = Path("ssvep")

# Channel layout:
# CH1       time
# CH2-CH9   EEG channels
# CH10      true stimulus frequency: 0 / 9 / 10 / 12 / 15 Hz
# CH11      LDA classifier output: 0=no answer, 1=15Hz, 2=12Hz, 3=10Hz, 4=9Hz
CLASS_TO_FREQ = {1: 15, 2: 12, 3: 10, 4: 9}
FREQ_TO_CLASS = {v: k for k, v in CLASS_TO_FREQ.items()}


def label(class_id):
    return "no answer" if class_id == 0 else f"{CLASS_TO_FREQ[class_id]} Hz"


def most_common_nonzero(values):
    values = values[values > 0].astype(int)
    return Counter(values).most_common(1)[0][0] if len(values) else 0


def get_trials(y):
    ch10_true = y[9]
    ch11_lda = y[10]

    active = ch10_true > 0

    starts = np.where(np.diff(active.astype(int)) == 1)[0] + 1
    ends = np.where(np.diff(active.astype(int)) == -1)[0] + 1

    if len(ends) < len(starts):
        ends = np.append(ends, y.shape[1])

    trials = []

    for i, (start, end) in enumerate(zip(starts, ends), 1):
        true_class = FREQ_TO_CLASS[int(ch10_true[start])]
        lda_class = most_common_nonzero(ch11_lda[start:end])

        trials.append({
            "trial": i,
            "true": true_class,
            "lda": lda_class,
            "correct": true_class == lda_class,
            "no_answer": lda_class == 0,
        })

    return trials


def metrics(trials):
    total = len(trials)
    correct = sum(t["correct"] for t in trials)
    no_answer = sum(t["no_answer"] for t in trials)
    wrong_label = total - correct - no_answer

    return {
        "total": total,
        "correct": correct,
        "wrong_label": wrong_label,
        "no_answer": no_answer,
        "total_wrong": total - correct,
        "accuracy": 100 * correct / total if total else 0,
    }


def print_metrics(m):
    print(
        f"{'Trials':<12}"
        f"{'Correct':<12}"
        f"{'Wrong LDA':<14}"
        f"{'No answer':<14}"
        f"{'Total wrong':<14}"
        f"{'Accuracy'}"
    )
    print("-" * 78)
    print(
        f"{m['total']:<12}"
        f"{m['correct']:<12}"
        f"{m['wrong_label']:<14}"
        f"{m['no_answer']:<14}"
        f"{m['total_wrong']:<14}"
        f"{m['accuracy']:.1f}%"
    )


def print_report(title, trials, show_trials=True):
    m = metrics(trials)

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    print_metrics(m)

    if not show_trials:
        return

    print()
    print(f"{'Trial':<8}{'CH10 true':<14}{'CH11 LDA':<14}{'Result'}")
    print("-" * 50)

    for t in trials:
        if t["correct"]:
            result = "correct"
        elif t["no_answer"]:
            result = "no answer"
        else:
            result = "wrong"

        print(
            f"{t['trial']:<8}"
            f"{label(t['true']):<14}"
            f"{label(t['lda']):<14}"
            f"{result}"
        )


all_trials = []

for file in sorted(DATA.glob("*.mat")):
    y = scipy.io.loadmat(file)["y"]

    trials = get_trials(y)
    all_trials.extend(trials)

    print_report(file.name, trials)

print_report("TOTAL RESULT", all_trials, show_trials=False)
