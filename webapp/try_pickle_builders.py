"""Small smoke test for the extra pickle builders.

Usage:
    python3 webapp/try_pickle_builders.py
"""

from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT))

from generate_pickle import (
    build_feature_snapshots,
    build_snr_vs_success,
    load_preprocessed_sessions,
)


def main():
    sessions = load_preprocessed_sessions()
    snapshots = build_feature_snapshots(sessions)
    snr_vs_success = build_snr_vs_success(sessions)

    print("Loaded sessions:", sorted(sessions))
    print()

    print("Feature snapshot previews:")
    for key in ["cca_features_sub1", "cca_features_sub2", "psd_features_sub1", "psd_features_sub2"]:
        df = snapshots[key]
        print(f"  {key}: shape={df.shape}")
        print(f"    columns={df.columns[:8].tolist()} ...")

    print()
    print(f"snr_vs_success: shape={snr_vs_success.shape}")
    print(snr_vs_success.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
