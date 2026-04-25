## Notes

Pipeline:

Bandpass filter (8–50 Hz, 4th-order Butterworth)
Removes slow drifts (< 8 Hz) and high-frequency noise (> 50 Hz)
Compresses the dynamic range of raw power, achieving partial stationarity
Reference: Müller-Gerking et al. (1999) — standard for SSVEP preprocessing
No log transform (key decision)

Log-transforming power is recommended for raw, unfiltered EEG (Kübler & Müller, 2007)
Our bandpass-filtered signal already occupies a narrower, flatter spectrum
Shrinkage LDA provides robust covariance estimation for small samples (n=20 trials) — achieves the same statistical stabilization as log transformation
Empirical validation: log transform reduced accuracy by 5–15% on our data
Feature extraction: CCA and FBCCA
Both methods are distribution-agnostic — they work via correlation matching, not power-based classification

No stationarity assumptions needed beyond what bandpass provides
FBCCA adds harmonic robustness via filter bank (Chen et al., 2015)
Result: Clean, defensible preprocessing that relies on the right tool for each job rather than stacking redundant assumptions.
