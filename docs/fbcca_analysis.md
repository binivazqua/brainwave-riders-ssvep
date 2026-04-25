# Insightful FBCCA analysis: the layer cake

     FBCCA = 100% everywhere when along SVM.

![](/results/figures/discord/09_fbcca_gain.png)

## The "problems"

![](/results/figures/discord/01_psd_per_class_all_sessions.png)

PSD features look clean for Subject 1 but Subject 2 SNR is weaker, clearer here:

![](/results/figures/discord/02_snr_heatmap_comparison.png)

     Diagonal is key!!

### CCA enters the chat

CCA heatmap is much stronger diagonal for both subjects.

> Feature space PCA shows clean class separation with CCA vs messy blobs with PSD. Diagonal is way clearer!

![](/results/figures/discord/03_cca_heatmap_comparison.png)

### CROSS SESSION spices things up yeah

What we see:

- Subject 1 sessions overlap well (circles + squares mixed)
- Subject 2 has more drift between sessions.

> Hypotheses: session variability, external factors affecting signal quality.

![](/results/figures/discord/06_cross_session_pca_cca.png)

### SNR to test the hypotheses

Low SNR trials are the ones that fail cross-session transfer with PSD!!!

![](/results/figures/discord/07_subject2_snr_correlation.png)

### Key insights

FBCCA + SVM = perfect cross-session accuracy for both subjects.

![](/results/figures/discord/08_cca_vs_fbcca_bar_chart.png)

Subject 2 LDA dips to 80% in one direction.

    what's LDA?

FBCCA `cv_gap = 0` which validates the approach, literally zero overfitting signal in HPO.

     what's HPO?

PSD weakness is now contextualized: **Subject 2's sudden increasy in acc when using FBCCA**

# Neuroscientific Lense

## Neural Resonance Harmonics...

When a person fixates on a flickering stimulus, the visual cortex (specifically the occipital electrodes like Oz, O1, O2, and POz) resonates at that specific frequency.

> **_The brain doesn't just react to the fundamental frequency, but also prodecues harmonics_**

     Harmonics:  integer multiples of that frequency.

## SNR (Signal-to-Noise Ratio) Optimization.
