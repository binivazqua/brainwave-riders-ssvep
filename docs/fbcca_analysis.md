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

_**\*About LDA**_

> Classic ML pattern recognition algorithm widely used in BCIs.
> Go-to for MI tasks and p300

Characteristics:

- Supervised (trained on labeled data).
- Stationary signal assumption.
  @Notebook LM says:
  _Experts like Johannes Kunvald recommend applying a log transform to your power features before feeding them into an LDA; it stabilizes the distribution and makes the classifier much more effective_

FBCCA `cv_gap = 0` which validates the approach, literally zero overfitting signal in HPO.

_***About HPO***_

> Fine tuning on steroids.

**In BCI specifically:**
Involves systematically testing different combinations of parameters, like number of harmonics in CCA, sub-band boundaries on FBCCA, etc...

`cv_gap`
_The difference between training accuracy and cross-validation accuracy_

- No overfitting sign.

PSD weakness is now contextualized: **Subject 2's sudden increasy in acc when using FBCCA**

# Neuroscientific Lense

## Neural Resonance Harmonics...

When a person fixates on a flickering stimulus, the visual cortex (specifically the occipital electrodes like Oz, O1, O2, and POz) resonates at that specific frequency.

> **_The brain doesn't just react to the fundamental frequency, but also prodecues harmonics_**

     Harmonics:  integer multiples of that frequency.

#### _Why results get better..._

> Standard PSD often ignores these harmonics or loses them in bckg noise. FBCCA on the other hand decomposes into sub-bands to capture **energy** from these harmonics, providing a **richer signature** on user's stimulation activity.

## SNR (Signal-to-Noise Ratio) Optimization.

About SNR:
_Ratio of meaningful brain response to noise_

Subject 2's low SNR might cause PSD features to be weak and messy!

#### _Why results get better..._

> FBCCA uses a weighting scheme that prioritizes sub-bands with higher SNR contributions!! On the other hand, PSD is easily "drowned out" if the power at a single frequency is low

      Higher SNR -> Better Signal Quality

## Filter Bank Frequency Decomposition

> Doen't look at the broadband eeg, but narrower frequency ranges using the filter bank.

#### _Why results get better..._

> By focusing on specific sub-bands, the model can isolate the stimulus-locked resonance from the non-stationary background noise

      Prevents low freq alpha waves (strong + noisy) from the detection of higher SSVEP harmonics.

## Templates vs Peaks

Context:

- PSD relies on peak detection: looks for the highest spikes in power at a single point.
- FBCCA and CCA use _template matching_, that is a correlation between the entire eeg signal against sine + cosine ref signals.

#### _Why results get better..._

> CCA is less dependent on SNR because it uses information across the entire EEG space, FBCCA takes this a step further by matching templates across multiple sub-bands simultaneously.

Example @Notebook LM.

> _If the fundamental frequency (10 Hz) is noisy but the second harmonic (20 Hz) is clear, FBCCA identifies that the 20 Hz band has a higher SNR and gives it more "relevance" in the final decision_

### Data Analysis Notes

- High SNR Subjects: Simple methods (PSD) and complex methods (FBCCA) both work well.
- Low SNR Subjects (BCI-Illiterate): Simple methods fail, but SNR-robust methods like FBCCA are necessary to achieve high accuracy
- FBCCA is significantly more robust than standard CCA for shorter epochs.
  By leveraging harmonic information in the sub-bands, it can reach high accuracy even with less data, **which likely contributed to the zero cv_gap (no overfitting)** WE OBTAINED!
