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

## Last Year's winners

To win the BR41N.IO Data Analysis track for SSVEP, you should look at the 2025 first-place winner, NeuroPulse, who set a high standard by building an exhaustive preprocessing pipeline, comparing multiple machine learning models, and delivering a functional web application
.
The following rubric provides the essential components for a competitive submission based on past winners and current state-of-the-art standards.

1. Minimals (The "Must-Haves")
   To be considered for a prize, your submission must provide a technically sound and reproducible foundation
   .
   Clean Preprocessing Pipeline: Implement data loading, notch filtering (50 Hz/100 Hz to remove line noise), and bandpass filtering (typically 1–40 Hz)
   .
   Epoch Segmentation: Accurately extract time windows around stimulus onset and implement basic baseline correction
   .
   Standard Feature Extraction: Implement Canonical Correlation Analysis (CCA) as a baseline, as it is unsupervised and robust for standard setups
   .
   Rigorous Evaluation: Strictly separate your training and validation/testing matrices; the jury will look for "no mixing" of these datasets
   .
   Essential Visualization: Include raw EEG traces from occipital channels (Oz, O1, O2, POz) with clearly marked stimulus points
   .
2. Goals (Strong Competitive Level)
   Top-tier teams go beyond basic classification to optimize performance and algorithmic comparison
   .
   Advanced Classifiers: Implement Filter Bank CCA (FBCCA), which is often more robust for shorter epochs, or Ensemble Task-Related Component Analysis (eTRCA)
   .
   Algorithmic Comparison: Compare results across different models, such as Random Forest, SVM, and Logistic Regression, as seen in the 2025 winner's suite
   .
   High Accuracy: Aim for a target accuracy of at least 80-90% for 8-class problems, matching the benchmarks set by NeuroPulse
   .
   Parameter Tuning: Show evidence of optimizing epoch lengths, the number of harmonics in reference signals (typically 2–5), and sub-band boundaries for FBCCA
   .
   Metric Depth: Report not just accuracy, but F1 scores and Information Transfer Rate (ITR)
   .
3. Nice to Haves (The "Winner's Edge")
   These elements often distinguish the first-place winner from the runners-up by demonstrating practical utility and innovation
   .
   Interactive Visualization: Build an interactive frequency spectrum explorer or Power Spectral Density (PSD) dashboard showing clear peaks at target frequencies and their harmonics
   .
   Deployable Interface: Create a web application or a demo-ready notebook that allows judges to explore feature extraction, model comparisons, and video annotations in real-time
   .
   Artifact Suppression: Move beyond simple bad-channel removal to automated artifact handling (like ICA for eyeblinks or Oscar for muscle noise) to ensure signal fidelity
   .
   Interpretability: Use Saliency Maps or spectral characterization (like the spectral mapping of the fusiform gyrus used by NeuroPulse) to explain why the model is making specific decisions
   .
4. Strategic Advice for Winning
   Signal Quality is Design: Remember that a complex model cannot fix a bad signal; your preprocessing and SNR optimization are your most important decisions
   .
   Think Outside the Box: Winners are often teams that do not just replicate existing work but find a personal touch or a unique application for the data
   .
   Multidisciplinary Presentation: Success is an "ecosystem story"
   . Ensure your presentation communicates the neuroscience behind the data, not just the engineering metrics
   .
   Presentation Quality: Compelling visual presentation of results was identified as a key gap; use high-quality vector graphics (e.g., Inkscape) and clear confusion matrices to make your results "printable" quality

## Notebook Theory Insights on Sliding Window Robustness + ITR

### Sliding Window

1. Statistical Significance and the Binomial Test
   The fact that your worst result (57.5%) achieved a p-value of 9.35×10
   −4
   against a 25% chance baseline is theoretically sound because of the Binomial Test. [Conversation History]
   Significance over Chance: In BCI research, a binomial test is used to prove that a classifier's accuracy is not just a "lucky run" but is physiologically meaningful. [Conversation History]
   Validation of Representation: Experts emphasize that "better classifiers don't fix the wrong question."
   By achieving high statistical significance (p<0.001), you have validated that your preprocessing and feature extraction are capturing a true stimulus-locked neural resonance rather than random artifacts. [87, Conversation History]
2. Why FBCCA Consistently Outperforms CCA
   Theory suggests that FBCCA is state-of-the-art for SSVEP because it extracts a "richer" signature of user intent compared to standard CCA. [1745, Conversation History]
   Neural Harmonics: When the visual cortex resonates at a flickering frequency (e.g., 10 Hz), it also produces harmonics at integer multiples (20 Hz, 30 Hz). [1743, Conversation History] Standard CCA focuses primarily on the fundamental frequency, but FBCCA uses a Filter Bank to decompose the signal into multiple sub-bands to capture these harmonics. [1745, Conversation History]
   SNR-Based Weighting: FBCCA employs a weighting scheme that prioritizes sub-bands with higher Signal-to-Noise Ratio (SNR).
   This is why you see the widest gap at short windows; FBCCA "rescues" the signal in Subject 2 by weighting informative harmonics even when the fundamental 1-second signal is buried in noise. [Conversation History]
3. Temporal Resolution vs. Frequency Resolution
   The theory of SSVEP identifies a fundamental trade-off: longer epoch lengths provide better frequency resolution but result in a slower BCI. [1746, Conversation History]
   Temporal Integration: For Subject 2 (who you identified as a "low SNR responder"), moving from 1s to 6s allows for temporal integration. [Conversation History] This effectively gives the algorithm more cycles to "average out" non-stationary background EEG noise. [Conversation History]
   Convergence at 6.85s: Theoretically, as the window grows, the frequency resolution becomes high enough that even unweighted methods like standard CCA can eventually resolve the target frequency, explaining why both of your methods converged to 100%. [Conversation History]
4. Generalization and the "Zero Gap"
   Your report of statistical significance and the "perfect" cross-session accuracy with a "cv_gap" of zero is a critical theoretical marker. [1739, Conversation History]
   Avoiding Shortcut Learning: Neural networks can sometimes fall into "shortcut learning," where they latch onto non-brain artifacts (like muscle tension) to reach a decision.
   Theoretical Validation: Achieving a zero gap between training and validation sets during Hyperparameter Optimization (HPO) provides statistical proof that your model has truly learned the neural resonance of the stimulus rather than memorizing session-specific noise. [1739, Conversation History]
   Strategic Documentation Note for Your Docs:
   Your results satisfy the "Rigorous Evaluation" and "Metric Depth" requirements found in winning rubrics for the BR41N.IO Data Analysis track.
   Demonstrating that every data point is statistically significant (p<0.001) while documenting the practical advantage of FBCCA at short windows is what distinguishes a top-tier submission from a baseline analysis. [1733, 1735, Conversation History]

### ITR

What is ITR Used For?
Information Transfer Rate (ITR) is the standard metric used in BCI research to quantify the communication throughput of a system.
While accuracy tells you how often the BCI is right, ITR tells you how much information (in bits) is successfully transmitted per unit of time (usually per minute).
In your specific hackathon context, ITR is the "Metric Depth" that judges look for to distinguish a basic analysis from a competitive, system-level evaluation.
It is used to find the optimal balance between speed and reliability, which is the most critical decision when deploying a BCI for a real patient.

---

Analysis of Your Results: The Accuracy-Speed Trade-off
Your data provides a textbook illustration of how Signal-to-Noise Ratio (SNR) dictates the optimal BCI configuration.

1. Subject 1: The "Throughput-Limited" Scenario
   For Subject 1, who has high SNR, maximizing accuracy and maximizing ITR are conflicting targets. [Conversation History]
   The Findings: While Subject 1 hits 100% accuracy at 6.85 seconds, their ITR actually drops significantly at that length (12.0 bits/min). Their peak performance is at a 1-second window (21.7 bits/min). [Conversation History]
   Theory Insight: This subject’s brain resonates so strongly with the stimulus that the algorithm reaches "statistical certainty" almost immediately. [Conversation History] Waiting for 6.85 seconds is "wasted time" that kills the communication speed. In a real-world speller, Subject 1 would much rather have a 92.5% accurate system that types every second than a 100% accurate system that makes them wait 7 seconds per letter.
2. Subject 2: The "Accuracy-Limited" Scenario
   Subject 2, the "hard subject" with low SNR, shows a monotonic increase in ITR. [Conversation History]
   The Findings: Their ITR peaks at the longest window (6.85s, 12.0 bits/min). [Conversation History]
   Theory Insight: Because their signal is so weak, the "cost" of a wrong decision (which conveys 0 bits of information) is higher than the "cost" of a slow trial. [Conversation History] For this subject, the extra time is a necessary investment to "average out" the non-stationary background noise and allow the stimulus-locked resonance to emerge. [Conversation History]

---

Benchmarking and Success Criteria
Your results place your team in a very strong position relative to the benchmarks found in the source materials and typical clinical standards:
Clinical Standard: Your Subject 1 results (~21 bits/min) fall squarely within the 15–30 bits/min range cited for typical clinical SSVEP systems. [143, Conversation History]
State of the Art Gap: While laboratory settings using high-density EEG and massive training sets can reach 229–315 bits/min, those systems often require high SNR and extensive calibration.
FBCCA Advantage: Your data confirms that FBCCA provides more "information per second." The 75% higher throughput for Subject 1 at 1s vs 6.85s is a "killer stat" for your documentation because it proves the algorithm's practical utility for real-time interaction.
Strategic Documentation Tip:
In your final report, highlight the Subject 1 vs. Subject 2 ITR curves. This proves to the judges that you understand the Neuroscience of Utility: that the "best" BCI isn't always the one with 100% accuracy, but the one that maximizes the user's ability to communicate efficiently based on their individual biological profile.
