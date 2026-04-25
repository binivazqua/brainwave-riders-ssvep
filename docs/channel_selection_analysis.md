# 3ch vs 8 ch Behavior

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>feat</th>
      <th colspan="2" halign="left">CCA</th>
      <th colspan="2" halign="left">FBCCA</th>
    </tr>
    <tr>
      <th></th>
      <th>channels</th>
      <th>3-ch (O1/Oz/O2)</th>
      <th>8-ch (all)</th>
      <th>3-ch (O1/Oz/O2)</th>
      <th>8-ch (all)</th>
    </tr>
    <tr>
      <th>subject</th>
      <th>alg</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Subject 1</th>
      <th>LDA</th>
      <td>1.000</td>
      <td>0.975</td>
      <td>1.000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>SVM</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Subject 2</th>
      <th>LDA</th>
      <td>0.600</td>
      <td>0.975</td>
      <td>0.700</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>SVM</th>
      <td>0.675</td>
      <td>1.000</td>
      <td>0.675</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

## In a Nutshell

### Subject 1:

     3 channels are more than enough + consistent!

Dropping to O1/Oz/O2 maintained or slightly improved accuracy. The extra 5 channels were possibly adding noise.

### Subject 2:

    Other way around: 5 additional channels actually helping.

Dropping to O1/Oz/O2 "hurted accuracy" +- 30%... . It means Subject 2's SSVEP response is spread across more channels, not concentrated purely in O1/Oz/O2. Possibly PO7/PO3/POz are picking up signal too.

_That means a different neurophysiological variable? Or even anatomical_

#### Supporting evidence:

1. **Preliminar** topo map on subject 2:

![](/results/figures/discord/topomap_s2_initial.png)

9 Hz — signal is strongest on the far left and right edges (PO7, PO8), almost nothing in the center bottom (Oz)

10 Hz — again spread laterally, stronger on the sides than the center

12 Hz — interesting: the center (Oz) is actually teal/dark = weak, while the sides are warmer

15 Hz — spread across the bottom but not centered on Oz

2. **Preliminar** PSD Across 3 channels on subject 2:

Look at the dashed lines (stimulus frequencies):

9 Hz (pink) — tiny bump, barely visible
10 Hz (orange) — slight peak on the orange line only
12 Hz (teal) — almost nothing
15 Hz (gray) — nothing

The 4 colored lines are nearly on top of each other. A good SSVEP signal would show each colored line peaking clearly at its own dashed vertical line (Task: look for reference papers to compare).

![](/results/figures/discord/sub2_psd_initial.png)

## Why this matters...

Why PSD fails on S2:
No clean spectral peaks at the stimulus frequencies on O1/Oz/O2. PSD literally has nothing to grab onto.

Why CCA/FBCCA still worked
CCA doesn't need sharp peaks, because it finds subtle correlation patterns that are invisible to the eye here but mathematically present.

Why 3 channels hurt Subject 2
O1/Oz/O2 simply don't carry the signal for this subject.
The response must be on the other channels (PO7, PO3, POz...).

Why session quality **might not** be the issue
The signal isn't weak because of a bad recording (_directionality_) it's structurally flat across all 4 classes on these channels!!

## Final Conclusions (for now)

    Channel selection isn't one-size-fits-all.

BecauseL

- Subject 1's response is focused in pure occipital channels,
- Subject 2's is broader. This is another dimension of subject variability, and it explains why using all 8 channels matters for robustness.
