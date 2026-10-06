# Data, provenance and limits

The software is MIT-licensed. The research data described here, the manuscript
and derived research results do not yet have a separate reuse license; the code
license does not automatically apply to them.

## Included source data

- `data/kernels.npz`: 104 arrays named `s<seed>_e<epoch>`. Four runs
  (34100-34103), 26 checkpoints (2, 6, ..., 98, 100), shape `(300, 600)`, float32.
  Each row is one receiver's learned incoming kernel, flattened as six spatial
  channels of 10 by 10. The archived binary arrays are unchanged from the
  figure-analysis input. These are not complete model checkpoints.
- `data/evaluation.json`: 67 evaluation records at completed epoch 20. MNIST
  main/replication have five matched seeds per condition; N-MNIST has seven;
  kernel/homeostasis has four seeds per cell. Seeds and GPU models are retained;
  internal run names, service IDs, hostnames, filesystem paths and source-control
  metadata are excluded. Accuracy and no-decision are fractions before plotting.
- `data/trajectory_reference.json`: published PCA orientation, fixed random
  sample, and per-seed angles for regression checking. The pipeline recomputes
  PCA and angles from the kernel arrays, then checks these references.
- `data/geometry_reference.json`: per-seed geometry and paired-interval
  statistics, including undisplayed relative changes and confidence intervals.
- `data/paired_reference.json`: independent frozen paired-contrast summaries
  exported from the manuscript's final evaluation analysis.
- `data/captions.json`: the manuscript's scientific captions, with simple
  notation adapted for a no-TeX overview.
- `data/manifest.json`: SHA-256 and byte counts for the input files.

Evaluation records are archived scalar outputs of completed training runs.
This package recomputes group means and paired Student-t 95% CIs, but does not
re-evaluate the original models or regenerate those evaluation scalars from
images/events. Dataset splitting, encoding and original training determinism
are therefore outside the scope of this analysis-only reproduction release.

## Measurement contracts

- Figure 2 uses example seed 34101; the first five entries of an eight-element
  ordered draw without replacement from 300 neurons, RNG seed 2027, are
  `[2, 24, 185, 43, 142]`. This preserves the approved nested sampling design.
  No angle, quadrant or performance filter is applied.
- PCA is centered and fitted on all saved epochs of the example run. The origin
  is the PCA mean, not the zero kernel. The projection is descriptive and uses
  future checkpoints; it is not a training-time algorithm.
- Full-dimensional reference angles average 300 neurons within each seed, then
  four seeds. The incremental curve compares four-epoch intervals and excludes
  the final two-epoch interval. The epoch-50 continuation does not restore
  original RNG/data-loader state and may change GPU.
- Participation ratio is `(sum(s**2))**2 / sum(s**4)`, not entropy effective rank.
  DC means the spatial mean within each input channel. Signed mass is signed
  weight sum divided by absolute weight sum, not a positive-entry fraction.
- Orthogonal fraction averages `norm(delta_perp)/norm(delta)` per receiver,
  relative to the interval's starting kernel. It is not an energy fraction or
  the fraction of neurons that changed. All Table II intervals span 20 epochs.
- The intervention `@2` starts before training epoch 2, after one completed
  formation epoch; a saved epoch-2 checkpoint follows two completed epochs.
- Figure 3 contrasts native **validation** and hidden linear Readout **test**
  accuracy. These are different splits and decoders. The matched schedule
  comparison is not a pure rotation intervention or homeostatic-state transplant.
- Figure 4 displays Unrestricted, Radial-only, Hard Freeze. The accuracy axis
  spans four percentage points. Gap panels use equal spans with different
  offsets; broken N-MNIST axes retain the zero/no-decision observations.
- Hard freeze disables hidden learning and homeostasis; radial-only retains
  homeostasis. MNIST amplifier adapts, whereas N-MNIST amplifier is fixed at 10.
- Different GPU models occur in matched-seed comparisons. The GPU field is
  retained for inspection; confidence intervals do not remove this confound.

## Boundaries

This release does not establish causal harm from rotation alone, convergence
or nonconvergence, formal equivalence to hard freeze, or useful lifelong kernel
adaptation. Fixed-target normalization may cancel purely radial updates.
No missing fixed-input drive replay, winner-count diagnostic or additional
experiment is represented as completed by this reproduction package.
