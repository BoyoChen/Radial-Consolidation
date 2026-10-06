# Radial Consolidation: figure and analysis reproduction

Reproduction materials for **Radial Consolidation: Constraining Post-Formation
Kernel Drift in Spiking Neural Networks**.

Radial consolidation constrains existing STDP updates after an initial formation
period, as an alternative to fully freezing the hidden cortex. It is not a new
STDP rule applied throughout training. The primary intervention begins before
training epoch 2, after one completed formation epoch.

**Scope:** this package regenerates all four figures, three tables and Algorithm 1
from bundled research data. It does not retrain the original networks. No GPU,
dataset download, research server or LaTeX installation is needed for analysis.

## Installation

Python 3.10 is the locally tested version; Python 3.11 is included in the planned
CI matrix. From the repository root:

```bash
python -m pip install -r requirements.txt
```

For an isolated Conda environment:

```bash
conda env create -f environment.yml
conda activate radial-consolidation
```

Alternatively use `python -m venv .venv`, activate it, and install the same
requirements. The runtime and direct/transitive dependencies are pinned. All
subsequent analysis works offline.

## Reproduce

```bash
# One command from the repository root.
python reproduce.py
```

The code-directory entry point produces the same outputs:

```bash
cd code
python main.py --analyze
```

Both commands write to `result/`, independently of the current working directory.
Open [`result/figures_and_tables.pdf`](result/figures_and_tables.pdf) to review all
eight manuscript items together. Use `--output another-directory` to preserve an
existing output version. Named artifacts in the selected directory are replaced.
There is no training command or experiment-dispatch side effect.

## Layout

```text
code/main.py              analysis entry point
code/reproduction/       plotting, statistics and projection reference code
data/                    bundled numeric inputs and checksum manifest
result/                  paper-ready graphics, tables and computed statistics
paper/                   clearly labeled working manuscript draft
tests/                   source, method and artifact checks
docs/                    optional original method LaTeX fragments
reproduce.py             convenient root entry point
```

## Figures and tables

The committed samples below can be inspected without running Python. Rebuilding
computes the quantities again; it does not simply copy these PDFs.

| Item | Content | Inputs | Sample |
|---|---|---|---|
| Figure 1 | Parallel/orthogonal update decomposition | Conceptual diagram | [PDF](result/figure_01_method.pdf) |
| Figure 2 | Kernel snapshots and full-dimensional angular drift | 104 kernel arrays, four seeds | [PDF](result/figure_02_drift.pdf) |
| Figure 3 | Kernel boundary x homeostasis; Valid and Readout | 16 paired-seed evaluation rows | [PDF](result/figure_03_kernel_homeostasis.pdf) |
| Figure 4 | Unrestricted / Radial-only / Hard Freeze accuracy and gap | 51 paired-seed evaluation rows | [PDF](result/figure_04_accuracy_gap.pdf) |
| Table I | Model and training protocol | Explicit manuscript constants | [PDF](result/table_01_protocol.pdf), [CSV](result/table_01_protocol.csv) |
| Table II | Checkpoint statistics and 20-epoch direction changes | 20 saved kernels | [PDF](result/table_02_geometry.pdf), [snapshots CSV](result/table_02_snapshots.csv), [intervals CSV](result/table_02_intervals.csv) |
| Table III | Accuracy and paired contrasts with 95% CI | Per-seed evaluation data | [PDF](result/table_03_results.pdf), [accuracy CSV](result/table_03_accuracy.csv), [paired CSV](result/table_03_paired.csv) |
| Algorithm 1 | Reference-constrained hidden update | Method definition | [PDF](result/algorithm_01_update.pdf) |

Figure 2 is also exported as independent snapshot and angle panels for LaTeX
composition. Individual graphics have PNG/SVG exports where applicable; tables
have CSV and LaTeX exports. Full statistical calculations and output hashes are
recorded in JSON. Selected PDFs, PNG previews, tables and numeric audits are
versioned under `result/`; disposable image variants are ignored.

### Representative paired result

Radial-only minus hard freeze at epoch 20; paired Student-t 95% CIs. Accuracy
differences are percentage points (pp); timing gap is normalized earliness.

| Dataset / batch | Seeds | Validation difference (pp) | Evidence-gap difference |
|---|---:|---:|---:|
| MNIST main | 5 | +0.057 [-0.043, 0.157] | -0.02763 [-0.03151, -0.02375] |
| MNIST replication | 5 | +0.060 [-0.162, 0.282] | -0.02695 [-0.02953, -0.02437] |
| N-MNIST | 7 | -0.007 [-0.226, 0.211] | +0.00961 [0.00771, 0.01151] |

An interval spanning zero is not proof of equivalence. The timing residual is
dataset dependent, and paired seeds do not remove mixed-GPU confounding.

## Verification

```bash
python tests/validate.py
python reproduce.py
python tests/artifact_smoke.py
```

For just the bundled input integrity check: `python reproduce.py --verify-only`.
Every build checks input hashes, paired seeds, array shapes, PCA/sample identity,
full-dimensional angles, geometry and paired contrasts against frozen numeric
references. References are regression checks, not replacements for computation.
Success is recorded in `result/reproduction_manifest.json`.

The reference implementation of the projection is
[`code/reproduction/method.py`](code/reproduction/method.py). It is a minimal
NumPy implementation using receiver-by-coefficient matrices, not the original
simulator. It does not implement spikes, STDP traces, homeostasis, masks or floors.

The included GitHub Actions workflow targets Windows/Linux and Python 3.10/3.11.
Remote CI must still run after publication; local testing is not evidence that
the remote matrix has passed.

Local verification on 2026-10-07 used a fresh Windows/Python 3.10 environment:
nine unit tests, all artifact checks, both entry points, and an independent Git
clone rebuilt from an external working directory with Python network connections
blocked. The complete build took approximately 9-10 seconds on that machine,
excluding dependency installation. Committed result checksums also passed on
the clone before rebuilding. Runtime varies with hardware.

## Data and measurement contracts

See [DATA.md](DATA.md) for the schemas and full interpretation limits. The
approximately 52 MB kernel archive contains learned coefficients, not MNIST or
N-MNIST samples. Evaluation records are scalar outputs of completed runs; this
package does not regenerate those scalars from raw images/events.

- The five illustrated neurons and seven checkpoints are fixed. PCA is refitted
  once across all saved checkpoints of the example run, then its arbitrary SVD
  signs are aligned with the archived axes. No per-epoch reorientation is used.
- Angles are recomputed in 600 dimensions, not measured from the PCA drawing.
  The terminal two-epoch interval is excluded from the four-epoch change curve.
- Table II does not display parenthesized relative changes; the full calculations
  are retained in `result/geometry_statistics.json`.
- Figure 3 uses native validation accuracy and linear Readout test accuracy:
  these are different splits and decoders. The schedule comparison is not a
  pure rotation intervention.
- Unrestricted N-MNIST has 100% no-decision. Its zero gap means silence, not
  healthy decisions; the broken axis explicitly retains this result.
- Hard freeze disables hidden learning and homeostasis. Radial-only retains
  homeostasis; N-MNIST amplifier is fixed at 10, unlike the adaptive MNIST setting.
- Epoch-50 trajectory continuation does not restore RNG/data-loader state and
  may change GPU. This is not a bitwise uninterrupted 100-epoch run.

## Paper and scope limitations

[`paper/radial-consolidation-draft.pdf`](paper/radial-consolidation-draft.pdf) is
a **pre-submission working draft**, not an accepted or published article. The
current snapshot has nine content pages plus references and still needs page-limit
revision. No publication DOI or acceptance claim is made.

The portable figures use DejaVu fonts and no-TeX rendering for the diagram and
tables; Figure 4 uses DejaVu Serif instead of system Times New Roman. Numerical
values, ordering, sampling, colors and axis limits are preserved, but pixel or
binary identity to the typeset manuscript is not promised. Original method
LaTeX fragments remain in `docs/method_latex.tex`.

This release demonstrates analysis/figure reproducibility, not from-scratch
training reproduction, causal harm from rotation alone, universal effectiveness
or useful lifelong adaptation. Normalization may cancel purely radial updates.

## Citation and license

Repository metadata is in [CITATION.cff](CITATION.cff). It describes this software
release without borrowing another paper's DOI or claiming conference acceptance.

The code is licensed under the [MIT License](LICENSE). The MIT license applies
to the software and its software documentation, not the research data in `data/`,
the manuscript in `paper/`, or data-derived graphical/tabular results in `result/`.
Those materials do not yet have a separate reuse license. Third-party dependencies
retain their own licenses. See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for the
remaining publication decisions.
