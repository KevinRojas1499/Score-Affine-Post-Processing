# Affine post-processing of Monte Carlo score estimators

Code for the paper *Improving Score-Based Sampling via Affine Post-Processing*.
A diffusion-based sampler needs the score of the noised target at every reverse
step; when only the target density is available, that score is estimated by
Monte Carlo. This package implements several base estimators through IS, RDMC and ZODMC as well as our PostProcessed version.
## Installation

    uv sync

Requires PyTorch, POT and matplotlib; a GPU is used when available.

## Usage

```python
from affine_score import Banana, ImportanceSampling, PostProcessed, VPSchedule, sample

target = Banana(dim=10, device='cuda')
base = ImportanceSampling(target, VPSchedule(), num_samples=500, num_batches=10)
x = sample(PostProcessed(base), num_samples=500, disc_steps=20, device='cuda')
```

A target is any object with `dim`, `dtype`, `log_prob(x)` (IS, ZOD-MC),
`grad_log_prob(x)` (RDMC and the exact endpoint) and `log_prob_max()` (ZOD-MC's
acceptance ceiling). `affine_score.TARGETS` has the paper's targets, each
with an exact sampler: the dimension sweep's `banana`, `x_gmm` and `ell_shell`, and
`logconcave`, `random_gmm`, `two_mode_gmm`, `zodmc_gmm` and `vertex_gmm`.
Any of them runs through the sweep script with
`--targets <name> --dims <d>`; the settings there are tuned for the first three only.

### Options

| Option | Values | Meaning |
|---|---|---|
| schedule | `VPSchedule` (default), `VPLinearSchedule` | a(t) = e^{−t} or a(t) = 1 − t. The same process in two time variables; the VP-linear time grid visits the same signal levels as the VP one. |
| `PostProcessed(sigma=…)` | `'global'` (default), `'point'` | Σ estimated once for the batch, or separately for each particle. |
| `PostProcessed(weights=…)` | `'estimate'` (default), `'coordinate'` | One weight vector h per score estimate, or one per coordinate. |
| `PostProcessed(groups=…)` | integer, default 2 | Replicate groups Σ is estimated from; the cost is `groups × num_levels` base calls per query. |

Σ has `(groups − 1) × (particles pooled) × (coordinates pooled)` degrees of freedom
and the solve needs at least `num_levels` of them, so `sigma='point'` needs
`groups ≥ 2` only for d ≥ 6, and `sigma='point', weights='coordinate'` needs
`groups ≥ 7`. `PostProcessed` raises otherwise.

## Reproducing the dimension sweep

    uv run python3 dimension_sweep.py          # writes results/dimension_sweep.pt
    uv run python3 plot_dimension_sweep.py     # prints the tables, writes figures/dimension_sweep.pdf

Both scripts take `--schedule`, `--sigma`, `--weights` and `--groups`; the sweep
also takes `--targets`, `--dims`, `--seeds`, `--budget` and `--methods`. Results
are cached per cell, so an interrupted sweep resumes and a changed setting never
reads a stale row. The full sweep takes about 50 minutes on a laptop GPU.

**Expected results** (W2 / floor, mean over seeds 0–2):

| Banana, d          |    2 |    5 |   10 |   20 |   30 |   50 |
|--------------------|-----:|-----:|-----:|-----:|-----:|-----:|
| IS                 | 1.40 | 29.8 | 26.8 | 23.8 | 30.4 | 38.6 |
| IS + ours          | 1.64 | 1.13 | 1.08 | 1.13 | 1.19 | 1.32 |
| RDMC               |  315 |  142 | 77.6 | 51.7 | 45.7 | 39.1 |
| RDMC + ours        | 9.30 | 4.70 | 3.38 | 2.88 | 2.73 | 2.60 |
| ZOD-MC             | 1.13 | 1.15 | 1.13 | 1.02 | 0.96 | 0.89 |

| X-GMM, d           |    2 |    4 |    6 |    8 |   12 |   16 |
|--------------------|-----:|-----:|-----:|-----:|-----:|-----:|
| IS                 | 1.25 | 1.65 | 8.75 | 20.0 | 33.7 | 37.1 |
| IS + ours          | 1.02 | 1.05 | 0.98 | 1.05 | 1.21 | 1.37 |
| RDMC               | 8.04 | 3.83 | 2.77 | 2.30 | 1.94 | 1.81 |
| RDMC + ours        | 3.81 | 1.81 | 1.44 | 1.33 | 1.27 | 1.21 |
| ZOD-MC             | 1.22 | 1.13 | 1.20 | 1.27 | 1.26 | 1.22 |

| Ellipsoidal shell, d |    2 |    5 |   10 |   20 |   30 |   50 |
|----------------------|-----:|-----:|-----:|-----:|-----:|-----:|
| IS                   | 4.30 | 1.37 | 3.81 | 11.7 | 17.1 | 23.1 |
| IS + ours            | 5.08 | 1.70 | 1.15 | 1.03 | 1.00 | 1.03 |
| RDMC                 | 1.30 | 1.43 | 1.17 | 1.09 | 1.08 | 1.08 |
| RDMC + ours          | 3.39 | 1.20 | 1.03 | 1.02 | 1.04 | 1.05 |
| ZOD-MC               | 4.29 | 1.55 | 1.14 | 1.05 | 1.00 | 0.95 |
