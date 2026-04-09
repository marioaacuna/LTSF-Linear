# Analysis_MA — Pipeline Overview

End-to-end pipeline for extracting pose features, normalizing them, running
the DLinear forecasting model, and generating figures.

---

## Step 1 — Extract features (MATLAB)

**Script:** `extract_cohort_data.m`

Loads raw pose data from the SFC and HNG cohorts and saves per-condition
feature matrices (66 features × N frames) as `.mat` files.

**Prerequisites:**
- `Figures_BEACON_paper` repo must be on the MATLAB path (or run `fbi()`).
- `path_config.json` must point to the CAPTURE project root.

```matlab
run('Analysis_MA/extract_cohort_data.m')
```

**Output** (`Analysis_MA/data/`):
```
SFC_S_features.mat   SFC_F_features.mat
HNG_H_features.mat   HNG_N_features.mat
```

---

## Step 2 — Normalize and PCA-reduce (MATLAB)

**Script:** `convert_and_normalize.m`

Two-part pipeline. Requires Step 1 outputs.

**Part 1** — Per-condition min-max normalization to `[-1, 1]` on all 66
features, with a synthetic date column at 12.5 Hz.

**Part 2** — Pooled PCA per cohort. Within each cohort (SFC, HNG), all
conditions are concatenated, jointly normalized, and PCA is fit on the
pooled data. The number of retained components `k` is chosen automatically
to explain at least 80% of total variance. Each condition is then projected
onto those **shared** PCA axes and saved independently. This ensures
cross-condition comparisons use the same feature space.

PCA metadata (coefficients, explained variance, normalization params) is
saved to `<cohort>_pca_info.mat` for reference.

```matlab
run('Analysis_MA/convert_and_normalize.m')
```

**Output** (`Analysis_MA/data/`):
```
SFC_S_normalized.csv        SFC_F_normalized.csv
HNG_H_normalized.csv        HNG_N_normalized.csv

SFC_S_normalized_pca.csv    SFC_F_normalized_pca.csv
HNG_H_normalized_pca.csv    HNG_N_normalized_pca.csv

SFC_pca_info.mat            HNG_pca_info.mat
```

To change the variance threshold, edit `pca_variance_threshold` at the top
of the script (default: 80%).

---

## Step 3 — Train & evaluate DLinear model (Python)

**Script:** `run_all_conditions.py`

Trains a DLinear model on each condition and saves predictions. Run from the
**repository root**.

```bash
# All 66 original features
python Analysis_MA/run_all_conditions.py

# PCA-reduced features
python Analysis_MA/run_all_conditions.py --pca
```

Sanity-check plots are saved to `Analysis_MA/sanity_<tag>.png`.
Model predictions are saved under `results/`.

---

## Step 4 — Plot figures (MATLAB)

**Script:** `plot_fig2_cohorts.m`

Loads `pred.npy` / `true.npy` from `results/` and generates three
publication-quality figures per cohort:

1. Error metrics by prediction horizon (MAE, RMSE, MAPE, Correlation)
2. Scatter: predictions vs ground truth at selected horizons
3. Zoomed prediction timeline with confidence bands

Requires Step 3 to have been run first.

```matlab
run('Analysis_MA/plot_fig2_cohorts.m')
```

**Output** (`Analysis_MA/figures/`):
```
Fig02_SFC_error_metrics.pdf    Fig02_HNG_error_metrics.pdf
Fig02_SFC_scatter.pdf          Fig02_HNG_scatter.pdf
Fig02_SFC_timeline.pdf         Fig02_HNG_timeline.pdf
```
