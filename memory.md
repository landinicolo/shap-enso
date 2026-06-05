# SHAP-ENSO Project — Implementation Log

> **Purpose:** Running record of decisions made, steps completed, bugs fixed, and the full execution plan.  
> **Last updated:** 2026-06-05  
> **Repo:** `aneeshcs/shap-enso` (NCAR Casper: `/glade/work/acsubram/GitRepos/shap-enso`)

---

## Science Goal

Predict the Niño 3.4 SST anomaly index at **3-, 6-, and 12-month lead times** using three ML architectures (XGBoost, LSTM, CNN), then apply SHAP explainability to answer:

1. Which ocean-atmosphere variables drive ENSO predictability at each lead?
2. Does SHAP recover the spring predictability barrier (SPB)?
3. Are El Niño and La Niña forced by different mechanisms (asymmetry)?
4. Do spatial SHAP maps agree with known physical teleconnection patterns?

---

## Repository Structure

```
shap-enso/
├── configs/
│   ├── default.yaml                   # Main config (used by all PBS jobs)
│   ├── xgb_regression_all_leads.yaml
│   ├── xgb_classification_all_leads.yaml
│   ├── lstm_regression_all_leads.yaml
│   └── cnn_regression_all_leads.yaml
├── data/
│   ├── download_era5.py               # ERA5 (SST, τx, OLR, SLP) via CDS API
│   ├── download_d20.py                # D20 proxy (ORAS5 OHC300) via CDS API
│   ├── download_noaa_indices.py       # NOAA Niño 3.4 index
│   └── raw/
│       ├── era5/                      # Downloaded ERA5 NetCDF files
│       └── d20/                       # Downloaded D20/OHC300 NetCDF file
├── src/
│   ├── utils/
│   │   ├── config.py                  # YAML config loader
│   │   ├── logging_utils.py           # Structured logging
│   │   ├── preprocessing.py           # Anomaly computation, regridding, feature extraction
│   │   └── io_utils.py                # Zarr I/O helpers
│   ├── models/
│   │   ├── xgb_model.py               # XGBoost wrapper (regression + classification)
│   │   ├── lstm_model.py              # PyTorch LSTM
│   │   ├── cnn_model.py               # PyTorch CNN
│   │   └── metrics.py                 # RMSE, MAE, correlation, R², BSS
│   └── shap_analysis/
│       ├── compute_shap.py            # TreeExplainer / GradientExplainer / DeepExplainer
│       ├── aggregate.py               # SHAP aggregation functions (8 functions)
│       └── plotting.py                # SHAP visualisation functions (8 functions)
├── scripts/
│   ├── run_preprocess.py              # Orchestrates download → preprocess
│   ├── train_xgb.py                   # Train / evaluate one XGBoost run
│   ├── train_lstm.py                  # Train / evaluate one LSTM run
│   ├── train_cnn.py                   # Train / evaluate one CNN run
│   ├── compute_shap.py                # Compute + save SHAP values
│   ├── compile_metrics.py             # Build data/metrics.csv from Zarr stores
│   ├── pbs_preprocess.sh              # PBS: 8 CPUs, 64GB, 8h, casper queue
│   ├── pbs_train_xgb.sh               # PBS: 8 CPUs, 32GB, 4h, casper queue
│   ├── pbs_train_lstm.sh              # PBS: GPU (A100), 64GB, 4h, gpudev queue
│   ├── pbs_train_cnn.sh               # PBS: GPU (A100), 64GB, 4h, gpudev queue
│   ├── pbs_compute_shap.sh            # PBS: GPU (A100), 64GB, 12h, gpudev queue
│   ├── pbs_analysis.sh                # PBS: 4 CPUs, 32GB, 2h, casper queue
│   ├── test_phase0.py                 # Smoke tests: config + logging
│   ├── test_phase1.py                 # Smoke tests: preprocessing pipeline
│   ├── test_phase2.py                 # Smoke tests: model training
│   ├── test_phase3.py                 # Smoke tests: SHAP computation
│   └── test_phase4.py                 # Smoke tests: aggregation + plotting (13 tests)
├── notebooks/
│   ├── 01_model_skill.py              # Marimo: skill vs. lead line chart
│   ├── 02_feature_importance.py       # Marimo: bar chart (model/lead/task dropdowns)
│   ├── 03_seasonal_analysis.py        # Marimo: seasonal heatmap + SPB table
│   ├── 04_enso_asymmetry.py           # Marimo: El Niño vs La Niña bars
│   ├── 05_spatial_maps.py             # Marimo: CNN spatial SHAP maps (cartopy)
│   ├── 06_lead_dependence.py          # Marimo: row-normalised heatmap across leads
│   └── 07_shap_scatter.py             # Marimo: SHAP vs prediction scatter
├── figures/                           # Output plots from pbs_analysis.sh
├── logs/                              # PBS job output logs
├── environment.yml                    # Conda environment definition
├── literature_review.md               # Key papers and literature gaps
├── memory.md                          # This file
└── README.md
```

---

## Conda Environment

**Name:** `shap-enso`  
**Location:** `/glade/work/acsubram/conda-envs/shap-enso/`  
**Python:** 3.11

Key packages: `numpy`, `pandas`, `xarray`, `scikit-learn`, `dask`, `netcdf4`, `zarr`, `pydap`, `requests`, `pyyaml`, `xesmf`, `matplotlib`, `cartopy` (conda-forge), plus `torch>=2.2`, `shap>=0.44`, `xgboost>=2.0`, `cdsapi`, `marimo` (pip).

**Create / update:**
```bash
# Set scratch TMPDIR to avoid filling /tmp during large installs
export TMPDIR=/glade/derecho/scratch/acsubram/tmp
mkdir -p $TMPDIR

module load conda
conda env create -f environment.yml -p /glade/work/acsubram/conda-envs/shap-enso
# or to update:
conda env update -f environment.yml -p /glade/work/acsubram/conda-envs/shap-enso --prune
```

---

## Data Sources

| Variable | Source | Download script | Notes |
|---|---|---|---|
| SST | ERA5 (`sea_surface_temperature`) | `data/download_era5.py` | ✅ Downloaded |
| τx (zonal wind stress) | ERA5 (`eastward_turbulent_surface_stress`) | `data/download_era5.py` | ✅ Downloaded |
| OLR | ERA5 (`top_net_thermal_radiation`) | `data/download_era5.py` | ✅ Downloaded; sign-flipped to outgoing |
| SLP | ERA5 (`mean_sea_level_pressure`) | `data/download_era5.py` | ✅ Downloaded |
| D20 proxy (OHC300) | ORAS5 via CDS (`ocean_heat_content_for_the_upper_300m`) | `data/download_d20.py` | Job running (4200546) |

**Domain:** 30°S–30°N, 120°E–290°E (0–360° grid)  
**Period:** 1979–2023 monthly  
**Climatology baseline:** 1981–2010

### ERA5 raw files (on disk)
```
data/raw/era5/sst_monthly_1979_2023.nc
data/raw/era5/tauu_monthly_1979_2023.nc
data/raw/era5/olr_monthly_1979_2023.nc
data/raw/era5/slp_monthly_1979_2023.nc
```

### D20 note — CDS API change
`depth_of_20c_isotherm` is **no longer available** in the new Copernicus CDS API for ORAS5. We use `ocean_heat_content_for_the_upper_300m` (OHC300) as a proxy, which is physically equivalent for ENSO prediction (deep thermocline = high OHC300). The variable is renamed to `"d20"` during download so all downstream code is unchanged.

---

## Implementation Phases

### Phase 0 — Scaffold ✅
**Commit:** `3d722e2`, `c5920fe`, `3557fae`

- Project directory structure and `git init`
- `configs/default.yaml` — all hyperparameters, paths, HPC settings
- `src/utils/config.py` — YAML loader with deep merge
- `src/utils/logging_utils.py` — structured logging with timestamps
- `literature_review.md` — key ENSO ML/XAI papers
- `scripts/test_phase0.py` — smoke tests for config + logging

### Phase 1 — Data Download & Preprocessing ✅
**Commit:** `42aeb5e`, `019c91a`, `fb4d7e1`

- `data/download_era5.py` — CDS API download for 4 ERA5 variables
- `data/download_d20.py` — GODAS (OPeNDAP) and ORAS5 (CDS) D20 download; computes D20 from temperature profiles for GODAS
- `data/download_noaa_indices.py` — Niño 3.4 index from NOAA ERDDAP
- `src/utils/preprocessing.py` — full pipeline:
  - Regrid to common 2° grid (xesmf bilinear)
  - Compute monthly climatology and anomalies (1981–2010 baseline)
  - Extract basin-index features (Niño 3.4, Niño 4, IOD boxes, AMM)
  - Create lagged feature matrix (lag_months = 3)
  - Build target array (Niño 3.4 at +3/+6/+12 months)
  - Save to `data/processed/` as Zarr stores
- `scripts/run_preprocess.py` — orchestrates download → preprocess
- `scripts/pbs_preprocess.sh` — PBS job (8 CPUs, 64GB, 8h, casper)
- `scripts/test_phase1.py` — smoke tests

### Phase 2 — Model Training ✅
**Commit:** `3923898`

- `src/models/xgb_model.py` — XGBoost wrapper (regression + classification, early stopping)
- `src/models/lstm_model.py` — PyTorch LSTM (2-layer, dropout, batch training)
- `src/models/cnn_model.py` — PyTorch CNN (3-layer conv, global avg pool, dropout)
- `src/models/metrics.py` — RMSE, MAE, Pearson r, R², Brier Skill Score
- `scripts/train_xgb.py`, `train_lstm.py`, `train_cnn.py` — per-model training scripts
- `scripts/pbs_train_xgb.sh` — PBS: 8 CPUs, 32GB, 4h, casper
- `scripts/pbs_train_lstm.sh`, `pbs_train_cnn.sh` — PBS: GPU A100, 64GB, 4h, gpudev
- `scripts/test_phase2.py` — smoke tests (synthetic data, forward pass, metric checks)
- Model outputs saved to `data/models/{model_type}/`

### Phase 3 — SHAP Computation ✅
**Commit:** `b3a1fa4`

- `src/shap_analysis/compute_shap.py` — three explainer wrappers:
  - `TreeExplainerWrapper` — XGBoost (exact, fast)
  - `GradientExplainerWrapper` — LSTM (gradient × input)
  - `DeepExplainerWrapper` — CNN (DeepLIFT approximation)
- `project_shap_to_grid()` — maps 1D feature SHAP values back onto lat/lon grid for CNN spatial analysis
- SHAP outputs saved as Zarr stores in `data/shap/`
- `scripts/compute_shap.py` — loads model + preprocessed data, runs explainer, saves
- `scripts/pbs_compute_shap.sh` — PBS: GPU A100, 4 CPUs, 64GB, 12h, gpudev
- `scripts/test_phase3.py` — smoke tests (dim naming fix: `"variable"` → `"var"` to avoid xarray property conflict)

### Phase 4 — SHAP Analysis & Visualisation ✅
**Commit:** `c7a816d`

- `src/shap_analysis/aggregate.py` — 8 analysis functions:
  - `load_shap_store`, `load_spatial_shap_store`
  - `global_mean_abs_shap` — mean |SHAP| per feature
  - `lead_importance_table` — importance at each lead
  - `seasonal_shap_mean` — importance by calendar month (spring barrier)
  - `spring_barrier_stats` — SPB ratio (MAM / SON)
  - `enso_composite_shap` — El Niño vs La Niña composite SHAP
  - `shap_prediction_corr` — correlation of SHAP magnitude with prediction skill
- `src/shap_analysis/plotting.py` — 8 plot functions:
  - `plot_feature_importance_bar`, `plot_shap_scatter`
  - `plot_seasonal_heatmap`, `plot_spring_barrier`
  - `plot_enso_asymmetry`, `plot_lead_importance_heatmap`
  - `plot_spatial_shap` (cartopy maps, fallback to imshow)
  - `plot_skill_vs_lead`
  - All cap `figsize` width at 11 inches (marimo WASM constraint)
- `scripts/compile_metrics.py` — builds `data/metrics.csv`
- `scripts/pbs_analysis.sh` — PBS: 4 CPUs, 32GB, 2h, casper
- 7 marimo interactive notebooks in `notebooks/`
- `scripts/test_phase4.py` — 13 smoke tests, all pass

### Environment + Cleanup ✅
**Commits:** `9cbfd91`, `a4a49aa`

- Trimmed `environment.yml` — removed unused packages (`scipy`, `dask-jobqueue`, `bottleneck`, `torchvision`, `captum`)
- Added `pydap` for GODAS OPeNDAP streaming
- Conda environment created at `/glade/work/acsubram/conda-envs/shap-enso/`

### D20 Download Fix ✅
**Commit:** `269ee07`

- `depth_of_20c_isotherm` is no longer in the new CDS API
- Replaced with `ocean_heat_content_for_the_upper_300m` (OHC300)
- Added ZIP extraction (CDS returns a zip for this variable)
- Added spatial subsetting before saving

---

## PBS Job Execution Plan

Run jobs in this order. Each step depends on the previous completing successfully.

```
Step 1 (running):  qsub scripts/pbs_preprocess.sh      → job 4200546
Step 2 (pending):  qsub scripts/pbs_train_xgb.sh       → XGBoost (CPU, casper)
Step 3 (pending):  qsub scripts/pbs_train_lstm.sh      → LSTM (GPU, gpudev)
Step 4 (pending):  qsub scripts/pbs_train_cnn.sh       → CNN (GPU, gpudev)
Step 5 (pending):  qsub scripts/pbs_compute_shap.sh    → SHAP values (GPU, gpudev)
Step 6 (pending):  qsub scripts/pbs_analysis.sh        → metrics + figures
```

Steps 2–4 can be submitted in parallel (they write to different model subdirectories). Step 5 requires all three model dirs to exist. Step 6 requires Step 5.

**Useful PBS commands:**
```bash
qstat -u acsubram              # check all your jobs
qstat -f <jobid>               # full job details
qdel <jobid>                   # cancel a job
tail -f logs/preprocess.log    # stream job output
```

---

## Bugs Fixed & Key Gotchas

| Issue | Root cause | Fix |
|---|---|---|
| `No module named 'pydap'` | Not in initial env | `conda install pydap`; added to `environment.yml` |
| GODAS THREDDS 503 | NCEI server outage | Switched to `d20_source: "oras5"` |
| ORAS5 CDS 400 (wrong year range) | All 1979-2023 as `consolidated` — invalid | Split into consolidated (≤2021) + operational (2022+) |
| ORAS5 CDS 400 (`depth_of_20c_isotherm` unavailable) | New CDS API dropped this variable | Use `ocean_heat_content_for_the_upper_300m` instead |
| CDS returns ZIP not NC | New API behavior | Added `zipfile.ZipFile` extraction in `download_oras5()` |
| `xr.Dataset.dims` FutureWarning | Newer xarray returns a set from `.dims` | Use `.sizes` everywhere for numeric dimension sizes |
| `"variable"` dim conflicts with xarray property | DataArray dim named `"variable"` shadows `.variable` attribute | Renamed to `"var"` in `project_shap_to_grid()` and `save_spatial_shap()` |
| conda env fills `/tmp` | Large PyTorch wheel | Set `TMPDIR=/glade/derecho/scratch/acsubram/tmp` before create |
| GitHub push without SSH key | No SSH configured on Casper | Use `gh auth token` embedded in HTTPS remote URL |

---

## Data Flow Summary

```
Raw NetCDF (ERA5 + ORAS5 OHC300)
        │
        ▼
src/utils/preprocessing.py
  ├─ Regrid → 2° × 2° (xesmf)
  ├─ Anomalies (1981–2010 climatology)
  ├─ Basin indices (Niño 3.4, Niño 4, IOD, AMM) × (SST, τx, OLR, SLP, D20)
  ├─ Lag-3 feature matrix
  └─ Target: Niño 3.4 at lead +3/+6/+12 months
        │
        ▼
data/processed/
  ├─ features.zarr         # (time, features)
  ├─ targets.zarr          # (time, lead)
  └─ spatial_features.zarr # (time, lat, lon, vars) — for CNN
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
XGBoost (CPU)                          LSTM / CNN (GPU)
data/models/xgb/                       data/models/lstm/, cnn/
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
             src/shap_analysis/compute_shap.py
             data/shap/{xgb,lstm,cnn}/shap_values.zarr
                       │
                       ▼
             src/shap_analysis/aggregate.py
             data/metrics.csv
             figures/*.png
                       │
                       ▼
             notebooks/01–07 (marimo interactive)
```

---

## Marimo Notebook Rules (WASM/Pyodide constraints)

All notebooks in `notebooks/` follow these hard rules:

1. **No `__import__()` inside cell bodies** — use top-level `import` only
2. **`figsize` width ≤ 11 inches** — WASM canvas limit
3. **Dropdown `value=` must equal the key string, not the display label**
4. **Use `ds.sizes["dim"]` not `ds.dims["dim"]`** — newer xarray FutureWarning
5. **Each marimo cell is a pure function** — all inputs declared as cell arguments

---

## Key Literature (see `literature_review.md`)

- Ham et al. (2019) — CNN ENSO prediction 18 months ahead (*Nature*)
- Guo et al. (2022) — SHAP for ENSO feature attribution
- Chen & Tung (2018) — spring predictability barrier mechanism
- Lim et al. (2023) — El Niño / La Niña asymmetry in ML models
- Bonan et al. (2023) — SHAP spatial maps vs physical teleconnections
