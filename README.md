# SHAP-Based Explainability for ENSO Prediction

Explainability analysis of machine learning models for El Niño–Southern Oscillation (ENSO) prediction using SHAP (SHapley Additive exPlanations).

## Overview

This project applies SHAP values to interpret ML models trained to predict ENSO indices (Niño 3.4, ONI), identifying which atmospheric and oceanic variables drive predictions across lead times and seasons.

## Project Structure

```
shap-enso/
├── data/              # Data download/preprocessing scripts
├── notebooks/         # Analysis and visualization notebooks
├── src/               # Core model training and SHAP analysis code
│   ├── models/        # ML model definitions
│   ├── shap_analysis/ # SHAP computation and aggregation
│   └── utils/         # Shared utilities
├── scripts/           # HPC job scripts (PBS/SLURM)
├── configs/           # Experiment configuration files
└── figures/           # Output figures
```

## Goals

- Train ML models (CNN, linear, ensemble) to predict Niño 3.4 at various lead times (1–12 months)
- Compute SHAP values across training ensemble / bootstrap samples
- Identify spatially coherent predictors and their seasonal dependence
- Relate SHAP-identified features to known ENSO physics (thermocline depth, Walker circulation, etc.)

## Data

- **ERA5**: SST, Z500, precip, OLR, thermocline depth (via NCAR RDA or Copernicus)
- **ERSST / HadISST**: Extended SST for longer training records
- **ENSO indices**: NOAA CPC Niño 3.4 / ONI

## Environment

```bash
conda activate shap-enso
```

See `environment.yml` for dependencies.

## HPC

Jobs run on NCAR Casper/Derecho (PBS). Example job scripts in `scripts/`.
