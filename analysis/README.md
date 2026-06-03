# Analysis

This directory contains analysis and figure-generation utilities used throughout the project.

## Structure

analysis/
├── README.md
└── sensitivity_analysis/
    ├── rf_rn_sensitivity_analysis.py
    └── README.md

## Sensitivity Analysis

The sensitivity analysis evaluates the effect of:

- RF (Repetition Frequency)
- RN (Repetition Number)

for:

- ESER
- XSER
- MixSER

under:

- SAC
- TD3

The analysis computes normalized AUC, compares results against non-repetition baselines, and generates publication-ready heatmaps used in the paper.

## Reproducibility

Large experiment logs, checkpoints, videos, and raw training outputs are intentionally excluded from the repository.

Only analysis scripts and lightweight processed results required for reproducing paper figures should be stored here.
