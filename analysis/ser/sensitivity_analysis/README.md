# RF-RN Sensitivity Analysis

This directory contains the analysis pipeline used to evaluate the sensitivity of the proposed **Spaced Episode Repetition (SER)** methods to repetition scheduling parameters.

The analysis was used to generate the sensitivity studies, heatmaps, summary figures, and supporting results reported in the paper.

---

## Overview

The objective of this analysis is to understand how different repetition schedules affect reinforcement learning performance.

Two scheduling parameters are evaluated:

* **RF (Repetition Frequency):** controls how often repetition phases are activated during training.
* **RN (Repetition Number):** controls how many repetition executions are performed once repetition is activated.

The analysis measures how different RF-RN configurations influence learning performance relative to the corresponding non-repetition baseline.

---

## Evaluated Methods

The sensitivity study includes the three proposed SER variants:

* **ESER**: Episode-based Spaced Episode Repetition
* **XSER**: Experience-based Spaced Episode Repetition
* **MixSER**: Mixed Spaced Episode Repetition

Each method is evaluated using:

* **SAC** (Soft Actor-Critic)
* **TD3** (Twin Delayed Deep Deterministic Policy Gradient)

---

## Evaluated Tasks

The experiments cover eight continuous-control benchmark environments:

* Ant-v4
* HalfCheetah-v4
* Humanoid-v4
* Hopper-v4
* Walker-Walk
* Cartpole-Balance
* Finger-Turn-Hard
* Cheetah-Run

---

## Analysis Workflow

The complete analysis pipeline follows the workflow below:

```text
Experiment Runs
        │
        ▼
Load eval.csv files
        │
        ▼
Compute normalized AUC for each seed
        │
        ▼
Aggregate across seeds
(IQM / Mean / Median)
        │
        ▼
Compare against baseline
        │
        ▼
Compute Delta-AUC (%)
        │
        ▼
Generate RF-RN result tables
        │
        ▼
Generate task-level heatmaps
        │
        ▼
Aggregate results across tasks
        │
        ▼
Generate summary heatmaps
```

---

## Files

### rf_rn_sensitivity_analysis.py

Main analysis script used for task-level sensitivity analysis.

#### What the script does

For every RF-RN configuration:

1. Automatically discovers experiment folders.
2. Finds all available seeds.
3. Loads evaluation results from:

```text
SEED/data/eval.csv
```

4. Computes normalized AUC for each seed.
5. Aggregates seed performance using:

   * IQM (default)
   * Mean
   * Median
6. Computes performance relative to the corresponding baseline.
7. Generates RF-RN heatmaps and summary tables.

#### Computed Metric

Performance is reported as:

[
\Delta AUC(%) =
\frac{AUC_{method}-AUC_{baseline}}
{AUC_{baseline}}
\times 100
]

where positive values indicate an improvement over the baseline algorithm.

#### Generated Outputs

```text
heatmap.png
auc_pivot.csv
delta_pivot.csv
auc_delta_per_rn_rf.csv
summary.txt
runlog.txt
```

---

### aggregate_rf_rn_results.py

Aggregates task-level sensitivity results across multiple environments.

#### What the script does

1. Loads RF-RN performance tables from all benchmark tasks.
2. Computes average performance across environments.
3. Produces benchmark-level RF-RN matrices.
4. Generates summary heatmaps used in the paper.
5. Identifies RF-RN regions that perform consistently well across tasks.

This script is used to generate the final sensitivity figures reported in the manuscript.

---

## Directory Structure

### raw_results/

Stores processed RF-RN results extracted from experiment runs.

```text
raw_results/
├── SAC/
│   ├── ESER/
│   ├── XSER/
│   └── MixSER/
└── TD3/
    ├── ESER/
    ├── XSER/
    └── MixSER/
```

---

### heatmaps/

Stores task-level RF-RN heatmaps.

```text
heatmaps/
├── SAC/
│   ├── ESER/
│   ├── XSER/
│   └── MixSER/
└── TD3/
    ├── ESER/
    ├── XSER/
    └── MixSER/
```

Each heatmap visualizes the percentage change in normalized AUC relative to the corresponding baseline.

---

### summary_heatmaps/

Stores benchmark-level RF-RN heatmaps averaged across all tasks.

These figures summarize the overall sensitivity trends of:

* ESER-SAC
* XSER-SAC
* MixSER-SAC
* ESER-TD3
* XSER-TD3
* MixSER-TD3

These are the primary sensitivity-analysis figures used in the paper.

---

## Example Usage

### Generate a task-level heatmap

```bash
python rf_rn_sensitivity_analysis.py \
    --base /path/to/ESER/SAC/Ant \
    --baseline /path/to/SAC/Ant \
    --heatmap-title "ESER-Ant"
```

### Generate benchmark-level summary heatmaps

```bash
python aggregate_rf_rn_results.py
```

---

## Relation to the Paper

This analysis was used to:

* Evaluate RF-RN scheduling sensitivity.
* Compare ESER, XSER, and MixSER.
* Compare SAC and TD3.
* Identify effective repetition schedules.
* Generate task-level sensitivity heatmaps.
* Generate benchmark-level summary heatmaps.
* Support the sensitivity analysis and ablation studies presented in the paper.

---

## Notes

This repository contains analysis scripts and processed results required to reproduce the reported sensitivity figures.

Large training logs, checkpoints, videos, raw experiment folders, and other heavy artifacts are intentionally excluded from version control.

