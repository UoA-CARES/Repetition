# Analysis

This directory contains analysis scripts, utilities, and figure-generation code used to evaluate the proposed repetition-based reinforcement learning methods and produce publication-quality figures.

## Directory Structure

```text
analysis/
├── README.md
├── sensitivity_analysis/
│   ├── generate_rf_rn_heatmaps.py
│   ├── aggregate_rf_rn_results.py
│   ├── create_ser_summary_figure.py
│   └── README.md
│
└── paper_figures/
    ├── average_ser_heatmaps.py
    └── README.md
```

---

## sensitivity_analysis

This folder contains scripts used to analyse the sensitivity of the proposed Spaced Episode Repetition (SER) methods to repetition scheduling parameters.

### Parameters

* **RF (Repetition Frequency)**: controls how often repetition phases are activated.
* **RN (Repetition Number)**: controls how many repetitions are performed once repetition is activated.

### Supported Methods

* ESER (Episode-based Spaced Episode Repetition)
* XSER (Experience-based Spaced Episode Repetition)
* MixSER (Mixed Spaced Episode Repetition)

### Supported Algorithms

* SAC
* TD3

### Scripts

#### generate_rf_rn_heatmaps.py

Generates task-level RF-RN sensitivity heatmaps and performance tables.

Input:

```text
Method results directory
Baseline results directory
```

Outputs:

```text
TASK-delta-pivot.csv
TASK-auc-pivot.csv
TASK-auc-delta-per-rn-rf.csv
TASK-heatmap-v1.png
TASK-table.txt
```

Example:

```bash
python3 generate_rf_rn_heatmaps.py \
    --base /path/to/results \
    --baseline /path/to/baseline \
    --heatmap-title "ESER-Ant"
```

---

#### aggregate_rf_rn_results.py

Aggregates RF-RN sensitivity results across benchmark tasks.

Typical outputs:

```text
eser_sac_average.csv
eser_td3_average.csv

xser_sac_average.csv
xser_td3_average.csv

mixser_sac_average.csv
mixser_td3_average.csv
```

---

#### create_ser_summary_figure.py

Creates publication-ready summary heatmaps used in the paper.

Typical layout:

```text
ESER-SAC      XSER-SAC      MixSER-SAC
ESER-TD3      XSER-TD3      MixSER-TD3
```

---

## paper_figures

Additional scripts used for generating paper-ready figures and visualisations.

### average_ser_heatmaps.py

Computes average RF-RN heatmaps across multiple tasks and environments.

Used to create the summary sensitivity figures reported in the paper.

---

## Experimental Evaluation

The analysis scripts support evaluation across the benchmark environments used in the paper:

* Ant-v4
* HalfCheetah-v4
* Hopper-v4
* Humanoid-v4
* Walker-Walk
* Cartpole-Balance
* Finger-Turn-Hard
* Cheetah-Run

---

## Output Data

Generated results are typically stored under:

```text
results/
└── sensitivity_analysis/
    ├── task_level_results/
    ├── aggregated_results/
    └── publication_figures/
```

Large experiment logs and raw training outputs are intentionally excluded from the repository.

---

## Reproducibility

The scripts in this directory were used to generate the sensitivity analyses, heatmaps, summary figures, and supporting results reported in the paper:

**Repetition as a Third Mode of Interaction in Reinforcement Learning: Immediate and Spaced Episode Repetition for Efficient Reinforcement Learning**
