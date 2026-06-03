# RF-RN Sensitivity Analysis

This directory contains all code and outputs used to analyse the sensitivity of SER to:

- RF (Repetition Frequency)
- RN (Repetition Number)

## Variants

- ESER
- XSER
- MixSER

## Algorithms

- SAC
- TD3

## Tasks

- Ant-v4
- HalfCheetah-v4
- Humanoid-v4
- Hopper-v4
- Walker-Walk
- Cartpole-Balance
- Finger-Turn-Hard
- Cheetah-Run

## Generate Task-Level Heatmaps

Example:

python generate_rf_rn_heatmaps.py \
    --base <experiment_folder> \
    --baseline <baseline_folder> \
    --heatmap-title "ESER-Ant"

Outputs:

- RF-RN heatmap
- Percentage change relative to baseline

## Main Paper Figures

The paper reports:

1. Task-level heatmaps (Appendix)
2. Average heatmaps across all tasks
3. Best RF-RN configurations
4. Mean performance across tasks

## Folder Structure

raw_results/
    Experiment outputs

heatmaps/
    Task-specific RF-RN heatmaps

summary_heatmaps/
    Average heatmaps used in the main paper
