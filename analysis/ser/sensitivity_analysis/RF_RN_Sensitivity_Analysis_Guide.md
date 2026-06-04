# RF-RN Sensitivity Analysis Guide

## Overview

This document describes the purpose, methodology, inputs, outputs, and interpretation of the RF-RN sensitivity analysis performed by:

```text
generate_average_rf_rn_analysis.py
```

The script is designed to analyse the sensitivity of SER-based methods, including XSER, MixSER, and ESER, to two key hyperparameters:

* RF: Repetition Frequency
* RN: Repetition Number

The analysis aggregates results across multiple RL tasks to identify RF-RN configurations that provide strong performance improvements, robust behaviour, and consistent results across environments.

---

# 1. Why This Analysis Is Needed

When evaluating RF-RN combinations, it is not sufficient to look at a single task.

A configuration may:

* perform extremely well on one environment,
* provide moderate gains on many environments,
* improve most tasks but not be the best on any single task,
* or achieve large improvements with highly unstable behaviour.

As a result, selecting RF-RN settings solely based on average performance can be misleading.

For example:

```text
Configuration A:
+30%, +25%, +20%, -5%, -10%

Configuration B:
+12%, +11%, +10%, +13%, +12%
```

Configuration A has a higher average gain but is less reliable.

Configuration B is more consistent and may generalize better across tasks.

Therefore, this analysis evaluates RF-RN sensitivity from multiple complementary perspectives:

* Performance magnitude
* Cross-task consistency
* Robustness
* Stability

Together, these metrics provide a complete picture of RF-RN behaviour.

---

# 2. Questions Answered by This Analysis

The generated figures collectively answer the following questions:

| Question                                                     | Metric                  |
| ------------------------------------------------------------ | ----------------------- |
| Which RF-RN setting gives the highest average improvement?   | Raw Average             |
| Which RF-RN setting performs well independent of task scale? | Task-Normalized Average |
| How many tasks improve?                                      | Win Count               |
| What fraction of tasks improve?                              | Win Rate                |
| Which RF-RN settings are consistently near the top?          | Mean Rank               |
| Which RF-RN settings are most stable?                        | Standard Deviation      |

Rather than relying on a single metric, the analysis evaluates RF-RN sensitivity from all of these viewpoints.

---

# 3. Expected Input Files

The script expects one RF-RN heatmap CSV per task.

Each CSV must already be formatted as a pivot table:

* Rows represent RN values.
* Columns represent RF values.
* Cells contain performance improvements (typically ΔAUC (%)).

Example:

```csv
RN,10,20,30,50,100,200
1,...
2,...
3,...
4,...
5,...
10,...
```

Example task files:

```text
Ant-delta-pivot.csv
HalfCheetah-delta-pivot.csv
Humanoid-delta-pivot.csv
Hopper-delta-pivot.csv
Walker-delta-pivot.csv
Cartpole-delta-pivot.csv
Finger-delta-pivot.csv
Cheetah-delta-pivot.csv
```

The script aligns all tasks using:

```python
RF_ORDER = [10, 20, 30, 50, 100, 200]
RN_ORDER = [1, 2, 3, 4, 5, 10]
```

This guarantees consistent averaging across environments.

---

# 4. Running the Analysis

Example:

```bash
python generate_average_rf_rn_analysis.py \
    --files task1.csv task2.csv task3.csv \
    --title "XSER Average RF-RN Sensitivity" \
    --out-dir results \
    --prefix xser_all_tasks
```

Arguments:

| Argument  | Description                           |
| --------- | ------------------------------------- |
| --files   | List of task-level RF-RN pivot tables |
| --title   | Figure title prefix                   |
| --out-dir | Output directory                      |
| --prefix  | Output filename prefix                |

---

# 5. Generated Outputs

The script generates both CSV files and visualisations.

```text
*_raw_average.csv
*_raw_average.png

*_normalized_average.csv
*_normalized_average.png

*_win_count.csv
*_win_count.png

*_win_rate.csv
*_win_rate.png

*_mean_rank.csv
*_mean_rank.png

*_std.csv
*_std.png

*_summary.csv
```

Each output provides a different perspective on RF-RN sensitivity.

---

# 6. Raw Average Heatmap

Outputs:

```text
*_raw_average.csv
*_raw_average.png
```

## Purpose

Measures the average performance improvement across all tasks.

## Computation

```text
Raw Average = Mean(ΔAUC across tasks)
```

## Question Answered

```text
Which RF-RN configuration provides the largest average improvement?
```

## Interpretation

Higher values indicate larger average gains.

Example:

```text
RN = 4
RF = 50k

Mean ΔAUC = +13.2%
```

This means the configuration improves performance by 13.2% on average across all tasks.

## Strength

Preserves the true magnitude of improvement.

## Limitation

Large improvements on a few tasks can dominate the average.

---

# 7. Task-Normalized Average Heatmap

Outputs:

```text
*_normalized_average.csv
*_normalized_average.png
```

## Purpose

Removes task-scale effects before averaging.

## Computation

Each task heatmap is normalized by its maximum absolute response:

```text
normalized_task = task_heatmap / max(abs(task_heatmap))
```

The normalized heatmaps are then averaged.

## Question Answered

```text
Which RF-RN settings perform well when every task contributes equally?
```

## Interpretation

Higher values indicate configurations that consistently perform well across tasks.

## Strength

Prevents large-magnitude tasks from dominating the analysis.

---

# 8. Win Count Heatmap

Outputs:

```text
*_win_count.csv
*_win_count.png
```

## Purpose

Counts the number of tasks that improve.

## Computation

```text
Win Count = Number of Tasks with ΔAUC > 0
```

## Question Answered

```text
How many tasks improve under this RF-RN configuration?
```

## Interpretation

Higher values indicate broader effectiveness.

Example:

```text
Win Count = 8
```

means all eight tasks improved.

---

# 9. Win Rate Heatmap

Outputs:

```text
*_win_rate.csv
*_win_rate.png
```

## Purpose

Measures robustness across tasks.

## Computation

```text
Win Rate = Win Count / Number of Tasks
```

## Question Answered

```text
What fraction of tasks improve?
```

## Interpretation

Values close to 1 indicate improvements across nearly all tasks.

Example:

```text
Win Rate = 1.00
```

means 100% of tasks improved.

## Why It Matters

Win rate is independent of the number of tasks and is therefore easier to compare across studies.

---

# 10. Mean Rank Heatmap

Outputs:

```text
*_mean_rank.csv
*_mean_rank.png
```

## Purpose

Measures consistency across tasks.

## Computation

For each task:

```text
Best configuration = Rank 1
Second-best = Rank 2
...
Worst = Rank 36
```

Ranks are averaged across tasks.

## Question Answered

```text
Which RF-RN settings are consistently near the top?
```

## Interpretation

Lower values are better.

Example:

```text
Mean Rank = 8.9
```

means the configuration is consistently among the strongest performers across tasks.

## Why It Matters

Mean rank is often the most informative metric because it removes scale effects and focuses on consistency.

---

# 11. Standard Deviation Heatmap

Outputs:

```text
*_std.csv
*_std.png
```

## Purpose

Measures stability across tasks.

## Computation

```text
Std = Standard Deviation(ΔAUC across tasks)
```

## Question Answered

```text
Which RF-RN settings are most stable across environments?
```

## Interpretation

Lower values indicate more stable behaviour.

Higher values indicate greater variability.

Example:

```text
Std = 10
```

indicates substantially more consistent behaviour than:

```text
Std = 18
```

## Important Note

Standard deviation should never be interpreted in isolation.

Always interpret it together with:

* Raw Average
* Win Rate
* Mean Rank

A configuration can have high variance while still being highly effective.

---

# 12. How to Select the Best RF-RN Configuration

No single metric should determine the final configuration.

A strong RF-RN candidate should exhibit:

* High Raw Average
* High Win Rate
* Low Mean Rank
* Low Standard Deviation
* Strong Task-Normalized Score

Recommended priority:

1. Mean Rank
2. Win Rate
3. Raw Average
4. Standard Deviation
5. Task-Normalized Average

This prioritization favours configurations that are consistently strong across environments rather than those that achieve large gains on only a few tasks.





# Summary

The RF-RN sensitivity analysis evaluates configurations from multiple complementary perspectives.

| Figure             | Question Answered                                       |
| ------------------ | ------------------------------------------------------- |
| Raw Average        | Which RF-RN gives the highest average performance gain? |
| Normalized Average | Which RF-RN performs well independent of task scale?    |
| Win Count          | How many tasks improve?                                 |
| Win Rate           | What fraction of tasks improve?                         |
| Mean Rank          | Which RF-RN is consistently near the top?               |
| Standard Deviation | Which RF-RN is most stable across tasks?                |

Together, these metrics distinguish configurations that achieve large gains on a few environments from configurations that provide robust, consistent, and stable improvements across the entire benchmark suite.

The combination of Raw Average, Mean Rank, Win Rate, and Standard Deviation provides a comprehensive  RF-RN sensitivity analysis suitable for practical hyperparameter selection.

