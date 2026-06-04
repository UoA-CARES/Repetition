#!/usr/bin/env python3
"""
Combine SAC + TD3 taskwise RN summaries into paper-ready tables and plots.

Fixes:
- Bold titles, axis labels, and tick labels for all plots.
- Canonical task names for plot titles.
- Correct ALL-8 small-multiples layout order:
    Row 1: MuJoCo (-v4)   -> ant, halfcheetah, humanoid, hopper
    Row 2: DMControl      -> walker, cheetah-run, cartpole, finger-h
- Grouping rule:
    canonical name contains "-v4" => MuJoCo, else DMControl

Inputs (taskwise outputs):
  TD3 root: <td3_root>/<task>/rn_summary.csv
  SAC root: <sac_root>/<task>/rn_summary.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MEAN_CANDS = ["MeanAUC_norm", "MeanAUC", "mean_value", "mean_auc", "mean"]
DELTA_CANDS = ["DeltaPercent_vs_RN0", "delta_pct", "DeltaPercent", "DeltaAUC_percent"]
RN_CANDS = ["RN", "rn"]


# ------------------------------ Canonical names + grouping ------------------------------
def canonical_task_name(task: str) -> str:
    rename_map = {
        "ant": "Ant-v4",
        "halfcheetah": "HalfCheetah-v4",
        "humanoid": "Humanoid-v4",
        "hopper": "Hopper-v4",
        "walker": "Walker-Walk",
        "cheetah-run": "Cheetah-Run",
        "cartpole": "Cartpole-Swingup",
        "finger-h": "Finger-Turn-Hard",
    }
    return rename_map.get(task.strip().lower(), task.strip())


def task_key(task: str) -> str:
    return task.strip().lower()


def task_group_from_pretty(task_pretty: str) -> str:
    return "mujoco" if "-v4" in task_pretty else "dmcontrol"


# Small-multiples ALL-8 order (fixes Ant-v4 vs Walker-Walk placement)
TASK_ORDER_ALL_8 = [
    "ant", "halfcheetah", "humanoid", "hopper",      # MuJoCo (-v4)
    "walker", "cheetah-run", "cartpole", "finger-h"  # DMControl
]


# ------------------------------ Plot style (paper-ready, bold) ------------------------------
def set_pub_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 2.5,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
        }
    )


def bold_ticks(ax):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


def save_fig(fig, out_png):
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    out_pdf = os.path.splitext(out_png)[0] + ".pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)


# ------------------------------ Utilities ------------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"Tried {cands}. Found {list(df.columns)}")


def list_tasks(root):
    if not os.path.isdir(root):
        return []
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])


def load_rn_summary(root, task):
    path = os.path.join(root, task, "rn_summary.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    rn_col = pick_col(df, RN_CANDS)
    mean_col = pick_col(df, MEAN_CANDS)
    delta_col = pick_col(df, DELTA_CANDS)

    out = df.rename(columns={rn_col: "RN", mean_col: "Mean", delta_col: "Delta"})[["RN", "Mean", "Delta"]].copy()
    out["RN"] = out["RN"].astype(int)
    return out.sort_values("RN")


# ------------------------------ Build tables ------------------------------
def build_algo_tables(root, algo_name):
    tasks = list_tasks(root)
    rows_best = []
    rows_delta = []

    for t in tasks:
        df = load_rn_summary(root, t)
        if df is None or df.empty:
            continue

        tkey = task_key(t)
        tpretty = canonical_task_name(tkey)
        grp = task_group_from_pretty(tpretty)

        df_nz = df[df["RN"] != 0].copy()
        if df_nz.empty:
            continue

        best_row = df_nz.loc[df_nz["Mean"].astype(float).idxmax()]
        rows_best.append([tkey, tpretty, grp, int(best_row["RN"]), float(best_row["Delta"])])

        for _, r in df.iterrows():
            rows_delta.append([tkey, tpretty, grp, int(r["RN"]), float(r["Delta"])])

    best = pd.DataFrame(
        rows_best,
        columns=["Task", "TaskPretty", "Group", "Best_RN", f"DeltaAUC_percent_{algo_name}"],
    ).sort_values(["Group", "TaskPretty"])

    delta = pd.DataFrame(
        rows_delta,
        columns=["Task", "TaskPretty", "Group", "RN", f"Delta_{algo_name}"],
    )

    mean_curve = (
        delta.groupby("RN", as_index=False)
        .agg(
            mean_delta=(f"Delta_{algo_name}", "mean"),
            std_delta=(f"Delta_{algo_name}", "std"),
            n_tasks=("Task", "nunique"),
        )
        .sort_values("RN")
    )

    return best, delta, mean_curve


def robust_range_table(root, algo_name, frac=0.95):
    tasks = list_tasks(root)
    rows = []

    for t in tasks:
        df = load_rn_summary(root, t)
        if df is None or df.empty:
            continue

        tkey = task_key(t)
        tpretty = canonical_task_name(tkey)
        grp = task_group_from_pretty(tpretty)

        df_nz = df[df["RN"] != 0].copy()
        if df_nz.empty:
            continue

        best_mean = float(df_nz["Mean"].max())
        thr = frac * best_mean
        ok = df_nz[df_nz["Mean"].astype(float) >= thr]["RN"].astype(int).tolist()

        if not ok:
            rn_range = ""
        else:
            rn_min, rn_max = min(ok), max(ok)
            rn_range = f"RN{rn_min}-RN{rn_max}" if rn_min != rn_max else f"RN{rn_min}"

        rows.append([tkey, tpretty, grp, rn_range])

    return pd.DataFrame(rows, columns=["Task", "TaskPretty", "Group", f"Robust_RN_range_{algo_name}"]).sort_values(
        ["Group", "TaskPretty"]
    )


# ------------------------------ Plotting ------------------------------
def plot_mean_delta_curves(curve_sac, curve_td3, out_png, title):
    df = curve_sac.merge(curve_td3, on="RN", how="outer", suffixes=("_sac", "_td3")).sort_values("RN")

    fig, ax = plt.subplots(figsize=(7.8, 5.6), constrained_layout=True)

    if "mean_delta_sac" in df.columns:
        ax.plot(df["RN"], df["mean_delta_sac"], marker="o", markersize=7, label="SAC")
    if "mean_delta_td3" in df.columns:
        ax.plot(df["RN"], df["mean_delta_td3"], marker="s", markersize=7, label="TD3")

    ax.axhline(0.0, linestyle="--", linewidth=1.8)

    ax.set_xlabel("Repetition Number (RN)", fontweight="bold")
    ax.set_ylabel("Mean ΔAUC (%) Across Tasks", fontweight="bold")
    ax.set_title(title, fontweight="bold")

    bold_ticks(ax)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.25, linewidth=0.9)

    save_fig(fig, out_png)
    plt.close(fig)

    out = df.rename(columns={"mean_delta_sac": "mean_delta_SAC", "mean_delta_td3": "mean_delta_TD3"})
    return out


def plot_top1_frequency(best_tbl, out_png, title):
    freq = (
        best_tbl.groupby("Best_RN", as_index=False)
        .agg(n_tasks=("Task", "count"))
        .sort_values("Best_RN")
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    bars = ax.bar(freq["Best_RN"].astype(int), freq["n_tasks"].astype(int), width=0.65)

    for bar in bars:
        h = int(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h,
            f"{h}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Repetition Number (RN)", fontweight="bold")
    ax.set_ylabel("Number of Tasks", fontweight="bold")
    ax.set_title(title, fontweight="bold")

    bold_ticks(ax)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.9)

    save_fig(fig, out_png)
    plt.close(fig)

    return freq


def plot_small_multiples_combined(delta_sac, delta_td3, out_png, title, task_order=None):
    tasks_in_data = sorted(set(delta_sac["Task"]).union(set(delta_td3["Task"])))
    if not tasks_in_data:
        return

    if task_order is not None:
        ordered = [t for t in task_order if t in tasks_in_data]
        remaining = [t for t in tasks_in_data if t not in ordered]
        tasks = ordered + remaining
    else:
        tasks = tasks_in_data

    cols = 4
    rows = int(np.ceil(len(tasks) / cols))
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for i, t in enumerate(tasks, start=1):
        ax = fig.add_subplot(rows, cols, i)

        ds = delta_sac[delta_sac["Task"] == t].sort_values("RN")
        dt = delta_td3[delta_td3["Task"] == t].sort_values("RN")

        if not ds.empty:
            ax.plot(ds["RN"], ds["Delta_SAC"], marker="o", linewidth=2.2, markersize=5, label="SAC")
        if not dt.empty:
            ax.plot(dt["RN"], dt["Delta_TD3"], marker="s", linewidth=2.2, markersize=5, label="TD3")

        ax.axhline(0.0, linestyle="--", linewidth=1.2)

        ax.set_title(canonical_task_name(t), fontsize=12, fontweight="bold")
        ax.set_xlabel("RN", fontsize=11, fontweight="bold")
        ax.set_ylabel("ΔAUC (%)", fontsize=11, fontweight="bold")

        ax.tick_params(labelsize=10)
        bold_ticks(ax)
        ax.grid(True, alpha=0.25, linewidth=0.8)

        if i == 1:
            ax.legend(fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, out_png)
    plt.close(fig)


# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--td3_root", required=True, help="TD3 taskwise outputs folder, e.g., ier_rn_outputs_td3")
    ap.add_argument("--sac_root", required=True, help="SAC taskwise outputs folder, e.g., ier_rn_outputs_sac")
    ap.add_argument("--out", default="combined_pack", help="Output folder")
    args = ap.parse_args()

    set_pub_style()

    td3_root = os.path.abspath(args.td3_root)
    sac_root = os.path.abspath(args.sac_root)
    out = os.path.abspath(args.out)
    ensure_dir(out)

    # Build algorithm-specific tables
    best_sac, delta_sac, curve_sac = build_algo_tables(sac_root, "SAC")
    best_td3, delta_td3, curve_td3 = build_algo_tables(td3_root, "TD3")

    # Robust RN range tables (all tasks)
    robust_sac = robust_range_table(sac_root, "SAC", frac=0.95)
    robust_td3 = robust_range_table(td3_root, "TD3", frac=0.95)

    # Combined best table (all tasks): keep both SAC and TD3 best RN values
    combined_best = best_sac.merge(best_td3, on=["Task", "TaskPretty", "Group"], how="outer", suffixes=("_SAC", "_TD3"))
    combined_best = combined_best.merge(robust_sac[["Task", "Robust_RN_range_SAC"]], on="Task", how="left")
    combined_best = combined_best.merge(robust_td3[["Task", "Robust_RN_range_TD3"]], on="Task", how="left")
    combined_best = combined_best.sort_values(["Group", "TaskPretty"])

    # Save base CSVs
    best_sac.to_csv(os.path.join(out, "best_rn_sac.csv"), index=False)
    best_td3.to_csv(os.path.join(out, "best_rn_td3.csv"), index=False)
    combined_best.to_csv(os.path.join(out, "best_rn_combined_all.csv"), index=False)

    delta_sac.to_csv(os.path.join(out, "delta_profile_sac_all.csv"), index=False)
    delta_td3.to_csv(os.path.join(out, "delta_profile_td3_all.csv"), index=False)

    curve_sac.to_csv(os.path.join(out, "mean_delta_curve_sac_all.csv"), index=False)
    curve_td3.to_csv(os.path.join(out, "mean_delta_curve_td3_all.csv"), index=False)

    robust_sac.to_csv(os.path.join(out, "robust_range_sac.csv"), index=False)
    robust_td3.to_csv(os.path.join(out, "robust_range_td3.csv"), index=False)

    # Group splits (based on TaskPretty containing -v4)
    mujoco_tasks = combined_best[combined_best["Group"] == "mujoco"]["Task"].dropna().unique().tolist()
    dmcs_tasks = combined_best[combined_best["Group"] == "dmcontrol"]["Task"].dropna().unique().tolist()

    def filt_by_tasks(df, tasks):
        return df[df["Task"].isin(tasks)].copy()

    # Save combined best splits
    combined_best[combined_best["Group"] == "mujoco"].to_csv(os.path.join(out, "best_rn_combined_mujoco.csv"), index=False)
    combined_best[combined_best["Group"] == "dmcontrol"].to_csv(os.path.join(out, "best_rn_combined_dmcontrol.csv"), index=False)

    # Mean delta curves: ALL (from existing curves)
    merged_all = plot_mean_delta_curves(
        curve_sac.rename(columns={"mean_delta": "mean_delta_sac"}),
        curve_td3.rename(columns={"mean_delta": "mean_delta_td3"}),
        os.path.join(out, "mean_delta_curve_sac_vs_td3_all.png"),
        title="Mean ΔAUC (%) vs RN Across Tasks (ALL)",
    )
    merged_all.to_csv(os.path.join(out, "mean_delta_curve_sac_vs_td3_all.csv"), index=False)

    # For group curves, recompute mean curves from delta profiles
    def compute_curve_from_delta(delta_df, algo_col):
        return (
            delta_df.groupby("RN", as_index=False)
            .agg(
                mean_delta=(algo_col, "mean"),
                std_delta=(algo_col, "std"),
                n_tasks=("Task", "nunique"),
            )
            .sort_values("RN")
        )

    # MuJoCo curves
    ds_m = filt_by_tasks(delta_sac, mujoco_tasks)
    dt_m = filt_by_tasks(delta_td3, mujoco_tasks)
    curve_sac_m = compute_curve_from_delta(ds_m, "Delta_SAC").rename(columns={"mean_delta": "mean_delta_sac"})
    curve_td3_m = compute_curve_from_delta(dt_m, "Delta_TD3").rename(columns={"mean_delta": "mean_delta_td3"})
    merged_m = plot_mean_delta_curves(
        curve_sac_m,
        curve_td3_m,
        os.path.join(out, "mean_delta_curve_sac_vs_td3_mujoco.png"),
        title="Mean ΔAUC (%) vs RN Across Tasks (MuJoCo)",
    )
    merged_m.to_csv(os.path.join(out, "mean_delta_curve_sac_vs_td3_mujoco.csv"), index=False)

    # DMControl curves
    ds_d = filt_by_tasks(delta_sac, dmcs_tasks)
    dt_d = filt_by_tasks(delta_td3, dmcs_tasks)
    curve_sac_d = compute_curve_from_delta(ds_d, "Delta_SAC").rename(columns={"mean_delta": "mean_delta_sac"})
    curve_td3_d = compute_curve_from_delta(dt_d, "Delta_TD3").rename(columns={"mean_delta": "mean_delta_td3"})
    merged_d = plot_mean_delta_curves(
        curve_sac_d,
        curve_td3_d,
        os.path.join(out, "mean_delta_curve_sac_vs_td3_dmcontrol.png"),
        title="Mean ΔAUC (%) vs RN Across Tasks (DMControl)",
    )
    merged_d.to_csv(os.path.join(out, "mean_delta_curve_sac_vs_td3_dmcontrol.csv"), index=False)

    # Top-1 RN frequency plots (ALL + splits)
    freq_sac_all = plot_top1_frequency(
        best_sac,
        os.path.join(out, "top1_rn_frequency_sac_all.png"),
        title="Top-1 RN Frequency (SAC, ALL)",
    )
    freq_td3_all = plot_top1_frequency(
        best_td3,
        os.path.join(out, "top1_rn_frequency_td3_all.png"),
        title="Top-1 RN Frequency (TD3, ALL)",
    )
    freq_sac_all.to_csv(os.path.join(out, "top1_rn_frequency_sac.csv"), index=False)
    freq_td3_all.to_csv(os.path.join(out, "top1_rn_frequency_td3.csv"), index=False)

    best_sac_m = best_sac[best_sac["Group"] == "mujoco"].copy()
    best_td3_m = best_td3[best_td3["Group"] == "mujoco"].copy()
    best_sac_d = best_sac[best_sac["Group"] == "dmcontrol"].copy()
    best_td3_d = best_td3[best_td3["Group"] == "dmcontrol"].copy()

    plot_top1_frequency(best_sac_m, os.path.join(out, "top1_rn_frequency_sac_mujoco.png"),
                        title="Top-1 RN Frequency (SAC, MuJoCo)")
    plot_top1_frequency(best_td3_m, os.path.join(out, "top1_rn_frequency_td3_mujoco.png"),
                        title="Top-1 RN Frequency (TD3, MuJoCo)")
    plot_top1_frequency(best_sac_d, os.path.join(out, "top1_rn_frequency_sac_dmcontrol.png"),
                        title="Top-1 RN Frequency (SAC, DMControl)")
    plot_top1_frequency(best_td3_d, os.path.join(out, "top1_rn_frequency_td3_dmcontrol.png"),
                        title="Top-1 RN Frequency (TD3, DMControl)")

    # Small multiples (ALL + splits)
    plot_small_multiples_combined(
        delta_sac,
        delta_td3,
        os.path.join(out, "small_multiples_delta_sac_vs_td3_all.png"),
        title="Per-Task ΔAUC (%) vs RN: SAC vs TD3 (ALL)",
        task_order=TASK_ORDER_ALL_8,  # FIXES Ant-v4 and Walker-Walk placement
    )

    plot_small_multiples_combined(
        ds_m,
        dt_m,
        os.path.join(out, "small_multiples_delta_sac_vs_td3_mujoco.png"),
        title="Per-Task ΔAUC (%) vs RN: SAC vs TD3 (MuJoCo)",
        task_order=["ant", "halfcheetah", "humanoid", "hopper"],
    )

    plot_small_multiples_combined(
        ds_d,
        dt_d,
        os.path.join(out, "small_multiples_delta_sac_vs_td3_dmcontrol.png"),
        title="Per-Task ΔAUC (%) vs RN: SAC vs TD3 (DMControl)",
        task_order=["walker", "cheetah-run", "cartpole", "finger-h"],
    )

    print("[OK] Combined pack written to:", out)
    print("Key outputs:")
    print("  best_rn_combined_all.csv (+ mujoco, dmcontrol)")
    print("  mean_delta_curve_sac_vs_td3_all.png/pdf (+ mujoco, dmcontrol)")
    print("  top1_rn_frequency_*.png/pdf (all + splits)")
    print("  small_multiples_delta_*.png/pdf (all + splits)")


if __name__ == "__main__":
    main()

