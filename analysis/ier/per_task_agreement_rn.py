#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MEAN_CANDS  = ["MeanAUC_norm", "MeanAUC", "mean_value", "mean_auc", "mean"]
RN_CANDS    = ["RN", "rn"]

def pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"Tried {cands}. Found {list(df.columns)}")

def list_tasks(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def canonical(t: str) -> str:
    return t.strip().lower()

def load_rn_mean(root, task):
    path = os.path.join(root, task, "rn_summary.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    rn_col = pick_col(df, RN_CANDS)
    mean_col = pick_col(df, MEAN_CANDS)
    out = df.rename(columns={rn_col:"RN", mean_col:"Mean"})[["RN","Mean"]].copy()
    out["RN"] = out["RN"].astype(int)
    out["Mean"] = pd.to_numeric(out["Mean"], errors="coerce")
    out = out.dropna(subset=["Mean"]).sort_values("RN")
    return out

def top_set_within_frac(df, frac=0.95):
    """Return list of RN (excluding RN0) whose Mean is within frac of best."""
    d = df[df["RN"] != 0].copy()
    if d.empty:
        return []
    best = float(d["Mean"].max())
    thr = frac * best
    return sorted(d[d["Mean"] >= thr]["RN"].astype(int).tolist())

def rank_map(df):
    """Return dict RN -> rank (1 is best), excluding RN0."""
    d = df[df["RN"] != 0].copy()
    if d.empty:
        return {}
    d = d.sort_values("Mean", ascending=False).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d)+1)
    return dict(zip(d["RN"].astype(int), d["rank"].astype(int)))

def choose_agreement_rn(sac_df, td3_df, frac=0.95):
    sac_top = top_set_within_frac(sac_df, frac=frac)
    td3_top = top_set_within_frac(td3_df, frac=frac)
    inter = sorted(set(sac_top).intersection(set(td3_top)))

    sac_rank = rank_map(sac_df)
    td3_rank = rank_map(td3_df)

    # If intersection exists, choose RN with smallest rank-sum
    if inter:
        best_rn = min(inter, key=lambda rn: sac_rank.get(rn, 10**9) + td3_rank.get(rn, 10**9))
        reason = "intersection"
        return sac_top, td3_top, inter, best_rn, reason

    # If no intersection, choose RN with smallest rank-sum among union of top-sets
    union = sorted(set(sac_top).union(set(td3_top)))
    if union:
        best_rn = min(union, key=lambda rn: sac_rank.get(rn, 10**9) + td3_rank.get(rn, 10**9))
        reason = "rank_sum_union"
        return sac_top, td3_top, inter, best_rn, reason

    # Fallback: choose best RN by rank-sum across all RNs seen in both
    sac_all = sorted(sac_rank.keys())
    td3_all = sorted(td3_rank.keys())
    union_all = sorted(set(sac_all).union(set(td3_all)))
    if union_all:
        best_rn = min(union_all, key=lambda rn: sac_rank.get(rn, 10**9) + td3_rank.get(rn, 10**9))
        reason = "rank_sum_all"
        return sac_top, td3_top, inter, best_rn, reason

    return sac_top, td3_top, inter, None, "no_data"

def plot_freq(freq_df, out_png):
    plt.figure(figsize=(7,5))
    bars = plt.bar(freq_df["RN"].astype(int), freq_df["count"].astype(int), width=0.6)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h, f"{int(h)}",
                 ha="center", va="bottom", fontsize=13, fontweight="bold")
    plt.xlabel("RN", fontsize=16, fontweight="bold")
    plt.ylabel("Tasks selected", fontsize=16, fontweight="bold")
    plt.title("Agreement RN frequency across tasks", fontsize=18, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=400)
    plt.close()

def plot_selected_per_task(df, out_png):

    import numpy as np
    import matplotlib.pyplot as plt

    # ----- Canonical task renaming -----
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

    # ----- Define task groups -----
    mujoco = [
        "Ant-v4",
        "HalfCheetah-v4",
        "Humanoid-v4",
        "Hopper-v4",
    ]

    dmc = [
        "Walker-Walk",
        "Cheetah-Run",
        "Cartpole-Swingup",
        "Finger-Turn-Hard",
    ]

    # Apply renaming
    df2 = df.copy()
    df2["Task"] = df2["Task"].map(rename_map)

    ordered_tasks = mujoco + dmc
    df2 = df2.set_index("Task").loc[ordered_tasks].reset_index()

    # ----- Plot -----
    plt.figure(figsize=(10,6))
    y = np.arange(len(df2))

    plt.scatter(df2["Agreement_RN"], y, s=140)

    # Annotate RN numbers (bold)
    for i, rn in enumerate(df2["Agreement_RN"]):
        plt.text(
            rn + 0.05,
            i,
            f"{int(rn)}",
            fontsize=13,
            fontweight="bold",
            verticalalignment="center"
        )

    # Bold task labels
    plt.yticks(y, df2["Task"], fontsize=14, fontweight="bold")

    # Bold RN ticks
    plt.xticks(
        sorted(df2["Agreement_RN"].unique()),
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Selected RN (works for SAC and TD3)",
               fontsize=16, fontweight="bold")

    plt.title("Per-task Agreement RN",
              fontsize=18, fontweight="bold")

    # Separator line
    plt.axhline(len(mujoco)-0.5, linestyle="--", linewidth=1.5)

    # Group labels
    xmax = max(df2["Agreement_RN"]) + 0.7

    plt.text(
        xmax,
        (len(mujoco)-1)/2,
        "MuJoCo",
        fontsize=14,
        fontweight="bold",
        verticalalignment="center"
    )

    plt.text(
        xmax,
        len(mujoco)+(len(dmc)-1)/2,
        "DMControl",
        fontsize=14,
        fontweight="bold",
        verticalalignment="center"
    )

    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=400)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--td3_root", required=True)
    ap.add_argument("--sac_root", required=True)
    ap.add_argument("--outdir", default="agreement_rn_pack")
    ap.add_argument("--frac", type=float, default=0.95, help="Top-set threshold: within frac of best mean AUC")
    args = ap.parse_args()

    td3_root = os.path.abspath(args.td3_root)
    sac_root = os.path.abspath(args.sac_root)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # Match tasks by canonical names
    sac_tasks = {canonical(t): t for t in list_tasks(sac_root)}
    td3_tasks = {canonical(t): t for t in list_tasks(td3_root)}
    common_keys = sorted(set(sac_tasks.keys()).intersection(set(td3_tasks.keys())))

    rows = []
    for k in common_keys:
        sac_task = sac_tasks[k]
        td3_task = td3_tasks[k]
        sac_df = load_rn_mean(sac_root, sac_task)
        td3_df = load_rn_mean(td3_root, td3_task)
        if sac_df is None or td3_df is None:
            continue

        sac_top, td3_top, inter, best_rn, reason = choose_agreement_rn(sac_df, td3_df, frac=args.frac)

        rows.append({
            "Task": k,
            "TopSet_SAC": ",".join([f"RN{x}" for x in sac_top]),
            "TopSet_TD3": ",".join([f"RN{x}" for x in td3_top]),
            "Intersection": ",".join([f"RN{x}" for x in inter]),
            "Agreement_RN": best_rn,
            "SelectionReason": reason,
        })

    out = pd.DataFrame(rows).sort_values("Task")
    out.to_csv(os.path.join(outdir, "per_task_agreement_rn.csv"), index=False)

    # Frequency of selected RN
    freq = (out.dropna(subset=["Agreement_RN"])
              .groupby("Agreement_RN", as_index=False)
              .size()
              .rename(columns={"Agreement_RN":"RN", "size":"count"})
              .sort_values("RN"))
    freq.to_csv(os.path.join(outdir, "agreement_rn_frequency.csv"), index=False)

    # Plots
    if not freq.empty:
        plot_freq(freq, os.path.join(outdir, "agreement_rn_frequency.png"))
    if out["Agreement_RN"].notna().any():
        plot_selected_per_task(out.dropna(subset=["Agreement_RN"]), os.path.join(outdir, "per_task_agreement_rn.png"))

    print("[OK] Wrote:", outdir)
    print("  per_task_agreement_rn.csv")
    print("  agreement_rn_frequency.csv")
    print("  agreement_rn_frequency.png")
    print("  per_task_agreement_rn.png")

if __name__ == "__main__":
    main()

