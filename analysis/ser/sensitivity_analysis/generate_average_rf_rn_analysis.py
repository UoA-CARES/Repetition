import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

RF_ORDER = [10, 20, 30, 50, 100, 200]
RN_ORDER = [1, 2, 3, 4, 5, 10]


def read_pivot(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df.reindex(index=RN_ORDER, columns=RF_ORDER)


def normalize_task_maxabs(df):
    data = df.to_numpy(dtype=float)
    max_abs = np.nanmax(np.abs(data))

    if not np.isfinite(max_abs) or max_abs == 0:
        return df.copy()

    return df / max_abs


def average_matrices(mats):
    return sum(mats) / len(mats)


def compute_win_count(mats):
    win_arrays = [(df > 0).astype(int) for df in mats]
    return sum(win_arrays)


def compute_win_rate(win_count, n_tasks):
    return win_count / n_tasks


def compute_std_heatmap(mats):
    arr = np.stack([df.to_numpy(dtype=float) for df in mats], axis=0)
    std = np.nanstd(arr, axis=0)

    return pd.DataFrame(
        std,
        index=RN_ORDER,
        columns=RF_ORDER,
    )


def compute_rank_matrices(mats):
    rank_mats = []

    for df in mats:
        flat = df.stack(dropna=False)

        ranked = flat.rank(
            ascending=False,
            method="average",
            na_option="bottom",
        )

        rank_df = ranked.unstack()
        rank_df = rank_df.reindex(index=RN_ORDER, columns=RF_ORDER)

        rank_mats.append(rank_df)

    return rank_mats


def compute_mean_rank(mats):
    rank_mats = compute_rank_matrices(mats)
    return sum(rank_mats) / len(rank_mats)


def plot_heatmap(
    df,
    out_path,
    title,
    colorbar_label,
    center_zero=True,
    vmin=None,
    vmax=None,
    fmt="+.1f",
    cmap="coolwarm",
):
    data = df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    if center_zero:
        if vmin is None or vmax is None:
            max_abs = np.nanmax(np.abs(data))
            vmin = -max_abs
            vmax = max_abs

        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    else:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(RF_ORDER)))
    ax.set_xticklabels([f"{x}k" for x in RF_ORDER])

    ax.set_yticks(np.arange(len(RN_ORDER)))
    ax.set_yticklabels(RN_ORDER)

    ax.set_xlabel("RF")
    ax.set_ylabel("RN")
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:{fmt}}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_path}")


def build_summary_df(raw_avg, norm_avg, win_count, win_rate, mean_rank, std_heatmap):
    rows = []

    for rn in RN_ORDER:
        for rf in RF_ORDER:
            rows.append(
                {
                    "RN": rn,
                    "RF": rf,
                    "configuration": f"RN={rn}, RF={rf}k",
                    "raw_mean_delta_auc_percent": raw_avg.loc[rn, rf],
                    "normalized_mean_score": norm_avg.loc[rn, rf],
                    "win_count": win_count.loc[rn, rf],
                    "win_rate": win_rate.loc[rn, rf],
                    "mean_rank": mean_rank.loc[rn, rf],
                    "std_across_tasks": std_heatmap.loc[rn, rf],
                }
            )

    return pd.DataFrame(rows)


def save_top_configurations(summary, out_path, top_k):
    ranked = summary.sort_values(
        by=[
            "mean_rank",
            "win_rate",
            "raw_mean_delta_auc_percent",
            "normalized_mean_score",
        ],
        ascending=[True, False, False, False],
    )

    top = ranked.head(top_k)
    top.to_csv(out_path, index=False)

    print(f"Saved {out_path}")
    return top


def plot_top_configurations(top, out_path, title):
    plot_df = top.copy()
    plot_df = plot_df.sort_values("mean_rank", ascending=True)

    labels = plot_df["configuration"].tolist()
    values = plot_df["mean_rank"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Mean Rank Across Tasks, Lower is Better")
    ax.set_title(title)

    for i, value in enumerate(values):
        raw = plot_df.iloc[i]["raw_mean_delta_auc_percent"]
        win = plot_df.iloc[i]["win_rate"]
        ax.text(
            value,
            i,
            f"  rank={value:.1f}, raw={raw:+.1f}%, win={win:.2f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Task-level RF-RN heatmap CSV files.",
    )

    ap.add_argument(
        "--title",
        required=True,
        help="Base title for the generated figures.",
    )

    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory where all outputs will be saved.",
    )

    ap.add_argument(
        "--prefix",
        default="rf_rn_average",
        help="Filename prefix for outputs.",
    )

    ap.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top RF-RN configurations to save and plot.",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_mats = [read_pivot(f) for f in args.files]
    norm_mats = [normalize_task_maxabs(df) for df in raw_mats]

    n_tasks = len(raw_mats)

    raw_avg = average_matrices(raw_mats)
    norm_avg = average_matrices(norm_mats)
    win_count = compute_win_count(raw_mats)
    win_rate = compute_win_rate(win_count, n_tasks)
    mean_rank = compute_mean_rank(raw_mats)
    std_heatmap = compute_std_heatmap(raw_mats)

    raw_avg_csv = out_dir / f"{args.prefix}_raw_average.csv"
    norm_avg_csv = out_dir / f"{args.prefix}_normalized_average.csv"
    win_count_csv = out_dir / f"{args.prefix}_win_count.csv"
    win_rate_csv = out_dir / f"{args.prefix}_win_rate.csv"
    mean_rank_csv = out_dir / f"{args.prefix}_mean_rank.csv"
    std_csv = out_dir / f"{args.prefix}_std.csv"
    summary_csv = out_dir / f"{args.prefix}_summary.csv"
    top_csv = out_dir / f"{args.prefix}_top_configurations.csv"

    raw_avg.to_csv(raw_avg_csv)
    norm_avg.to_csv(norm_avg_csv)
    win_count.to_csv(win_count_csv)
    win_rate.to_csv(win_rate_csv)
    mean_rank.to_csv(mean_rank_csv)
    std_heatmap.to_csv(std_csv)

    summary = build_summary_df(
        raw_avg,
        norm_avg,
        win_count,
        win_rate,
        mean_rank,
        std_heatmap,
    )

    summary.to_csv(summary_csv, index=False)

    top = save_top_configurations(
        summary,
        top_csv,
        args.top_k,
    )

    print(f"Saved {raw_avg_csv}")
    print(f"Saved {norm_avg_csv}")
    print(f"Saved {win_count_csv}")
    print(f"Saved {win_rate_csv}")
    print(f"Saved {mean_rank_csv}")
    print(f"Saved {std_csv}")
    print(f"Saved {summary_csv}")

    plot_heatmap(
        raw_avg,
        out_dir / f"{args.prefix}_raw_average.png",
        f"{args.title}: Raw Average",
        r"Mean $\Delta$AUC (%)",
        center_zero=True,
        fmt="+.1f",
    )

    plot_heatmap(
        norm_avg,
        out_dir / f"{args.prefix}_normalized_average.png",
        f"{args.title}: Task-Normalized Average",
        "Mean Task-Normalized Score",
        center_zero=True,
        vmin=-1,
        vmax=1,
        fmt="+.2f",
    )

    plot_heatmap(
        win_count,
        out_dir / f"{args.prefix}_win_count.png",
        f"{args.title}: Win Count",
        "Number of Tasks with Positive Improvement",
        center_zero=False,
        vmin=0,
        vmax=n_tasks,
        fmt=".0f",
        cmap="viridis",
    )

    plot_heatmap(
        win_rate,
        out_dir / f"{args.prefix}_win_rate.png",
        f"{args.title}: Win Rate",
        "Fraction of Tasks with Positive Improvement",
        center_zero=False,
        vmin=0,
        vmax=1,
        fmt=".2f",
        cmap="viridis",
    )

    plot_heatmap(
        mean_rank,
        out_dir / f"{args.prefix}_mean_rank.png",
        f"{args.title}: Mean Rank",
        "Mean Rank Across Tasks, Lower is Better",
        center_zero=False,
        vmin=1,
        vmax=len(RF_ORDER) * len(RN_ORDER),
        fmt=".1f",
        cmap="viridis_r",
    )

    plot_heatmap(
        std_heatmap,
        out_dir / f"{args.prefix}_std.png",
        f"{args.title}: Standard Deviation",
        "Std Across Tasks, Lower is Better",
        center_zero=False,
        vmin=0,
        vmax=np.nanmax(std_heatmap.to_numpy(dtype=float)),
        fmt=".1f",
        cmap="viridis_r",
    )

    plot_top_configurations(
        top,
        out_dir / f"{args.prefix}_top_configurations.png",
        f"{args.title}: Top RF-RN Configurations",
    )

    print("\nFinished.")
    print(f"Number of tasks: {n_tasks}")
    print(f"Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
