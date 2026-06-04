#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


RN_ORDER = [1, 2, 3, 4, 5, 10]
RF_ORDER = [10, 20, 30, 50, 100, 200]


def read_delta_pivot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df.reindex(index=RN_ORDER, columns=RF_ORDER)


def average_pivots(files):
    matrices = [read_delta_pivot(f) for f in files]
    stacked = np.stack([m.to_numpy(dtype=float) for m in matrices], axis=0)
    avg = np.nanmean(stacked, axis=0)
    return pd.DataFrame(avg, index=RN_ORDER, columns=RF_ORDER)


def plot_heatmap(df: pd.DataFrame, title: str, out_png: str, out_pdf: str | None):
    data = df.to_numpy(dtype=float)
    max_abs = np.nanmax(np.abs(data))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(data, cmap="coolwarm", norm=norm, aspect="auto")

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("RF", fontsize=13, fontweight="bold")
    ax.set_ylabel("RN", fontsize=13, fontweight="bold")

    ax.set_xticks(np.arange(len(RF_ORDER)))
    ax.set_xticklabels([f"{x}k" for x in RF_ORDER], fontsize=11)
    ax.set_yticks(np.arange(len(RN_ORDER)))
    ax.set_yticklabels(RN_ORDER, fontsize=11)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:+.1f}",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\Delta$AUC vs baseline (%)", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_png, dpi=350, bbox_inches="tight")
    if out_pdf:
        plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Average task-level SER delta-pivot CSV files and create an AIJ-ready heatmap."
    )
    parser.add_argument("--files", nargs="+", required=True,
                        help="List of task-level *-delta-pivot.csv files.")
    parser.add_argument("--title", required=True,
                        help="Heatmap title, e.g., ESER-SAC.")
    parser.add_argument("--outdir", default="results/paper_figures/ser_sensitivity",
                        help="Output directory.")
    parser.add_argument("--name", required=True,
                        help="Output base name, e.g., eser_sac_avg.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    avg_df = average_pivots(args.files)

    out_csv = outdir / f"{args.name}.csv"
    out_png = outdir / f"{args.name}.png"
    out_pdf = outdir / f"{args.name}.pdf"

    avg_df.to_csv(out_csv)
    plot_heatmap(avg_df, args.title, str(out_png), str(out_pdf))

    print(f"[SAVED] {out_csv}")
    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_pdf}")


if __name__ == "__main__":
    main()
