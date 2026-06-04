#!/usr/bin/env python3
"""
RF-RN Sensitivity Analysis

This script evaluates sensitivity of repetition-based RL methods to:

- RF: Repetition Frequency
- RN: Repetition Number

It reads experiment folders organised as:

BASE/
├── RN1/
│   ├── method_RF10/
│   │   ├── 10/data/eval.csv
│   │   ├── 20/data/eval.csv
│   │   └── ...
│   └── method_RF20/
├── RN2/
└── ...

For each RN-RF configuration, it computes normalized AUC across seeds,
compares it with a baseline, and saves heatmaps and CSV summaries.

Outputs:
- delta_pivot.csv
- auc_pivot.csv
- auc_delta_per_rn_rf.csv
- heatmap.png
- summary.txt
- runlog.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


DEFAULT_BUDGET = 1_000_000
DEFAULT_AGGREGATION = "iqm"
DEFAULT_ANCHOR_AT_ZERO = True

RN_ORDER = [1, 2, 3, 4, 5, 10]
RF_ORDER = [10, 20, 30, 50, 100, 200]

REQUIRED_COLUMNS = {"total_steps", "episode_reward"}

RN_DIR_PATTERN = re.compile(r"^RN(\d+)$", re.IGNORECASE)
RF_PATTERN = re.compile(r"(?:^|[_-])RF(\d+)(?:[_-]|$)", re.IGNORECASE)
RUN_DIR_PATTERN = re.compile(r".*RF\d+.*", re.IGNORECASE)


class Tee:
    """Write terminal output to both screen and log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def iqm(values: Iterable[float]) -> float:
    """Compute interquartile mean."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    if x.size < 4:
        return float(np.mean(x))

    x = np.sort(x)
    lo = int(np.floor(0.25 * x.size))
    hi = int(np.ceil(0.75 * x.size))

    if hi <= lo:
        return float(np.mean(x))

    return float(np.mean(x[lo:hi]))


def aggregate_values(values: Iterable[float], method: str) -> float:
    """Aggregate seed values using IQM, mean, or median."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    if method == "mean":
        return float(np.mean(x))

    if method == "median":
        return float(np.median(x))

    if method == "iqm":
        return iqm(x)

    raise ValueError(f"Unknown aggregation method: {method}")


def read_eval_csv(path: Path) -> pd.DataFrame:
    """Read eval.csv and standardise expected columns."""
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python")

    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {
        "step": "total_steps",
        "steps": "total_steps",
        "total_step": "total_steps",
        "reward": "episode_reward",
        "return": "episode_reward",
        "eval_reward": "episode_reward",
        "eval_return": "episode_reward",
        "episode_return": "episode_reward",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {path}. Found columns: {list(df.columns)}"
        )

    df["total_steps"] = pd.to_numeric(df["total_steps"], errors="coerce")
    df["episode_reward"] = pd.to_numeric(df["episode_reward"], errors="coerce")

    if "episode" in df.columns:
        df["episode"] = pd.to_numeric(df["episode"], errors="coerce")

    df = df.dropna(subset=["total_steps", "episode_reward"])
    df = df.sort_values("total_steps")

    return df


def mean_curve_per_eval_step(df: pd.DataFrame) -> pd.DataFrame:
    """Average evaluation rewards at each total_steps value."""
    curve = (
        df.groupby("total_steps", as_index=False)["episode_reward"]
        .mean()
        .sort_values("total_steps")
    )

    if curve["total_steps"].nunique() < 2:
        raise ValueError("Need at least two distinct total_steps values.")

    return curve


def normalized_auc(curve: pd.DataFrame, budget: int, anchor_at_zero: bool) -> float:
    """Compute normalized AUC up to a fixed interaction budget."""
    steps = curve["total_steps"].to_numpy(dtype=float)
    rewards = curve["episode_reward"].to_numpy(dtype=float)

    if steps.size < 2:
        return np.nan

    if anchor_at_zero and steps[0] > 0:
        steps = np.concatenate([[0.0], steps])
        rewards = np.concatenate([[rewards[0]], rewards])

    steps = np.clip(steps, 0.0, float(budget))

    if steps[-1] < budget:
        steps = np.concatenate([steps, [float(budget)]])
        rewards = np.concatenate([rewards, [rewards[-1]]])

    keep = np.concatenate([[True], np.diff(steps) > 0])
    steps = steps[keep]
    rewards = rewards[keep]

    if steps.size < 2:
        return np.nan

    return float(np.trapz(rewards, steps) / float(budget))


def extract_rf(folder_name: str) -> float:
    """Extract RF value from folder name."""
    match = RF_PATTERN.search(folder_name)
    if match is None:
        return np.nan
    return int(match.group(1))


def extract_rn(folder_name: str) -> int | None:
    """Extract RN value from RN folder name."""
    match = RN_DIR_PATTERN.match(folder_name)
    if match is None:
        return None
    return int(match.group(1))


def find_seed_eval_files(method_folder: Path) -> list[tuple[int, Path]]:
    """Find seed folders containing data/eval.csv."""
    found = []

    for child in sorted(method_folder.iterdir()):
        if not child.is_dir():
            continue

        if not child.name.isdigit():
            continue

        eval_path = child / "data" / "eval.csv"
        if eval_path.exists():
            found.append((int(child.name), eval_path))

    return found


def find_rn_folders(base_dir: Path) -> list[Path]:
    """Find RN folders inside base directory."""
    return sorted(
        [
            p for p in base_dir.iterdir()
            if p.is_dir() and RN_DIR_PATTERN.match(p.name)
        ],
        key=lambda p: extract_rn(p.name) or 0,
    )


def find_rf_run_folders(rn_dir: Path) -> list[Path]:
    """Find RF run folders inside an RN directory."""
    return sorted(
        [
            p for p in rn_dir.iterdir()
            if p.is_dir() and RUN_DIR_PATTERN.match(p.name)
        ],
        key=lambda p: extract_rf(p.name),
    )


def compute_method_auc(
    method_folder: Path,
    budget: int,
    seed_aggregation: str,
    anchor_at_zero: bool,
    verbose: bool = True,
) -> dict | None:
    """Compute seed-aggregated normalized AUC for one method folder."""
    seed_files = find_seed_eval_files(method_folder)

    if verbose:
        print(f"\n[INFO] Method folder: {method_folder}")
        print(f"[INFO] Seeds: {[s for s, _ in seed_files] if seed_files else 'NONE'}")

    if not seed_files:
        print(f"[WARN] No seed eval.csv files found in {method_folder}")
        return None

    auc_values = []
    used_seeds = []

    for seed, eval_path in seed_files:
        try:
            df = read_eval_csv(eval_path)
            curve = mean_curve_per_eval_step(df)
            auc = normalized_auc(curve, budget, anchor_at_zero)

            if np.isfinite(auc):
                auc_values.append(auc)
                used_seeds.append(seed)

            if verbose:
                print(f"[OK] seed {seed}: AUC_norm = {auc:.3f}")

        except Exception as exc:
            print(f"[SKIP] seed {seed}: {exc}")

    if not auc_values:
        print(f"[WARN] No valid AUC values for {method_folder}")
        return None

    aggregated_auc = aggregate_values(auc_values, seed_aggregation)

    if verbose:
        print(f"[RESULT] {seed_aggregation.upper()} AUC_norm = {aggregated_auc:.3f}")

    return {
        "method_folder": str(method_folder.resolve()),
        "method_name": method_folder.name,
        "rf": extract_rf(method_folder.name),
        "n_seeds_used": len(used_seeds),
        "seeds_used": ",".join(map(str, used_seeds)),
        "auc_norm": float(aggregated_auc),
    }


def delta_auc_percent(method_auc: float, baseline_auc: float) -> float:
    """Compute percentage change in normalized AUC relative to baseline."""
    return 100.0 * (float(method_auc) - float(baseline_auc)) / float(baseline_auc)


def create_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Create RN by RF pivot table with fixed ordering."""
    pivot = df.pivot_table(index="rn", columns="rf", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=RN_ORDER, columns=RF_ORDER)
    return pivot


def save_heatmap(
    pivot: pd.DataFrame,
    output_path: Path,
    title: str,
    colorbar_label: str = r"$\Delta$AUC relative to baseline (%)",
) -> None:
    """Save RF-RN heatmap from a pivot table."""
    data = pivot.to_numpy(dtype=float)

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("No finite values available for heatmap.")

    max_abs = float(np.max(np.abs(finite)))
    if max_abs == 0:
        max_abs = 1.0

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    im = ax.imshow(data, cmap="coolwarm", aspect="equal", norm=norm)

    ax.set_title(title, fontsize=22, fontweight="bold", pad=12)
    ax.set_xlabel("RF", fontsize=18, fontweight="bold")
    ax.set_ylabel("RN", fontsize=18, fontweight="bold")

    ax.set_xticks(np.arange(len(RF_ORDER)))
    ax.set_xticklabels([f"{rf}k" for rf in RF_ORDER], fontsize=14, fontweight="bold")

    ax.set_yticks(np.arange(len(RN_ORDER)))
    ax.set_yticklabels(RN_ORDER, fontsize=14, fontweight="bold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                )

    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(RF_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RN_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label, fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    output_path: Path,
    base_dir: Path,
    baseline_dir: Path,
    baseline_auc: float,
    df: pd.DataFrame,
    budget: int,
    seed_aggregation: str,
    anchor_at_zero: bool,
) -> None:
    """Write human-readable summary text."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("RF-RN Sensitivity Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Base directory: {base_dir.resolve()}\n")
        f.write(f"Baseline directory: {baseline_dir.resolve()}\n")
        f.write(f"Baseline AUC_norm: {baseline_auc:.4f}\n")
        f.write(f"Budget: {budget}\n")
        f.write(f"Seed aggregation: {seed_aggregation}\n")
        f.write(f"Anchor at zero: {anchor_at_zero}\n\n")

        f.write("Delta formula:\n")
        f.write("DeltaAUC% = 100 * (AUC_method - AUC_baseline) / AUC_baseline\n\n")

        if not df.empty:
            f.write("Global sensitivity statistics:\n")
            f.write(f"Mean DeltaAUC%: {df['delta_pct'].mean():+.2f}\n")
            f.write(f"Median DeltaAUC%: {df['delta_pct'].median():+.2f}\n")
            f.write(f"Min DeltaAUC%: {df['delta_pct'].min():+.2f}\n")
            f.write(f"Max DeltaAUC%: {df['delta_pct'].max():+.2f}\n")
            f.write(f"Positive configurations: {(df['delta_pct'] > 0).mean() * 100:.1f}%\n\n")

            best = df.loc[df["delta_pct"].idxmax()]
            f.write("Best configuration:\n")
            f.write(f"RN: {int(best['rn'])}\n")
            f.write(f"RF: {int(best['rf'])}k\n")
            f.write(f"DeltaAUC%: {best['delta_pct']:+.2f}\n")
            f.write(f"AUC_norm: {best['auc_norm']:.4f}\n\n")

            f.write("All configurations:\n")
            f.write(df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RF-RN sensitivity heatmaps and tables."
    )

    parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="Base directory containing RN folders.",
    )

    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Baseline experiment directory containing seed folders.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <base>/analysis_outputs",
    )

    parser.add_argument(
        "--title",
        "--heatmap-title",
        dest="title",
        default=None,
        help="Title shown on heatmap.",
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="Environment-step budget for normalized AUC.",
    )

    parser.add_argument(
        "--agg",
        choices=["iqm", "mean", "median"],
        default=DEFAULT_AGGREGATION,
        help="Seed aggregation method.",
    )

    parser.add_argument(
        "--no-anchor0",
        action="store_true",
        help="Disable anchor point at step 0.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce printed output.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = args.base.resolve()
    baseline_dir = args.baseline.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else base_dir / "analysis_outputs"

    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "runlog.txt"
    log_file = log_path.open("w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        verbose = not args.quiet
        anchor_at_zero = not args.no_anchor0
        title = args.title if args.title else base_dir.name

        print("[START] RF-RN sensitivity analysis")
        print(f"[TIME] {dt.datetime.now().isoformat(timespec='seconds')}")
        print(f"[BASE] {base_dir}")
        print(f"[BASELINE] {baseline_dir}")
        print(f"[OUTPUT] {output_dir}")

        baseline_result = compute_method_auc(
            baseline_dir,
            budget=args.budget,
            seed_aggregation=args.agg,
            anchor_at_zero=anchor_at_zero,
            verbose=verbose,
        )

        if baseline_result is None:
            raise RuntimeError("Could not compute baseline AUC.")

        baseline_auc = baseline_result["auc_norm"]

        if not np.isfinite(baseline_auc) or baseline_auc == 0:
            raise RuntimeError(f"Invalid baseline AUC: {baseline_auc}")

        rows = []

        rn_folders = find_rn_folders(base_dir)
        if not rn_folders:
            raise RuntimeError(f"No RN folders found in {base_dir}")

        for rn_folder in rn_folders:
            rn = extract_rn(rn_folder.name)
            rf_folders = find_rf_run_folders(rn_folder)

            if not rf_folders:
                print(f"[WARN] No RF folders found in {rn_folder}")
                continue

            for rf_folder in rf_folders:
                result = compute_method_auc(
                    rf_folder,
                    budget=args.budget,
                    seed_aggregation=args.agg,
                    anchor_at_zero=anchor_at_zero,
                    verbose=verbose,
                )

                if result is None:
                    continue

                rf = result["rf"]
                if rn is None or not np.isfinite(rf):
                    continue

                delta_pct = delta_auc_percent(result["auc_norm"], baseline_auc)

                rows.append({
                    "rn": int(rn),
                    "rf": int(rf),
                    "method_name": result["method_name"],
                    "auc_norm": result["auc_norm"],
                    "delta_pct": delta_pct,
                    "baseline_auc_norm": baseline_auc,
                    "n_seeds_used": result["n_seeds_used"],
                    "seeds_used": result["seeds_used"],
                    "method_folder": result["method_folder"],
                })

        if not rows:
            raise RuntimeError("No valid RN-RF results were collected.")

        df = pd.DataFrame(rows).sort_values(["rn", "rf", "method_name"])

        auc_pivot = create_pivot(df, "auc_norm")
        delta_pivot = create_pivot(df, "delta_pct")

        df.to_csv(output_dir / "auc_delta_per_rn_rf.csv", index=False)
        auc_pivot.to_csv(output_dir / "auc_pivot.csv")
        delta_pivot.to_csv(output_dir / "delta_pivot.csv")

        save_heatmap(
            delta_pivot,
            output_dir / "heatmap.png",
            title=title,
        )

        write_summary(
            output_dir / "summary.txt",
            base_dir=base_dir,
            baseline_dir=baseline_dir,
            baseline_auc=baseline_auc,
            df=df,
            budget=args.budget,
            seed_aggregation=args.agg,
            anchor_at_zero=anchor_at_zero,
        )

        print("[SAVED]", output_dir / "auc_delta_per_rn_rf.csv")
        print("[SAVED]", output_dir / "auc_pivot.csv")
        print("[SAVED]", output_dir / "delta_pivot.csv")
        print("[SAVED]", output_dir / "heatmap.png")
        print("[SAVED]", output_dir / "summary.txt")
        print("[SAVED]", log_path)
        print("[DONE]")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
