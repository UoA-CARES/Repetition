#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

RN_ORDER = [1, 2, 3, 4, 5, 10]
RF_ORDER = [10, 20, 30, 50, 100, 200]
TASKS = ["Ant", "HalfCheetah", "Humanoid", "Hopper", "Walker", "Cartpole", "Finger", "Cheetah"]
METHODS = ["ESER", "XSER", "MixSER"]
BACKBONES = ["SAC", "TD3"]

TASK_ROOT = Path("results/sensitivity_analysis/task_level_results")
OUT_ROOT = Path("results/sensitivity_analysis/aggregated_results")


def load_delta_pivot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df.reindex(index=RN_ORDER, columns=RF_ORDER)


def aggregate(method: str, backbone: str) -> None:
    frames = []
    missing = []

    for task in TASKS:
        path = TASK_ROOT / method / backbone / f"{task}-delta-pivot.csv"
        if path.exists():
            frames.append(load_delta_pivot(path))
        else:
            missing.append(str(path))

    if not frames:
        print(f"[SKIP] No files found for {method}-{backbone}")
        return

    stacked = np.stack([f.to_numpy(dtype=float) for f in frames], axis=0)
    avg = np.nanmean(stacked, axis=0)
    avg_df = pd.DataFrame(avg, index=RN_ORDER, columns=RF_ORDER)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{method.lower()}_{backbone.lower()}_average.csv"
    avg_df.to_csv(out_path)

    print(f"[SAVED] {out_path}")
    if missing:
        print(f"[WARN] Missing {len(missing)} files for {method}-{backbone}")


def main():
    for method in METHODS:
        for backbone in BACKBONES:
            aggregate(method, backbone)


if __name__ == "__main__":
    main()
