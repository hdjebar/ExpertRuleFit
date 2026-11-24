"""Download the three benchmark datasets to local CSVs (B1 input).

Writes ``experiments/data/{german,taiwan,hmeq}.csv``. Each file holds the raw
features plus a binary ``target`` column (1 = default / bad). Preprocessing
(encoding, imputation, scaling) is done later, inside each train split, by
``perturbation.py``.

If this machine cannot reach UCI / creditriskanalytics.net, create the CSVs
elsewhere with this script and copy them into ``experiments/data/``.

Usage:
    python experiments/fetch_data.py [--out experiments/data]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HMEQ_URL = "https://www.creditriskanalytics.net/uploads/1/9/5/1/19511601/hmeq.csv"


def fetch_german():
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=144)  # Statlog (German Credit Data)
    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0].to_numpy()
    X["target"] = (y == 2).astype(int)  # 1 = good, 2 = bad
    return X


def fetch_taiwan():
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=350)  # Default of credit card clients
    X = ds.data.features.copy()
    X["target"] = ds.data.targets.iloc[:, 0].astype(int).to_numpy()
    return X


def fetch_hmeq():
    df = pd.read_csv(HMEQ_URL)
    df = df[~df["BAD"].isna()].copy()
    df["target"] = df.pop("BAD").astype(int)
    return df


FETCHERS = {"german": fetch_german, "taiwan": fetch_taiwan, "hmeq": fetch_hmeq}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--datasets", nargs="+", default=list(FETCHERS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        df = FETCHERS[name]()
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(
            f"{name}: {df.shape[0]} rows, {df.shape[1] - 1} features, "
            f"positive rate {np.mean(df['target']):.3f} -> {path}"
        )


if __name__ == "__main__":
    main()
