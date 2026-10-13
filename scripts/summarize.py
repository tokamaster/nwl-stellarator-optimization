"""Recompute the paper's NWL summary from the saved diagnostic arrays."""
import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def nwl_metrics(values, weights, interpolate=False):
    values, weights = np.asarray(values).ravel(), np.asarray(weights).ravel()
    if values.shape != weights.shape or not values.size:
        raise ValueError("NWL values and area weights must have the same nonempty shape")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("NWL must be finite and wall-area weights must be positive")
    average = float(np.sum(values * weights) / weights.sum())
    if average <= 0:
        raise ValueError("Mean wall loading must be positive")
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / weights.sum()
    p95 = np.interp(0.95, cdf, values[order]) if interpolate else values[order][np.searchsorted(cdf, 0.95)]
    return dict(peak_MW_m2=float(values.max()), average_MW_m2=average,
        peaking_factor=float(values.max()) / average, p95_MW_m2=float(p95),
        area_fraction_above_0p80=float(weights[values > 0.80].sum() / weights.sum()))


def saved_rows(root=ROOT):
    rows = []
    for label, folder, prefix in (("reference", "reference", "baseline"), ("nwl-aware", "aggressive", "peaking")):
        path = root / "problemA/data" / folder
        rows.append(dict(problem="A", case=label, **nwl_metrics(np.load(path / f"{prefix}_nwl.npy"),
                    np.load(path / f"{prefix}_area_weights.npy"), interpolate=True)))
    with np.load(root / "problemB/data/verification/diagnostic_arrays.npz") as arrays:
        for label, prefix in (("reference", "qs-control"), ("nwl-aware", "stage-3")):
            rows.append(dict(problem="B", case=label, **nwl_metrics(arrays[f"{prefix}__nwl_MW_m2"],
                        arrays[f"{prefix}__nwl_area_weights"])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = saved_rows(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "optimization_nwl.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
