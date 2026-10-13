"""Compute uncertainty from a saved batch stack or a complete set of batch folders."""
import argparse
from pathlib import Path

import numpy as np


def load_batches(directory, expected_batches=None):
    directory = Path(directory)
    stack = directory / "nwl_batches.npy"
    if stack.exists():
        batches = np.load(stack, allow_pickle=False)
    else:
        folders = sorted((p for p in directory.glob("batch*") if p.is_dir() and p.name[5:].isdigit()),
                         key=lambda p: int(p.name[5:]))
        ids = [int(p.name[5:]) for p in folders]
        if not ids or ids != list(range(len(ids))):
            raise ValueError("Batch folders must be contiguous, starting with batch0")
        # Missing files and inconsistent shapes must fail; never insert zero batches.
        batches = np.stack([np.load(p / "nwl_mat.npy", allow_pickle=False) for p in folders])
        for filename in ("area_mat.npy", "poloidal_bins.npy", "toroidal_bins.npy"):
            reference = np.load(folders[0] / filename, allow_pickle=False)
            for folder in folders[1:]:
                if not np.array_equal(reference, np.load(folder / filename, allow_pickle=False)):
                    raise ValueError(f"Inconsistent {filename} across batches")
    if batches.ndim != 3 or batches.shape[0] < 2 or not np.all(np.isfinite(batches)):
        raise ValueError("Expected at least two finite two-dimensional batch maps")
    if expected_batches is not None and len(batches) != expected_batches:
        raise ValueError(f"Expected {expected_batches} batches, found {len(batches)}")
    return batches


def statistics(batches):
    mean = batches.mean(axis=0)
    std = batches.std(axis=0, ddof=1)
    stderr = std / np.sqrt(len(batches))
    relerr = np.divide(stderr, mean, out=np.full_like(mean, np.nan), where=mean != 0)
    return dict(mean=mean, std=std, stderr=stderr, relerr=relerr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-batches", type=int, default=100)
    args = parser.parse_args()
    batches = load_batches(args.data_dir, args.expected_batches)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "nwl_batches.npy", batches)
    for name, values in statistics(batches).items():
        np.save(args.output_dir / f"nwl_{name}.npy", values)
    source = args.data_dir if (args.data_dir / "area_mat.npy").exists() else args.data_dir / "batch0"
    for name in ("area_mat", "poloidal_bins", "toroidal_bins"):
        np.save(args.output_dir / f"{name}.npy", np.load(source / f"{name}.npy", allow_pickle=False))


if __name__ == "__main__":
    main()
