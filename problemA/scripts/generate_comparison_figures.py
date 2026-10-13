#!/usr/bin/env python
"""Generate the three Problem A paper figures from saved data."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from desc import set_device

set_device(os.environ.get("DESC_DEVICE", "gpu"))

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.plotting import plot_comparison


ROOT = Path(__file__).resolve().parents[1]

COLORS = ("#0072B2", "#D55E00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, help="Reference output folder from scripts/evaluate.py A")
    parser.add_argument("--nwl-aware-dir", type=Path, help="NWL-aware output folder from scripts/evaluate.py A")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Reference")
    parser.add_argument("--optimized-label", default="NWL-aware")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_case(directory: Path | None, folder: str, prefix: str):
    if directory is not None:
        eq = Equilibrium.load(str(directory / "equilibrium.h5"))
        with np.load(directory / "diagnostics.npz") as arrays:
            data = {key: arrays[key] for key in ("nwl", "theta", "zeta", "area_weights")}
    else:
        directory = ROOT / "data" / folder
        eq = Equilibrium.load(str(directory / f"{prefix}_eq.h5"))
        data = {key: np.load(directory / f"{prefix}_{key}.npy")
                for key in ("nwl", "theta", "zeta", "area_weights")}
    return eq, data


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(q / 100.0, cdf, values))


def nwl_metrics(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    average = float(np.sum(values * weights) / np.sum(weights))
    peak = float(np.max(values))
    return {
        "minimum_MW_m2": float(np.min(values)),
        "peak_MW_m2": peak,
        "area_average_MW_m2": average,
        "area_weighted_p95_MW_m2": weighted_percentile(values, weights, 95.0),
        "peaking_factor": peak / average,
        "area_fraction_above_0p80": float(
            np.sum(weights[values > 0.80]) / np.sum(weights)
        ),
        "area_fraction_above_0p85": float(
            np.sum(weights[values > 0.85]) / np.sum(weights)
        ),
        "area_fraction_above_0p90": float(
            np.sum(weights[values > 0.90]) / np.sum(weights)
        ),
    }


def mesh_values(theta: np.ndarray, zeta: np.ndarray, values: np.ndarray):
    theta_unique = np.unique(theta)
    zeta_unique = np.unique(zeta)
    if theta_unique.size * zeta_unique.size != values.size:
        raise ValueError("NWL arrays do not form a tensor-product surface grid")
    out = np.full((theta_unique.size, zeta_unique.size), np.nan)
    theta_idx = np.searchsorted(theta_unique, theta)
    zeta_idx = np.searchsorted(zeta_unique, zeta)
    out[theta_idx, zeta_idx] = values
    return theta_unique, zeta_unique, out


def close_periodic_map(
    theta: np.ndarray,
    zeta: np.ndarray,
    values: np.ndarray,
    zeta_period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close both periodic seams of a tensor-product surface map."""
    theta_unique, zeta_unique, value_map = mesh_values(theta, zeta, values)
    value_map = np.concatenate((value_map, value_map[:, :1]), axis=1)
    value_map = np.concatenate((value_map, value_map[:1, :]), axis=0)
    theta_closed = np.concatenate((theta_unique, [theta_unique[0] + 2 * np.pi]))
    zeta_closed = np.concatenate((zeta_unique, [zeta_unique[0] + zeta_period]))
    return theta_closed, zeta_closed, value_map


def plot_flux_surface_comparison(
    baseline: Equilibrium,
    optimized: Equilibrium,
    labels: tuple[str, str],
    output_dir: Path,
    dpi: int,
) -> None:
    phis = np.linspace(0, 2 * np.pi / baseline.NFP, 4, endpoint=False)
    fig, axes = plot_comparison(
        [baseline, optimized],
        rho=np.linspace(0, 1, 8),
        theta=False,
        phi=phis,
        color=COLORS,
        ls=("-", "--"),
        lw=(1.35, 1.35),
        labels=labels,
        figsize=(8.4, 8.2),
        legend=True,
        legend_kw={"loc": "upper center", "ncol": 2, "frameon": False},
    )
    for ax, phi in zip(np.atleast_1d(axes).ravel(), phis):
        ax.set_title(rf"$\zeta={phi:.2f}$")
    fig.subplots_adjust(top=0.90, hspace=0.30, wspace=0.22)
    save_figure(fig, output_dir / "flux_surface_comparison_zeta", dpi)


def profile_data(eq: Equilibrium, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = LinearGrid(
        rho=rho,
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=eq.NFP,
        sym=eq.sym,
    )
    data = eq.compute(["iota", "current"], grid=grid)
    iota = np.asarray(grid.compress(data["iota"]), dtype=float).squeeze()
    current = np.asarray(grid.compress(data["current"]), dtype=float).squeeze()
    return iota, current


def plot_profiles(
    baseline: Equilibrium,
    optimized: Equilibrium,
    labels: tuple[str, str],
    output_dir: Path,
    dpi: int,
) -> None:
    rho = np.linspace(0, 1, 101)
    base_iota, base_current = profile_data(baseline, rho)
    opt_iota, opt_current = profile_data(optimized, rho)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharex=True)
    for values, label, color, linestyle in (
        (base_iota, labels[0], COLORS[0], "-"),
        (opt_iota, labels[1], COLORS[1], "--"),
    ):
        axes[0].plot(rho, values, color=color, linestyle=linestyle, lw=2, label=label)
    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_ylabel(r"Rotational transform $\iota$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    for values, label, color, linestyle in (
        (base_current / 1e6, labels[0], COLORS[0], "-"),
        (opt_current / 1e6, labels[1], COLORS[1], "--"),
    ):
        axes[1].plot(rho, values, color=color, linestyle=linestyle, lw=2, label=label)
    axes[1].set_xlabel(r"$\rho$")
    axes[1].set_ylabel("Enclosed toroidal current [MA]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / "iota_current_profiles", dpi)

    with (output_dir / "iota_current_profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "rho",
                "baseline_iota",
                "optimized_iota",
                "baseline_current_A",
                "optimized_current_A",
            ]
        )
        writer.writerows(zip(rho, base_iota, opt_iota, base_current, opt_current))


def plot_nwl_maps(
    theta: np.ndarray,
    zeta: np.ndarray,
    baseline_values: np.ndarray,
    optimized_values: np.ndarray,
    baseline_metrics: dict[str, float],
    optimized_metrics: dict[str, float],
    labels: tuple[str, str],
    nfp: int,
    output_dir: Path,
    dpi: int,
) -> None:
    theta_unique, zeta_unique, base_map = close_periodic_map(
        theta, zeta, baseline_values, 2 * np.pi / nfp
    )
    _, _, opt_map = close_periodic_map(
        theta, zeta, optimized_values, 2 * np.pi / nfp
    )
    absolute_min = min(float(np.min(base_map)), float(np.min(opt_map)))
    absolute_max = max(float(np.max(base_map)), float(np.max(opt_map)))
    fig = plt.figure(figsize=(10.8, 3.9))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.035),
        left=0.075,
        right=0.94,
        bottom=0.17,
        top=0.88,
        wspace=0.10,
    )
    axes = np.asarray(
        [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
        ]
    )
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)
    image = None
    for ax, values, label in zip(axes, (base_map, opt_map), labels):
        image = ax.pcolormesh(
            zeta_unique,
            theta_unique,
            values,
            cmap="viridis",
            vmin=absolute_min,
            vmax=absolute_max,
            shading="gouraud",
            rasterized=True,
        )
        if float(np.min(values)) <= 0.80 <= float(np.max(values)):
            ax.contour(
                zeta_unique,
                theta_unique,
                values,
                levels=[0.80],
                colors="white",
                linewidths=1.0,
                linestyles="--",
            )
        ax.set_title(label)
        ax.set_xlabel(r"$\zeta$")
        ax.set_xlim(0, 2 * np.pi / nfp)
        ax.set_ylim(0, 2 * np.pi)
    axes[0].set_ylabel(r"$\theta$")
    colorbar_axis = fig.add_subplot(grid[0, 2])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"NWL [MW m$^{-2}$]")
    shared_average = 0.5 * (
        baseline_metrics["area_average_MW_m2"]
        + optimized_metrics["area_average_MW_m2"]
    )
    colorbar.set_ticks([absolute_min, shared_average, absolute_max])
    save_figure(fig, output_dir / "nwl_map_comparison_absolute", dpi)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )

    baseline, base = load_case(args.reference_dir, "reference", "baseline")
    optimized, opt = load_case(args.nwl_aware_dir, "aggressive", "peaking")
    labels = (args.baseline_label, args.optimized_label)

    if baseline.NFP != optimized.NFP:
        raise ValueError("Reference and NWL-aware equilibria must have the same field period")
    if not np.array_equal(base["theta"], opt["theta"]):
        raise ValueError("Baseline and optimized theta diagnostics are not identical")
    if not np.array_equal(base["zeta"], opt["zeta"]):
        raise ValueError("Baseline and optimized zeta diagnostics are not identical")

    baseline_metrics = nwl_metrics(base["nwl"], base["area_weights"])
    optimized_metrics = nwl_metrics(opt["nwl"], opt["area_weights"])
    plot_flux_surface_comparison(
        baseline, optimized, labels, args.output_dir, args.dpi
    )
    plot_profiles(
        baseline, optimized, labels, args.output_dir, args.dpi
    )
    plot_nwl_maps(
        base["theta"],
        base["zeta"],
        base["nwl"],
        opt["nwl"],
        baseline_metrics,
        optimized_metrics,
        labels,
        baseline.NFP,
        args.output_dir,
        args.dpi,
    )

    with (args.output_dir / "nwl_metrics.json").open("w") as stream:
        json.dump({"reference": baseline_metrics, "nwl_aware": optimized_metrics}, stream, indent=2)


if __name__ == "__main__":
    main()
