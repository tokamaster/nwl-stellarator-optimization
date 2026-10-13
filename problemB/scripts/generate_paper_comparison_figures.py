#!/usr/bin/env python
"""Generate the four Problem B paper figures from saved equilibria and arrays."""

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
from desc.geometry import FourierRZToroidalSurface
from desc.plotting import plot_boundaries, plot_comparison


ROOT = Path(__file__).resolve().parents[1]

COLORS = ("#0072B2", "#D55E00")
LINESTYLES = ("-", "--")
NWL_CAP_MW_M2 = 0.790
NWL_DIAGNOSTIC_CONTOUR_MW_M2 = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, help="Reference output folder from scripts/evaluate.py B")
    parser.add_argument("--nwl-aware-dir", type=Path, help="NWL-aware output folder from scripts/evaluate.py B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-label", default="Reference")
    parser.add_argument("--nwl-aware-label", default="NWL-aware")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_case(directory: Path | None, folder: str, prefix: str):
    keys = ("nwl", "theta", "zeta", "area_weights", "iota_rho", "iota")
    if directory is not None:
        eq_path = directory / "equilibrium.h5"
        wall_path = directory / "offset_wall.h5"
        with np.load(directory / "diagnostics.npz") as arrays:
            data = {key: arrays[key] for key in keys}
    else:
        eq_path = ROOT / "data" / folder / f"{folder}_eq.h5"
        wall_path = ROOT / "data" / folder / f"{folder}_offset_wall.h5"
        archived_keys = ("nwl_MW_m2", "nwl_theta", "nwl_zeta", "nwl_area_weights", "iota_rho", "iota")
        with np.load(ROOT / "data/verification/diagnostic_arrays.npz") as arrays:
            data = {key: arrays[f"{prefix}__{archived}"] for key, archived in zip(keys, archived_keys)}
    return Equilibrium.load(str(eq_path)), FourierRZToroidalSurface.load(str(wall_path)), data


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def weighted_percentile(
    values: np.ndarray, weights: np.ndarray, percentile: float
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = float(percentile) / 100.0 * cumulative[-1]
    index = min(np.searchsorted(cumulative, cutoff, side="left"), values.size - 1)
    return float(values[index])


def nwl_metrics(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    average = float(np.sum(values * weights) / np.sum(weights))
    peak = float(np.max(values))
    return {
        "minimum_MW_m2": float(np.min(values)),
        "peak_MW_m2": peak,
        "area_weighted_average_MW_m2": average,
        "area_weighted_p95_MW_m2": weighted_percentile(values, weights, 95.0),
        "peaking_factor": peak / average,
        "area_fraction_above_0p80": float(np.sum(weights[values > 0.80]) / np.sum(weights)),
        "area_fraction_above_0p79": float(
            np.sum(weights[values > NWL_CAP_MW_M2]) / np.sum(weights)
        ),
    }


def mesh_values(
    theta: np.ndarray, zeta: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_unique = np.unique(theta)
    zeta_unique = np.unique(zeta)
    if theta_unique.size * zeta_unique.size != values.size:
        raise ValueError("NWL arrays do not form a tensor-product surface grid")
    value_map = np.full((theta_unique.size, zeta_unique.size), np.nan)
    theta_index = np.searchsorted(theta_unique, theta)
    zeta_index = np.searchsorted(zeta_unique, zeta)
    value_map[theta_index, zeta_index] = values
    return theta_unique, zeta_unique, value_map


def close_periodic_map(
    theta: np.ndarray,
    zeta: np.ndarray,
    values: np.ndarray,
    zeta_period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_unique, zeta_unique, value_map = mesh_values(theta, zeta, values)
    value_map = np.concatenate((value_map, value_map[:, :1]), axis=1)
    value_map = np.concatenate((value_map, value_map[:1, :]), axis=0)
    theta_closed = np.concatenate((theta_unique, [theta_unique[0] + 2 * np.pi]))
    zeta_closed = np.concatenate((zeta_unique, [zeta_unique[0] + zeta_period]))
    return theta_closed, zeta_closed, value_map


def plot_boundary_wall_comparison(
    reference: Equilibrium,
    nwl_aware: Equilibrium,
    reference_wall: FourierRZToroidalSurface,
    nwl_aware_wall: FourierRZToroidalSurface,
    labels: tuple[str, str],
    output_dir: Path,
    dpi: int,
) -> None:
    phis = np.linspace(0, 2 * np.pi / reference.NFP, 4, endpoint=False)
    objects = [reference, reference_wall, nwl_aware, nwl_aware_wall]
    temporary, _, data = plot_boundaries(
        objects,
        labels=["", "", "", ""],
        phi=phis,
        plot_axis=False,
        return_data=True,
    )
    plt.close(temporary)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.3), sharex=True, sharey=True)
    line_specs = (
        (COLORS[0], "-", labels[0] + " LCFS"),
        (COLORS[0], ":", labels[0] + " wall"),
        (COLORS[1], "-", labels[1] + " LCFS"),
        (COLORS[1], ":", labels[1] + " wall"),
    )
    for j, ax in enumerate(axes.ravel()):
        for i, (color, linestyle, label) in enumerate(line_specs):
            ax.plot(
                data["R"][i][:, -1, j],
                data["Z"][i][:, -1, j],
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                label=label,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.text(
            0.03,
            0.97,
            rf"$\zeta/(2\pi/N_{{\mathrm{{FP}}}})={j / len(phis):.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
        )
    axes[1, 0].set_xlabel(r"$R$ [m]")
    axes[1, 1].set_xlabel(r"$R$ [m]")
    axes[0, 0].set_ylabel(r"$Z$ [m]")
    axes[1, 0].set_ylabel(r"$Z$ [m]")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=8.5,
        columnspacing=1.5,
        handlelength=2.4,
    )
    fig.subplots_adjust(top=0.90, hspace=0.24, wspace=0.16)
    save_figure(fig, output_dir / "boundary_wall_comparison_zeta_matchedshape", dpi)


def plot_flux_surface_comparison(
    reference: Equilibrium,
    nwl_aware: Equilibrium,
    labels: tuple[str, str],
    output_dir: Path,
    dpi: int,
) -> None:
    phis = np.linspace(0, 2 * np.pi / reference.NFP, 4, endpoint=False)
    fig, axes = plot_comparison(
        [reference, nwl_aware],
        rho=np.linspace(0, 1, 8),
        theta=False,
        phi=phis,
        color=COLORS,
        ls=LINESTYLES,
        lw=(1.35, 1.35),
        labels=labels,
        figsize=(8.4, 8.2),
        legend=True,
        legend_kw={"loc": "upper center", "ncol": 2, "frameon": False},
    )
    for j, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.set_title(
            rf"$\zeta/(2\pi/N_{{\mathrm{{FP}}}})={j / len(phis):.2f}$"
        )
    fig.subplots_adjust(top=0.90, hspace=0.30, wspace=0.22)
    save_figure(fig, output_dir / "flux_surface_comparison_zeta", dpi)


def plot_profiles(
    reference: Equilibrium,
    nwl_aware: Equilibrium,
    reference_data: dict,
    nwl_aware_data: dict,
    labels: tuple[str, str],
    output_dir: Path,
    dpi: int,
) -> None:
    reference_rho = reference_data["iota_rho"]
    nwl_aware_rho = nwl_aware_data["iota_rho"]
    if not np.array_equal(reference_rho, nwl_aware_rho):
        raise ValueError("Reference and NWL-aware iota grids are not identical")
    reference_iota = reference_data["iota"]
    nwl_aware_iota = nwl_aware_data["iota"]

    current_rho = np.linspace(0, 1, 101)
    reference_current = np.asarray(reference.current(current_rho), dtype=float)
    nwl_aware_current = np.asarray(nwl_aware.current(current_rho), dtype=float)
    if not np.allclose(
        reference_current, nwl_aware_current, rtol=1e-13, atol=1e-8
    ):
        raise ValueError("Problem B current profiles are not fixed across the pair")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for values, label, color, linestyle in (
        (reference_iota, labels[0], COLORS[0], LINESTYLES[0]),
        (nwl_aware_iota, labels[1], COLORS[1], LINESTYLES[1]),
    ):
        axes[0].plot(
            reference_rho,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )
    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_ylabel(r"Rotational transform $\iota$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        current_rho,
        reference_current / 1e6,
        color=COLORS[0],
        linestyle=LINESTYLES[0],
        linewidth=2.5,
        label=labels[0],
    )
    axes[1].plot(
        current_rho,
        nwl_aware_current / 1e6,
        color=COLORS[1],
        linestyle=LINESTYLES[1],
        linewidth=1.8,
        label=labels[1],
    )
    axes[1].text(
        0.04,
        0.93,
        "fixed profile; curves coincide",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
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
                "rho_iota",
                "reference_iota",
                "nwl_aware_iota",
                "rho_current",
                "reference_current_A",
                "nwl_aware_current_A",
            ]
        )
        for row in zip(
            reference_rho,
            reference_iota,
            nwl_aware_iota,
            current_rho[: reference_rho.size],
            reference_current[: reference_rho.size],
            nwl_aware_current[: reference_rho.size],
        ):
            writer.writerow(row)
        for index in range(reference_rho.size, current_rho.size):
            writer.writerow(
                [
                    "",
                    "",
                    "",
                    current_rho[index],
                    reference_current[index],
                    nwl_aware_current[index],
                ]
            )


def plot_nwl_maps(
    theta: np.ndarray,
    zeta: np.ndarray,
    reference_values: np.ndarray,
    nwl_aware_values: np.ndarray,
    reference_metrics: dict[str, float],
    nwl_aware_metrics: dict[str, float],
    labels: tuple[str, str],
    nfp: int,
    output_dir: Path,
    dpi: int,
) -> None:
    theta_closed, zeta_closed, reference_map = close_periodic_map(
        theta, zeta, reference_values, 2 * np.pi / nfp
    )
    _, _, nwl_aware_map = close_periodic_map(
        theta, zeta, nwl_aware_values, 2 * np.pi / nfp
    )
    shared_min = min(float(np.min(reference_map)), float(np.min(nwl_aware_map)))
    shared_max = max(float(np.max(reference_map)), float(np.max(nwl_aware_map)))
    shared_average = 0.5 * (
        reference_metrics["area_weighted_average_MW_m2"]
        + nwl_aware_metrics["area_weighted_average_MW_m2"]
    )

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
    axes = np.asarray([fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])])
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)
    image = None
    for ax, values, label in zip(
        axes, (reference_map, nwl_aware_map), labels
    ):
        image = ax.pcolormesh(
            zeta_closed,
            theta_closed,
            values,
            cmap="viridis",
            vmin=shared_min,
            vmax=shared_max,
            shading="gouraud",
            rasterized=True,
        )
        if (
            float(np.min(values))
            <= NWL_DIAGNOSTIC_CONTOUR_MW_M2
            <= float(np.max(values))
        ):
            ax.contour(
                zeta_closed,
                theta_closed,
                values,
                levels=[NWL_DIAGNOSTIC_CONTOUR_MW_M2],
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
    colorbar.set_ticks([shared_min, shared_average, shared_max])
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

    reference, reference_wall, ref = load_case(args.reference_dir, "qs_control_moreiter", "qs-control")
    nwl_aware, nwl_aware_wall, aware = load_case(args.nwl_aware_dir, "stage_3", "stage-3")
    labels = (args.reference_label, args.nwl_aware_label)

    if reference.NFP != nwl_aware.NFP:
        raise ValueError("Reference and NWL-aware equilibria must have the same field period")
    if not np.array_equal(ref["theta"], aware["theta"]):
        raise ValueError("Reference and NWL-aware NWL theta arrays are not identical")
    if not np.array_equal(ref["zeta"], aware["zeta"]):
        raise ValueError("Reference and NWL-aware NWL zeta arrays are not identical")
    reference_metrics = nwl_metrics(ref["nwl"], ref["area_weights"])
    nwl_aware_metrics = nwl_metrics(aware["nwl"], aware["area_weights"])

    plot_boundary_wall_comparison(
        reference,
        nwl_aware,
        reference_wall,
        nwl_aware_wall,
        labels,
        args.output_dir,
        args.dpi,
    )
    plot_flux_surface_comparison(
        reference, nwl_aware, labels, args.output_dir, args.dpi
    )
    plot_profiles(
        reference,
        nwl_aware,
        ref,
        aware,
        labels,
        args.output_dir,
        args.dpi,
    )
    plot_nwl_maps(
        ref["theta"],
        ref["zeta"],
        ref["nwl"],
        aware["nwl"],
        reference_metrics,
        nwl_aware_metrics,
        labels,
        reference.NFP,
        args.output_dir,
        args.dpi,
    )

    with (args.output_dir / "nwl_metrics.json").open("w") as stream:
        json.dump({"reference": reference_metrics, "nwl_aware": nwl_aware_metrics}, stream, indent=2)


if __name__ == "__main__":
    main()
