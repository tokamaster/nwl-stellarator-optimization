"""Plot aligned DESC/Monte Carlo maps and recompute their comparison table."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


ROOT = Path(__file__).resolve().parent

CASES = [
    ("Circular axisym.", "axisymmetric"),
    ("ITER-like axisym.", "iter"),
    ("Non-axisym.", "twisty"),
]

def load_case(case, data_dir=ROOT):
    """Return aligned MC, DESC, signed relative difference, coordinates, and area."""
    path = data_dir / case

    desc_nwl = np.load(path / "desc_nwl.npy")
    desc_theta = np.load(path / "desc_theta.npy")
    desc_zeta = np.load(path / "desc_zeta.npy")
    mc_nwl = np.load(path / "nwl_mean.npy")
    mc_theta = np.load(path / "poloidal_bins.npy")
    mc_zeta = np.load(path / "toroidal_bins.npy")
    area = np.load(path / "area_mat.npy")

    desc_aligned = np.flip(desc_nwl, axis=0)
    if case in {"axisymmetric", "iter"} and desc_aligned.shape[1] > mc_zeta.size:
        desc_aligned = desc_aligned[:, :101]
        desc_zeta = desc_zeta[:101]

    mc_aligned = mc_nwl.T
    if desc_aligned.shape[1] == mc_zeta.size and not np.allclose(desc_zeta, mc_zeta):
        # The ITER DESC map is saved with one value per comparison column but
        # zeta labels spanning 0..2*pi. The comparison itself is by column index
        # against the one-field-period MC grid, so plot it on the MC zeta grid.
        if case not in {"axisymmetric", "iter"}:
            raise ValueError("Non-axisymmetric toroidal coordinates must match")
        desc_zeta = mc_zeta

    if desc_aligned.shape != mc_aligned.shape or area.T.shape != mc_aligned.shape:
        raise ValueError("DESC and Monte Carlo maps must use matching surface grids")
    if not all(np.all(np.isfinite(v)) for v in (desc_aligned, mc_aligned, area)) or np.any(mc_aligned <= 0) or np.any(area <= 0):
        raise ValueError("Expected finite maps, positive MC loading, and positive areas")
    signed = desc_aligned - mc_aligned
    signed_rel_pct = 100.0 * signed / mc_aligned

    return {
        "desc": desc_aligned,
        "desc_theta": desc_theta,
        "desc_zeta": desc_zeta,
        "mc": mc_aligned,
        "mc_theta": mc_theta,
        "mc_zeta": mc_zeta,
        "signed": signed,
        "signed_rel_pct": signed_rel_pct,
        "area": area,
    }


def comparison_rows(data_dir):
    rows = []
    for label, case in CASES:
        data = load_case(case, data_dir)
        global_desc = json.loads((data_dir / case / "desc_global_metrics.json").read_text())
        for method in ("MC", "DESC"):
            values = data[method.lower()]
            area = float(data["area"].sum()) if method == "MC" else global_desc["wall_area_m2"]
            power = float(np.sum(values * data["area"].T)) if method == "MC" else global_desc["neutron_power_MW"]
            rows.append(dict(case=label, case_dir=case, method=method,
                min_nwl=float(values.min()), max_nwl=float(values.max()), p_n=power, s_a=area,
                p_n_over_s_a=power / area, peaking_factor=float(values.max()) / (power / area),
                area_pct_desc_gt_mc=float(100 * data["area"].T[data["signed"] > 0].sum() / data["area"].sum())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = comparison_rows(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "comparison_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig = plt.figure(figsize=(13.5, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=3,
        ncols=5,
        width_ratios=[1.0, 1.0, 0.045, 1.0, 0.045],
        wspace=0.05,
        hspace=0.12,
    )

    axes = []
    for row, (label, case) in enumerate(CASES):
        data = load_case(case, args.data_dir)
        mc = data["mc"]
        desc = data["desc"]
        signed_rel_pct = data["signed_rel_pct"]
        desc_avg = next(row["p_n_over_s_a"] for row in rows if row["case_dir"] == case and row["method"] == "DESC")

        nwl_min = min(mc.min(), desc.min())
        nwl_max = max(mc.max(), desc.max())
        diff_min = signed_rel_pct.min()
        diff_max = signed_rel_pct.max()
        nwl_levels = np.linspace(nwl_min, nwl_max, 80)
        diff_levels = np.linspace(diff_min, diff_max, 80)

        ax_mc = fig.add_subplot(grid[row, 0])
        ax_desc = fig.add_subplot(grid[row, 1])
        ax_cbar_nwl = fig.add_subplot(grid[row, 2])
        ax_diff = fig.add_subplot(grid[row, 3])
        ax_cbar_diff = fig.add_subplot(grid[row, 4])
        axes.extend([ax_mc, ax_desc, ax_diff])

        mc_plot = ax_mc.contourf(
            data["mc_zeta"],
            data["mc_theta"],
            mc,
            levels=nwl_levels,
            cmap="viridis",
            vmin=nwl_min,
            vmax=nwl_max,
        )
        ax_desc.contourf(
            data["desc_zeta"],
            data["desc_theta"],
            desc,
            levels=nwl_levels,
            cmap="viridis",
            vmin=nwl_min,
            vmax=nwl_max,
        )
        diff_plot = ax_diff.contourf(
            data["desc_zeta"],
            data["desc_theta"],
            signed_rel_pct,
            levels=diff_levels,
            cmap="RdBu_r",
            norm=(TwoSlopeNorm(vmin=diff_min, vcenter=0.0, vmax=diff_max)
                  if diff_min < 0 < diff_max else Normalize(vmin=diff_min, vmax=diff_max)),
        )

        if row == 0:
            ax_mc.set_title("Monte Carlo", fontsize=13)
            ax_desc.set_title("DESC", fontsize=13)
            ax_diff.set_title("(DESC - MC) / MC", fontsize=13)

        ax_mc.set_ylabel(f"{label}\n$\\theta$", fontsize=12)
        for ax in (ax_mc, ax_desc, ax_diff):
            ax.set_xlabel("$\\zeta$", fontsize=11)
            ax.tick_params(labelsize=9)

        cbar_nwl = fig.colorbar(mc_plot, cax=ax_cbar_nwl)
        cbar_nwl.set_label("MW m$^{-2}$", fontsize=10)
        cbar_nwl.set_ticks([nwl_min, desc_avg, nwl_max])
        cbar_nwl.set_ticklabels(
            [f"{nwl_min:.2f}", f"{desc_avg:.2f}", f"{nwl_max:.2f}"]
        )
        cbar_nwl.ax.tick_params(labelsize=8)

        cbar_diff = fig.colorbar(diff_plot, cax=ax_cbar_diff)
        cbar_diff.set_label("%", fontsize=10)
        cbar_diff.set_ticks([diff_min, 0.0, diff_max])
        cbar_diff.ax.tick_params(labelsize=8)

    fig.savefig(args.output_dir / "validation_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote comparison figure and table")


if __name__ == "__main__":
    main()
