"""Recompute DESC validation maps and global quantities in a new output directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import desc
desc.set_device(os.environ.get("DESC_DEVICE", "gpu"))
from desc.grid import LinearGrid
from desc.objectives import NeutronWallLoading


ROOT = Path(__file__).resolve().parents[1]


CASES = {
    "axisymmetric": {
        "eq_file": "wout_kap_1p00_del_0p00.h5",
        "offset": 0.0218,
        "surface_from_file": None,
    },
    "iter": {
        "eq_file": "wout_Spong_20160107_ITER_hybridAxisymmFixedBoundary_lasymF_aScaling.h5",
        "offset": 0.05255,
        "surface_from_file": None,
    },
    "twisty": {
        "eq_file": "wout_vmec_with_profiles.h5",
        "offset": 0.0554,
        "surface_from_file": None,
    },
}


def build_surface(eq, case_dir: Path, cfg: dict):
    if cfg["surface_from_file"]:
        return desc.io.load(case_dir / cfg["surface_from_file"])
    return eq.surface.constant_offset_surface(
        cfg["offset"],
        grid=LinearGrid(M=32, N=32, sym=True),
        M=8,
        N=8,
    )


def regenerate(case: str, data_dir: Path, output_dir: Path):
    cfg = CASES[case]
    case_dir = data_dir / case
    destination = output_dir / case
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("nwl_mean", "nwl_std", "nwl_stderr", "nwl_relerr", "area_mat", "poloidal_bins", "toroidal_bins"):
        shutil.copyfile(case_dir / f"{name}.npy", destination / f"{name}.npy")

    eq = desc.io.load(case_dir / cfg["eq_file"])
    surf = build_surface(eq, case_dir, cfg)

    source_nfp = eq.NFP if eq.NFP > 1 else 1
    source_grid = LinearGrid(
        rho=np.linspace(0, 0.9999, 20),
        theta=32,
        zeta=101,
        NFP=source_nfp,
    )
    surface_grid = LinearGrid(theta=101, zeta=101, NFP=eq.NFP, endpoint=True)

    print(f"case={case}")
    print(f"  eq.NFP={eq.NFP}")
    print(f"  source nodes={source_grid.num_nodes}")
    print(f"  surface nodes={surface_grid.num_nodes}")

    obj = NeutronWallLoading(
        eq,
        surf,
        source_grid=source_grid,
        surface_grid=surface_grid,
        surface_fixed=True,
    )
    obj.build()
    values = obj.compute(eq.params_dict)
    values = np.asarray(values).reshape(
        (surface_grid.num_theta, surface_grid.num_zeta), order="F"
    )

    theta = np.asarray(surface_grid.nodes[surface_grid.unique_theta_idx, 1])
    zeta = np.asarray(surface_grid.nodes[surface_grid.unique_zeta_idx, 2])

    np.save(destination / "desc_theta.npy", theta)
    np.save(destination / "desc_zeta.npy", zeta)
    np.save(destination / "desc_nwl.npy", values)

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    levels = np.linspace(values.min(), values.max(), 80)
    contour = ax.contourf(zeta, theta, values, levels=levels, cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("NWL (MW m$^{-2}$)")
    cbar.set_ticks([values.min(), values.max()])
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(f"{case} DESC Bosch-Hale NWL")
    fig.savefig(destination / "desc_nwl.png", dpi=300)
    plt.close(fig)

    fuspow = float(eq.compute("P_fusion")["P_fusion"] / 1e6)
    pneut_domain = 0.8 * fuspow / 4.0
    area_domain = float(surf.compute("S")["S"] / 4.0)

    (destination / "desc_global_metrics.json").write_text(json.dumps({
        "neutron_power_MW": pneut_domain, "wall_area_m2": area_domain, "domain_fraction": 0.25
    }, indent=2) + "\n")

    print(f"  min={values.min():.8f} MW/m^2")
    print(f"  max={values.max():.8f} MW/m^2")
    print(f"  mean_point={values.mean():.8f} MW/m^2")
    print(f"  Pn_DESC_domain={pneut_domain:.8f} MW")
    print(f"  S_DESC_domain={area_domain:.8f} m^2")
    print(f"  Pn_over_S_DESC={pneut_domain / area_domain:.8f} MW/m^2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    regenerate(args.case, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
