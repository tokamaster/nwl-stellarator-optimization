"""Evaluate a saved optimization equilibrium on the paper's diagnostic grids."""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import desc

desc.set_device(os.environ.get("DESC_DEVICE", "gpu"))
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives import BootstrapRedlConsistency, ForceBalance, NeutronWallLoading, ObjectiveFunction, QuasisymmetryTripleProduct
from desc.utils import rpz2xyz
from summarize import nwl_metrics


def stats(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite diagnostic values")
    return dict(max_abs=float(np.max(np.abs(values))), mean_abs=float(np.mean(np.abs(values))),
                rms=float(np.sqrt(np.mean(values**2))))


def objective_values(term, eq):
    obj = ObjectiveFunction(term)
    obj.build(use_jit=False, verbose=0)
    return np.asarray(obj.compute_unscaled(obj.x(eq))), np.asarray(obj.compute_scaled(obj.x(eq)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=("A", "B"))
    parser.add_argument("equilibrium", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    eq = Equilibrium.load(str(args.equilibrium))
    is_a = args.problem == "A"
    size = 30 if is_a else 40
    surface_grid = LinearGrid(rho=1, theta=size, zeta=size, NFP=eq.NFP)
    wall = eq.surface.constant_offset_surface(offset=0.15, grid=surface_grid, M=eq.surface.M, N=eq.surface.N)
    wall.save(str(args.output_dir / "offset_wall.h5"))
    nwl, _ = objective_values(NeutronWallLoading(eq=eq, offset=0.15, source_grid=None,
                                surface_grid=surface_grid, jac_chunk_size=75), eq)
    wd = wall.compute(["|e_theta x e_zeta|", "R", "phi", "Z"], grid=surface_grid)
    pd = eq.surface.compute(["R", "phi", "Z"], grid=surface_grid)
    weights = np.asarray(surface_grid.weights) * np.asarray(wd["|e_theta x e_zeta|"])
    xyz = lambda d: np.asarray(rpz2xyz(np.column_stack([d["R"], d["phi"], d["Z"]])))
    distances = np.linalg.norm(xyz(wd) - xyz(pd), axis=1)
    qs_grid = LinearGrid(rho=np.linspace(0, 0.9999, 10) if is_a else 10, theta=25, zeta=25, NFP=eq.NFP)
    force_grid = None if is_a else QuadratureGrid(L=12, M=12, N=12, NFP=eq.NFP)
    qs_raw, qs = objective_values(QuasisymmetryTripleProduct(eq, grid=qs_grid), eq)
    force_raw, force = objective_values(ForceBalance(eq, grid=force_grid), eq)
    rho = np.linspace(0, 1, 101 if is_a else 51)
    iota_grid = (LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
                if is_a else LinearGrid(rho=rho, NFP=eq.NFP))
    iota = np.asarray(iota_grid.compress(eq.compute("iota", grid=iota_grid)["iota"]))
    metrics = dict(nwl=nwl_metrics(nwl, weights, interpolate=is_a),
        quasisymmetry=stats(qs), force_balance=stats(force),
        iota=dict(min=float(iota.min()), mean=float(iota.mean()), max=float(iota.max())),
        aspect_ratio=float(np.asarray(eq.compute("R0/a")["R0/a"]).squeeze()),
        wall_offset_m=0.15, matched_point_wall_distance_m=dict(min=float(distances.min()),
            mean=float(distances.mean()), max=float(distances.max()), area_weighted_mean=float(np.average(distances, weights=weights))))
    if is_a:
        grid = LinearGrid(rho=np.linspace(1 / eq.L_grid, 1, eq.L_grid) - 1 / (2 * eq.L_grid),
                          M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=True)
        term = BootstrapRedlConsistency(eq=eq, grid=grid, helicity=(1, 0))
        term.build(use_jit=False, verbose=0)
        # Pointwise normalization, without the ObjectiveFunction quadrature weighting.
        bootstrap = np.asarray(term.compute_unscaled(eq.params_dict)) / term.normalization
        metrics["bootstrap_pointwise_normalized"] = stats(bootstrap)
    np.savez(args.output_dir / "diagnostics.npz", nwl=nwl, area_weights=weights,
        theta=surface_grid.nodes[:, 1], zeta=surface_grid.nodes[:, 2], qs_raw=qs_raw,
        qs_normalized=qs, force_raw=force_raw, force_normalized=force, iota_rho=rho, iota=iota)
    shutil.copyfile(args.equilibrium, args.output_dir / "equilibrium.h5")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
