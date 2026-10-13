"""Run the fixed-boundary reference or one stage of its NWL continuation."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import desc

desc.set_device(os.environ.get("DESC_DEVICE", "gpu"))
import jax
from desc.backend import jnp
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.objectives import NeutronWallLoading, ObjectiveFunction, QuasisymmetryTripleProduct

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "scripts"))
from solver_result import save_result
from reactor_reference import bootstrap_term, build_reference, fixed_boundary_constraints

# Peaking weight, QS weight, smooth-max sharpness, outer iteration limit.
STAGES = {
    "peaking-0p5": (0.5, 0, 20, 60),
    "peaking-1p0": (1.0, 0, 20, 60),
    "peaking-1p25": (1.25, 0, 20, 60),
    "qs-100": (2.0, 100, 20, 120),
    "qs-110": (2.0, 110, 20, 120),
    "nwl-aware": (2.0, 110, 40, 120),
}


# The smooth peaking objective uses fixed wall-area weights because the LCFS is fixed.
class NeutronWallLoadingPeaking(NeutronWallLoading):
    """Smooth scalar approximation of max(NWL) / surface-average(NWL)."""

    _units = "(dimensionless)"
    _print_value_fmt = "NWL smooth peaking factor: "

    def __init__(self, *args, smooth_alpha=20.0, **kwargs):
        eq = kwargs.get("eq", args[0] if args else None)
        surface_grid = kwargs.get("surface_grid")
        offset = float(kwargs.get("offset", 0.1))
        if eq is None or surface_grid is None:
            raise ValueError(
                "NeutronWallLoadingPeaking requires explicit eq and surface_grid "
                "to build fixed-LCFS offset-wall area weights"
            )
        wall = eq.surface.constant_offset_surface(
            offset=offset,
            grid=surface_grid,
            M=eq.surface.M,
            N=eq.surface.N,
        )
        wall_data = wall.compute(["|e_theta x e_zeta|"], grid=surface_grid)
        area_weights = np.asarray(surface_grid.weights, dtype=float) * np.asarray(
            wall_data["|e_theta x e_zeta|"], dtype=float
        )
        if not np.all(np.isfinite(area_weights)) or np.any(area_weights <= 0):
            raise ValueError("invalid offset-wall surface-area weights")
        self._surface_area_weights = area_weights
        self._surface_area_weight_sum = float(np.sum(area_weights))
        kwargs.setdefault("target", 1.0)
        kwargs.setdefault("normalize", False)
        kwargs.setdefault("normalize_target", False)
        kwargs.setdefault("name", "NWL smooth peaking factor")
        self._smooth_alpha = float(smooth_alpha)
        super().__init__(*args, **kwargs)
        for attr in (
            "_smooth_alpha",
            "_surface_area_weights",
            "_surface_area_weight_sum",
        ):
            if attr not in self._static_attrs:
                self._static_attrs += [attr]

    def build(self, use_jit=True, verbose=1):
        super().build(use_jit=use_jit, verbose=verbose)
        self._constants["surface_area_weights"] = jnp.asarray(
            self._surface_area_weights
        )
        self._dim_f = 1
        self._constants["quad_weights"] = jnp.ones((1,))
        self._check_dimensions()
        self._set_derivatives()

    def compute(self, params_1, params_2=None, constants=None):
        nwl = super().compute(params_1, params_2, constants=constants)
        nwl = jnp.maximum(nwl, 0.0)
        if constants is None:
            constants = self.constants
        weights = constants.get(
            "surface_area_weights", jnp.asarray(self._surface_area_weights)
        )
        surface_average = jnp.sum(nwl * weights) / jnp.sum(weights)
        scaled = nwl / (surface_average + 1e-12)
        smooth_max = jax.nn.logsumexp(self._smooth_alpha * scaled)
        smooth_max = smooth_max / self._smooth_alpha
        return jnp.atleast_1d(smooth_max)


def run_continuation(eq, case, output):
    peaking_weight, qs_weight, alpha, maxiter = STAGES[case]
    tight = case in ("qs-110", "nwl-aware")
    constraints = fixed_boundary_constraints(eq)
    eq.save(str(output / "initial_eq.h5"))
    terms = [bootstrap_term(eq),
        NeutronWallLoadingPeaking(eq=eq, offset=0.15, source_grid=None,
            surface_grid=LinearGrid(rho=1, theta=30, zeta=30, NFP=eq.NFP),
            weight=peaking_weight, smooth_alpha=alpha, jac_chunk_size=75)]
    if qs_weight:
        terms.append(QuasisymmetryTripleProduct(eq=eq,
            grid=LinearGrid(rho=np.linspace(0, 0.9999, 10), theta=30, zeta=30, NFP=eq.NFP),
            weight=qs_weight, jac_chunk_size=75))
    tol = 1e-10 if tight else 1e-8
    eq, result = eq.optimize(objective=ObjectiveFunction(tuple(terms)), constraints=constraints,
        optimizer="proximal-lsq-exact", copy=True, maxiter=maxiter,
        gtol=tol, ftol=1e-9 if tight else 1e-7, xtol=tol, ctol=1e-8 if tight else 1e-6,
        options={"solve_options": dict(ftol=tol, xtol=tol, gtol=tol, maxiter=200 if tight else 100)})
    eq.save(str(output / "pre_force_solve_eq.h5"))
    save_result(output / "optimizer_result.json", result)
    solve_options = dict(ftol=1e-10, gtol=1e-10, xtol=1e-10, maxiter=200)
    if tight:
        solve_options = dict(objective="force", optimizer="lsq-exact", ftol=1e-12,
            gtol=1e-12, xtol=1e-12, maxiter=400, options={"tr_method": "svd"})
    eq, result = eq.solve(copy=True, **solve_options)
    eq.save(str(output / "equilibrium.h5"))
    save_result(output / "final_solve_result.json", result)
    return eq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("reference", *STAGES))
    parser.add_argument("--input-eq", type=Path, help="Preceding stage's equilibrium.h5")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.case == "reference" and args.input_eq is not None:
        parser.error("reference starts from precise_QA; do not pass --input-eq")
    if args.case == "nwl-aware" and args.input_eq is None:
        args.input_eq = ROOT / "data/inputs/peaking_start_eq.h5"
    if args.case != "reference" and args.input_eq is None:
        parser.error("this continuation stage requires --input-eq")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.case == "reference":
        eq = build_reference(args.output_dir, bootstrap_ftol=1e-6, final_solve_ftol=1e-3)
    else:
        eq = run_continuation(Equilibrium.load(str(args.input_eq)), args.case, args.output_dir)
    wall = eq.surface.constant_offset_surface(offset=0.15,
        grid=LinearGrid(rho=1, theta=30, zeta=30, NFP=eq.NFP), M=eq.surface.M, N=eq.surface.N)
    wall.save(str(args.output_dir / "offset_wall.h5"))


if __name__ == "__main__":
    main()
