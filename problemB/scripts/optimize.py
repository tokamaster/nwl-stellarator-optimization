"""Build the Problem B initializer or run a QS/NWL boundary-shape stage."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import desc

desc.set_device(os.environ.get("DESC_DEVICE", "gpu"))
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.objectives import (
    AspectRatio, FixAtomicNumber, FixBoundaryR, FixCurrent, FixElectronDensity,
    FixElectronTemperature, FixIonTemperature, FixPsi, ForceBalance,
    NeutronWallLoading, ObjectiveFunction, QuasisymmetryTripleProduct, RotationalTransform,
)
from desc.optimize import Optimizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "scripts"))
from solver_result import save_result
from reactor_reference import build_reference

# NWL cap (None for QS-only), outer iteration limit, gradient tolerance.
STAGES = {
    "qs-control": (None, 1500, 1e-16),
    "reference": (None, 250, 1e-8),
    "cap-0p880": (0.880, 1500, 1e-16),
    "cap-0p865": (0.865, 500, 1e-16),
    "parent": (0.835, 500, 1e-16),
    "stage-1": (0.820, 500, 1e-16),
    "stage-2": (0.805, 500, 1e-16),
    "nwl-aware": (0.790, 500, 1e-16),
}


def run_shape(eq, case, output):
    cap, maxiter, gtol = STAGES[case]
    if type(eq.current).__name__ != "ScaledProfile" or not np.allclose(eq.current._profile.params[:2], 0, rtol=0, atol=1e-12):
        raise ValueError("Expected a scaled current profile with zero constant and linear coefficients")
    initial_current = np.asarray(eq.current.params).copy()
    eq.save(str(output / "initial_eq.h5"))
    qs_grid = LinearGrid(rho=np.linspace(0, 0.9999, 10), theta=25, zeta=25, NFP=eq.NFP)
    wall_grid = LinearGrid(rho=1, theta=30, zeta=30, NFP=eq.NFP)
    objective = ObjectiveFunction(QuasisymmetryTripleProduct(eq, grid=qs_grid))
    constraints = [FixAtomicNumber(eq=eq), FixCurrent(eq=eq), FixElectronDensity(eq=eq),
        FixElectronTemperature(eq=eq), FixIonTemperature(eq=eq), FixPsi(eq=eq),
        FixBoundaryR(eq, modes=[[0, 0, 0]]), ForceBalance(eq=eq)]
    if cap is not None:
        constraints.append(NeutronWallLoading(eq, offset=0.15, source_grid=None,
            surface_grid=wall_grid, weight=10, bounds=(0, cap), jac_chunk_size=75))
    constraints.extend([RotationalTransform(eq, target=0.42, loss_function="mean"),
                        AspectRatio(eq, target=float(np.asarray(eq.compute("R0/a")["R0/a"]).squeeze()))])
    (eq,), result = Optimizer("lsq-auglag").optimize([eq], objective=objective,
        constraints=tuple(constraints), maxiter=maxiter, gtol=gtol, ftol=1e-5, copy=True)
    eq.save(str(output / "pre_force_solve_eq.h5"))
    save_result(output / "optimizer_result.json", result)
    eq, result = eq.solve(ftol=1e-8, copy=True)
    if not np.allclose(np.asarray(eq.current.params), initial_current, rtol=1e-13, atol=1e-10):
        raise ValueError("Optimization changed the fixed current profile")
    eq.save(str(output / "equilibrium.h5"))
    save_result(output / "final_solve_result.json", result)
    return eq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("initialize", *STAGES))
    parser.add_argument("--input-eq", type=Path, help="Preceding stage's equilibrium.h5")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.case == "initialize" and args.input_eq is not None:
        parser.error("initialize starts from precise_QA; do not pass --input-eq")
    if args.input_eq is None and args.case in ("reference", "nwl-aware"):
        filename = "original_qs_control_eq.h5" if args.case == "reference" else "stage_2_eq.h5"
        args.input_eq = ROOT / "data/lineage_inputs" / filename
    if args.case != "initialize" and args.input_eq is None:
        parser.error("this continuation stage requires --input-eq")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.case == "initialize":
        # Historical bootstrap ftol was 1e-2, with no extra force solve afterward.
        eq = build_reference(args.output_dir, bootstrap_ftol=1e-2)
    else:
        eq = run_shape(Equilibrium.load(str(args.input_eq)), args.case, args.output_dir)
    wall_grid = LinearGrid(rho=1, theta=30, zeta=30, NFP=eq.NFP)
    wall = eq.surface.constant_offset_surface(offset=0.15, grid=wall_grid, M=eq.surface.M, N=eq.surface.N)
    wall.save(str(args.output_dir / "offset_wall.h5"))


if __name__ == "__main__":
    main()
