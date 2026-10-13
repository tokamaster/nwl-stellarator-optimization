"""Build the reactor-scale bootstrap reference from DESC's precise_QA example."""
import numpy as np

from desc.compat import rescale
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import (
    BootstrapRedlConsistency, FixAtomicNumber, FixBoundaryR, FixBoundaryZ,
    FixCurrent, FixElectronDensity, FixElectronTemperature, FixIonTemperature,
    FixPsi, ForceBalance, ObjectiveFunction,
)
from desc.profiles import PowerSeriesProfile

from solver_result import save_result


def bootstrap_term(eq):
    grid = LinearGrid(rho=np.linspace(1 / eq.L_grid, 1, eq.L_grid) - 1 / (2 * eq.L_grid),
                      M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=True)
    return BootstrapRedlConsistency(eq=eq, grid=grid, helicity=(1, 0), weight=1.0)


def fixed_boundary_constraints(eq):
    if type(eq.current).__name__ != "ScaledProfile" or not np.allclose(eq.current._profile.params[:2], 0, atol=1e-12, rtol=0):
        raise ValueError("Expected a scaled current profile with zero constant and linear coefficients")
    return (FixAtomicNumber(eq=eq), FixBoundaryR(eq=eq), FixBoundaryZ(eq=eq),
        FixCurrent(eq=eq, indices=np.array([0, 1, 2])), FixElectronDensity(eq=eq),
        FixElectronTemperature(eq=eq), FixIonTemperature(eq=eq), FixPsi(eq=eq), ForceBalance(eq=eq))


def build_reference(output, bootstrap_ftol, final_solve_ftol=None):
    eq = get("precise_QA")
    eq.pressure = None
    eq.current = PowerSeriesProfile(np.zeros(eq.L + 1), np.arange(eq.L + 1), sym=False)
    eq.atomic_number = PowerSeriesProfile([1], [0])
    eq.electron_density = PowerSeriesProfile(2.38e20 * np.array([1, -1]), [0, 10], sym=True)
    eq.electron_temperature = PowerSeriesProfile(9.45e3 * np.array([1, -1]), [0, 2], sym=True)
    eq.ion_temperature = PowerSeriesProfile(9.45e3 * np.array([1, -1]), [0, 2], sym=True)
    eq, result = eq.solve(copy=True)
    eq.save(str(output / "initial_solve_eq.h5"))
    save_result(output / "initial_solve.json", result)
    eq = rescale(eq, L=("a", 1.7), B=("<B>", 5.86), scale_pressure=False, copy=True)
    eq.save(str(output / "initial_eq.h5"))
    eq, result = eq.optimize(objective=ObjectiveFunction(bootstrap_term(eq)),
        constraints=fixed_boundary_constraints(eq), optimizer="proximal-lsq-exact",
        maxiter=50, gtol=1e-16, ftol=bootstrap_ftol, copy=True)
    eq.save(str(output / "pre_force_solve_eq.h5"))
    save_result(output / "optimizer_result.json", result)
    if final_solve_ftol is not None:
        eq, result = eq.solve(ftol=final_solve_ftol, copy=True)
    eq.save(str(output / "equilibrium.h5"))
    if final_solve_ftol is not None:
        save_result(output / "final_solve_result.json", result)
    return eq
