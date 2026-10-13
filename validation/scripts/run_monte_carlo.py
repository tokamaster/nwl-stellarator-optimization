"""Run one independent ParaStell/OpenMC batch for a validation case."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
import pystell

from parastell import nwl_utils, parastell
from fusion_source import plasma_conditions, reaction_rate


ROOT = Path(__file__).resolve().parents[1]
VMEC_FILES = {
    "axisymmetric": "wout_kap_1p00_del_0p00.nc",
    "iter": "wout_Spong_20160107_ITER_hybridAxisymmFixedBoundary_lasymF_aScaling.nc",
    "twisty": "wout_vmec.nc",
}


def main():
    parser = argparse.ArgumentParser(description="Run one independent Monte Carlo batch")
    parser.add_argument("case", choices=VMEC_FILES)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    vmec_file = (args.data_dir / args.case / VMEC_FILES[args.case]).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output_dir)

    stellarator = parastell.Stellarator(str(vmec_file))
    equilibrium = pystell.VMECData(str(vmec_file))
    nfp = equilibrium.nfp

    if args.case != "twisty":
        toroidal_extent = 90.0
    elif nfp == 1 or nfp == 2:
        toroidal_extent = 360.0
    else:
        toroidal_extent = 360.0 / nfp

    num_toroidal_angles = max(9, int(toroidal_extent / 90.0 * 9))
    toroidal_angles = np.linspace(0, toroidal_extent, num_toroidal_angles)
    poloidal_angles = np.linspace(0, 360, 13)
    wall_s = 1.05
    num_ribs = max(150, int(toroidal_extent / 1.5 + 1))
    num_rib_pts = max(160, int(toroidal_extent / 1.5 + 10))

    radial_build = {
        "void_component": {
            "thickness_matrix": 10
            * np.ones((len(toroidal_angles), len(poloidal_angles))),
            "mat_tag": "Vacuum",
        }
    }

    stellarator.construct_invessel_build(
        toroidal_angles,
        poloidal_angles,
        wall_s,
        radial_build,
        use_pydagmc=True,
        num_ribs=num_ribs,
        num_rib_pts=num_rib_pts,
    )
    stellarator.export_invessel_build_step()

    mesh_size = (11, 81, int(toroidal_extent / 1.5 + 1))
    source_mesh_filename = Path("source_mesh").with_suffix(".h5m")
    stellarator.construct_source_mesh(mesh_size, toroidal_extent,
        plasma_conditions=plasma_conditions, reaction_rate=reaction_rate)
    stellarator.export_source_mesh(filename=source_mesh_filename)
    strengths = stellarator.source_mesh.strengths

    dagmc_filename = Path("nwl_geom").with_suffix(".h5m")
    stellarator.build_pydagmc_model(
        magnet_exporter="cad_to_dagmc",
        max_mesh_size=60,
    )

    wall_surface = stellarator.invessel_build.dag_model.surfaces_by_id[1]
    group_id = 2
    group_map = {("boundary:Vacuum", group_id): [wall_surface]}
    stellarator.invessel_build.dag_model.add_groups(group_map)
    stellarator.export_pydagmc_model(dagmc_filename)

    num_parts = 1_000_000
    neutron_energy = 14.1e6 * 1.60218e-19 * 1e-6
    neutron_power = neutron_energy * np.sum(strengths)

    metadata = {
        "case": args.case,
        "nfp": int(nfp),
        "toroidal_extent_deg": float(toroidal_extent),
        "num_particles": int(num_parts),
        "wall_s": float(wall_s),
        "mesh_size": list(mesh_size),
        "num_toroidal_angles": int(num_toroidal_angles),
        "num_ribs": int(num_ribs),
        "num_rib_pts": int(num_rib_pts),
        "source_strength_sum": float(np.sum(strengths)),
        "neutron_power_mw": float(neutron_power),
    }
    Path("simulation_parameters.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    source_file = nwl_utils.fire_rays(
        dagmc_filename,
        source_mesh_filename,
        toroidal_extent,
        strengths,
        num_parts,
        random_seed=True,
    )
    nwl_mat, toroidal_bins, poloidal_bins, area_mat = nwl_utils.compute_nwl(
        source_file,
        str(vmec_file),
        wall_s,
        toroidal_extent,
        neutron_power,
        num_batches=4,
        num_threads=args.threads,
    )
    nwl_utils.plot_nwl(nwl_mat, toroidal_bins, poloidal_bins)
    np.save("nwl_mat.npy", nwl_mat)
    np.save("toroidal_bins.npy", toroidal_bins)
    np.save("poloidal_bins.npy", poloidal_bins)
    np.save("area_mat.npy", area_mat)


if __name__ == "__main__":
    main()

