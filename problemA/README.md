# Problem A: fixed boundary

`data/reference/` contains the reference equilibrium (`baseline_*`).
`data/aggressive/` contains the selected NWL-aware equilibrium (`peaking_*`).
Each includes a saved equilibrium, fitted offset wall, NWL values, coordinates,
wall-area weights and scalar metrics. `data/verification/` retains the fine
near-axis profile arrays and their numerical summary. `results/` contains the
paper metrics, bootstrap residuals, optimization settings and solver status.

The reference minimizes bootstrap-current inconsistency with the boundary and
kinetic profiles fixed. The NWL-aware case adds a smooth peaking objective
(weight 2, sharpness 40) and quasisymmetry (weight 110). The current profile scale
and its first two underlying coefficients remain fixed. The packaged checkpoint
for final-stage-only reruns is `data/inputs/peaking_start_eq.h5`.

## Reproduce from scratch

From the package root, in the DESC environment:

```sh
sh problemA/scripts/reproduce.sh runs/A-from-scratch
```

[reproduce.sh](scripts/reproduce.sh) contains the full sequence of commands.
It constructs the reference from DESC's `precise_QA` example, then passes each
newly saved equilibrium to the next stage. The output folder must be new.
Final equilibria are in `reference/equilibrium.h5` and `nwl-aware/equilibrium.h5`
inside that folder; their diagnostics and figures are generated there too.

| Stage | Input | Peaking weight | QS weight | Sharpness | Maximum iterations |
| --- | --- | ---: | ---: | ---: | ---: |
| `reference` | `precise_QA`, finite-beta solve and reactor rescaling | 0 | 0 | — | 50 |
| `peaking-0p5` | `reference` | 0.5 | 0 | 20 | 60 |
| `peaking-1p0` | `peaking-0p5` | 1.0 | 0 | 20 | 60 |
| `peaking-1p25` | `peaking-1p0` | 1.25 | 0 | 20 | 60 |
| `qs-100` | `peaking-1p25` | 2 | 100 | 20 | 120 |
| `qs-110` | `qs-100` | 2 | 110 | 20 | 120 |
| `nwl-aware` | `qs-110` | 2 | 110 | 40 | 120 |

The bootstrap term has weight 1 throughout. Reference optimization uses
`ftol=1e-6`, `gtol=1e-16`, then a force solve with `ftol=1e-3`.
The first four continuation stages use outer tolerances
`ftol=1e-7`, `gtol=xtol=1e-8`, `ctol=1e-6`, inner force tolerances `1e-8`
(100 iterations), and a final force solve at `1e-10` (200 iterations).
The last two use outer tolerances `ftol=1e-9`, `gtol=xtol=1e-10`, `ctol=1e-8`,
inner force tolerances `1e-10` (200 iterations), and a final SVD force solve
at `1e-12` (400 iterations). These choices are built into the stage names.

## Run individual stages

From the package root, in the DESC environment:

```sh
python problemA/scripts/optimize.py reference --output-dir runs/A-reference
python problemA/scripts/optimize.py nwl-aware --output-dir runs/A-nwl-aware
python problemA/scripts/optimize.py peaking-0p5 --input-eq runs/A-reference/equilibrium.h5 --output-dir runs/A-peaking-0p5
```

Intermediate stages require `--input-eq` pointing to the preceding stage's
`equilibrium.h5`. The final `nwl-aware` command defaults to the packaged
`peaking_start_eq.h5` when that option is omitted. The full workflow always
passes the newly generated input explicitly.

## Evaluate and plot saved endpoints

```sh
python scripts/evaluate.py A problemA/data/reference/baseline_eq.h5 --output-dir outputs/A-reference
python scripts/evaluate.py A problemA/data/aggressive/peaking_eq.h5 --output-dir outputs/A-nwl-aware
python problemA/scripts/generate_comparison_figures.py --reference-dir outputs/A-reference --nwl-aware-dir outputs/A-nwl-aware --output-dir outputs/A-evaluated-figures
```

`optimize.py` saves the initial state, outer-optimizer endpoint, final
force-restored equilibrium and fitted wall. `evaluate.py` reloads a saved
endpoint without modifying it and copies it into the output folder alongside
the wall and diagnostic arrays. To analyze a new optimization, replace the
equilibrium input above with `runs/A-reference/equilibrium.h5` or
`runs/A-nwl-aware/equilibrium.h5` and choose new output folders.
The diagnostics use a 30 × 30 NWL grid, a
10 × 25 × 25 QS grid with radial points through 0.9999, the equilibrium's default
force grid, and 101 uniform radial points for the reported transform statistics.
The area-weighted 95th percentile uses linear interpolation of the weighted CDF.

The paper's three final plots are in `figures/`. The figure script accepts
complete evaluation folders as shown above; its defaults use the saved paper
data. It reads NWL arrays from the selected folder and calculates profiles from
that folder's equilibrium. Bootstrap consistency is a separate diagnostic: it is
substantially worse in the saved NWL-aware case, as reported in
`results/bootstrap_metrics.json`.
