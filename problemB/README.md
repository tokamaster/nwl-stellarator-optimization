# Problem B: boundary shape optimization

The paper compares `data/qs_control_moreiter/` (reference) with `data/stage_3/`
(NWL-aware). The current and kinetic profiles are fixed, while the LCFS
shape changes. `data/lineage_inputs/` contains the equilibria used to initialize
the continuation stages. `data/verification/diagnostic_arrays.npz` retains the
six-case common-grid arrays; the final pair uses the prefixes `qs-control__`
and `stage-3__`.

## Reproduce from scratch

From the package root, in the DESC environment:

```sh
sh problemB/scripts/reproduce.sh runs/B-from-scratch
```

[reproduce.sh](scripts/reproduce.sh) contains the full sequence of commands.
It builds its own reactor-scale initializer from DESC's `precise_QA` example.
The QS-only and constrained branches both start from that initializer; the
first capped run does not start from the QS-only endpoint. Each later stage
loads the newly saved result named in the input column below.

| Stage | Input | NWL cap [MW/m²] | Maximum iterations |
| --- | --- | ---: | ---: |
| `initialize` | `precise_QA`, finite-beta solve and reactor rescaling | — | 50 bootstrap iterations |
| `qs-control` | `initialize` | — | 1500 |
| `reference` | `qs-control` | — | 250 |
| `cap-0p880` | `initialize` | 0.880 | 1500 |
| `cap-0p865` | `cap-0p880` | 0.865 | 500 |
| `parent` | `cap-0p865` | 0.835 | 500 |
| `stage-1` | `parent` | 0.820 | 500 |
| `stage-2` | `stage-1` | 0.805 | 500 |
| `nwl-aware` | `stage-2` | 0.790 | 500 |

The initializer uses bootstrap `ftol=1e-2`, `gtol=1e-16` and no additional
force solve after bootstrap optimization. Its saved equilibrium is also the
archived baseline reference: that case performed no further optimization.
Problem A's reference uses different tolerances and is not substituted here.

All shape stages use `lsq-auglag`, `ftol=1e-5`, and a final force solve with
`ftol=1e-8`. Outer `gtol` is `1e-16`, except the final reference continuation
uses `1e-8`. Cap constraint weight is 10 throughout. Stage names select these
settings, including each cap value.

The output folder must be new. The compared final equilibria are in
`reference/equilibrium.h5` and `nwl-aware/equilibrium.h5` within it, with their
new diagnostics and figures alongside the stage folders and logs.

## Run individual stages

From the package root, in the DESC environment:

```sh
python problemB/scripts/optimize.py reference --output-dir runs/B-reference
python problemB/scripts/optimize.py nwl-aware --output-dir runs/B-nwl-aware
python problemB/scripts/optimize.py initialize --output-dir runs/B-initialize
python problemB/scripts/optimize.py qs-control --input-eq runs/B-initialize/equilibrium.h5 --output-dir runs/B-qs-control
```

Intermediate stages require `--input-eq`. The final `reference` and
`nwl-aware` commands retain their packaged checkpoint defaults when that
option is omitted. The full workflow always passes newly generated inputs.

## Evaluate and plot saved endpoints

```sh
python scripts/evaluate.py B problemB/data/qs_control_moreiter/qs_control_moreiter_eq.h5 --output-dir outputs/B-reference
python scripts/evaluate.py B problemB/data/stage_3/stage_3_eq.h5 --output-dir outputs/B-nwl-aware
python problemB/scripts/generate_paper_comparison_figures.py --reference-dir outputs/B-reference --nwl-aware-dir outputs/B-nwl-aware --output-dir outputs/B-evaluated-figures
```

`evaluate.py` copies the input equilibrium into its output folder alongside the
wall and diagnostic arrays. The plotter reads NWL and transform arrays from that
folder and calculates current from its equilibrium. To analyze a new optimization,
replace the equilibrium input above with `runs/B-reference/equilibrium.h5` or
`runs/B-nwl-aware/equilibrium.h5` and choose new output folders. Omitting the
folder options plots the saved paper cases.

The reference continues `original_qs_control_eq.h5` for up to 250 iterations.
The constrained case continues `stage_2_eq.h5` for up to 500 iterations with an
NWL bound of 0.790 MW/m² and constraint weight 10. Both optimize QS using
`lsq-auglag`, maintain mean transform target 0.42 and the initial aspect ratio,
and then perform a fixed-boundary force solve. Both saved outer optimizations
reached their iteration limits; their final force solves succeeded.

The diagnostic NWL grid is 40 × 40. QS uses a 10 × 25 × 25 grid including the
magnetic axis and plasma boundary. Force balance uses a 12 × 12 × 12 quadrature
resolution, and the transform statistics use 51 uniform radial points. QS and
force residuals are normalized through `ObjectiveFunction.compute_scaled`,
including its quadrature weights. The NWL 95th percentile is the first value at
which the weighted CDF reaches 0.95.

`results/campaign_b_metrics.csv` and `.json` retain the paper metrics, including
the area fraction above 0.80 MW/m². The final PNGs are in `figures/paper/`.
The figure generator draws the matching LCFS/wall line styles directly and
writes new figure metrics from the input arrays.
