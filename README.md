# Neutron wall loading in stellarator optimization

Data and scripts for the paper's validation and optimization examples. The saved
numerical results and publication figures are included; reproducing the plots
does not require repeating an optimization or a Monte Carlo simulation.

| Directory | Contents |
| --- | --- |
| `method/` | Visibility schematic source and its two background images |
| `validation/` | Three equilibria, Monte Carlo batches, DESC maps, comparison tables and figures |
| `problemA/` | Fixed-boundary reference and NWL-aware equilibria, diagnostic arrays and figures |
| `problemB/` | Shape-optimized reference and NWL-aware equilibria, continuation inputs, diagnostics and figures |
| `scripts/` | Reactor reference construction, saved-data summary and equilibrium diagnostics |

Use an existing DESC installation with `NeutronWallLoading` and
`constant_offset_surface` support, plus NumPy and Matplotlib. DESC scripts
default to a GPU; set `DESC_DEVICE=cpu` to use a CPU installation.

Run commands below from this directory. Every script that writes results requires
a new output directory and refuses to overwrite it. Input defaults are relative
to the script location, so the package can be placed anywhere.

## Read and summarize the saved data

With NumPy and Matplotlib installed:

```sh
python scripts/summarize.py --output-dir outputs/summary
python validation/plot_validation_comparison.py --output-dir outputs/validation
python validation/nwl_statistics.py validation/twisty --output-dir outputs/twisty-statistics
python -m unittest discover -s tests -v
```

Figure-input tests also run when DESC is installed; otherwise they are skipped.

The summary reports wall-area-weighted averages and fractions. The stored and
recomputed `area_fraction_above_0p80` values are fractions; multiply by 100 for the
paper's percentage column. The threshold is strictly greater than 0.80 MW/m².

## Regenerate the optimization figures

These commands also require DESC with its native neutron-wall-loading objective:

```sh
python problemA/scripts/generate_comparison_figures.py --output-dir outputs/problemA-figures
python problemB/scripts/generate_paper_comparison_figures.py --output-dir outputs/problemB-figures
```

The included PNGs are the publication images. The scripts generate the same
scientific panels, with toroidal coordinates labelled ζ and matching LCFS/wall
line styles in the Problem B boundary figure. Rendering may vary with plotting
library versions; the original exported images are retained.

To plot newly evaluated cases, pass `--reference-dir` and `--nwl-aware-dir`
to either figure script. Each folder must be a complete output from
`scripts/evaluate.py`, containing `equilibrium.h5`, `offset_wall.h5`, and
`diagnostics.npz`. Omitting a folder uses the saved paper case for that side.
Individual equilibrium and array overrides are replaced by these folder options
so each plotted case uses one set of inputs. The Problem A and B instructions
include complete examples.

For calculations from equilibria, see [Problem A](problemA/README.md),
[Problem B](problemB/README.md), and [validation](validation/README.md).

## Reproduce the optimizations from scratch

In an environment with the required DESC features:

```sh
sh problemA/scripts/reproduce.sh runs/A-from-scratch
sh problemB/scripts/reproduce.sh runs/B-from-scratch
```

Each command starts from DESC's `precise_QA` example, runs the complete selected
optimization chain, evaluates its new reference and NWL-aware endpoints, and
plots them. No packaged equilibrium is used as a starting checkpoint. Every
stage has its own output folder and log; the command stops if a stage fails.
The top-level output folder must be new. Set `PYTHON` to a Python executable
if it is not named `python`.

The problem READMEs list the stage order, settings and commands for running
individual stages. The saved paper data remain available for comparison.
These workflows replay the archived optimization settings; numerical paths
can vary with DESC/JAX versions and hardware, so compare the newly evaluated
metrics with the saved results.

## Numerical interpretation

Problem A uses a 30 × 30 NWL surface grid; Problem B uses a 40 × 40 diagnostic
grid, following a 30 × 30 optimization grid. Both use the native DESC volume
quadrature and a requested wall offset of 0.15 m. The fitted wall's sampled
distance from the plasma can differ from the requested offset; saved metrics
report those distances.

Problem B's final optimization used a nominal 0.790 MW/m² cap. Its saved endpoint
has a diagnostic peak of approximately 0.7983 MW/m²: below the paper's reporting
threshold of 0.80 MW/m², but above the nominal optimization cap. Solver status
and equilibrium-quality metrics are retained in `results/`; the saved endpoints
do not imply convergence of every outer optimization.
