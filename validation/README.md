# Validation data

`axisymmetric/`, `iter/` and `twisty/` contain the circular, shaped axisymmetric
and non-axisymmetric cases. The three saved Monte Carlo stacks each contain
100 independent maps of shape 101 × 101 from batches of one million source
particles. The aggregate files are the mean, sample standard deviation,
standard error of the mean and relative standard error. NWL is in MW/m² and
wall area is in m².

Monte Carlo maps are stored as (toroidal, poloidal); DESC maps as (poloidal,
toroidal). Comparison reverses the DESC poloidal axis and transposes the Monte
Carlo map. For the axisymmetric cases the DESC toroidal labels cover a full
turn while the Monte Carlo domain covers 90 degrees; symmetry permits the
column-wise comparison. The non-axisymmetric case uses its 90-degree field
period. `desc_global_metrics.json` stores the DESC power and area for that domain.

## Reprocess saved data

```sh
python validation/nwl_statistics.py validation/axisymmetric --output-dir outputs/axisymmetric-statistics
python validation/plot_validation_comparison.py --output-dir outputs/validation-comparison
python validation/scripts/render_geometry.py --output-dir outputs/validation-geometry
```

The statistics command reads `nwl_batches.npy` when available. Otherwise it
requires complete batch folders numbered `batch0` through `batch99`, including
matching grids and areas. Missing batches fail explicitly. Use
`--expected-batches` only when intentionally analyzing a different batch count.

The comparison command writes both the figure and a newly computed CSV table.
Its `--data-dir` option selects another directory containing all three cases.
The saved paper outputs are `validation_comparison.png`, the three geometry
PNGs in `figures/`, and `tables/comparison_summary.csv`. Additional saved
uncertainty diagnostics remain in `diagnostics/`.

Rendering the saved three-dimensional scenes requires Plotly and Kaleido,
with Chrome/Chromium available for static image export.

## Recalculate DESC maps

Run in the DESC environment. Each case is written to a new subdirectory, together
with the Monte Carlo aggregates used for comparison:

```sh
python validation/scripts/regenerate_desc_maps.py axisymmetric --output-dir outputs/desc-validation
python validation/scripts/regenerate_desc_maps.py iter --output-dir outputs/desc-validation
python validation/scripts/regenerate_desc_maps.py twisty --output-dir outputs/desc-validation
python validation/plot_validation_comparison.py --data-dir outputs/desc-validation --output-dir outputs/desc-comparison
```

Each map and its global power/area values are calculated together. The requested
wall offsets are 0.0218, 0.05255 and 0.0554 m respectively. The DESC source grid
has 20 radial, 32 poloidal and 101 toroidal points; its surface grid is 101 × 101.

## Repeat Monte Carlo batches

Repeating transport calculations requires ParaStell, OpenMC with DAGMC support,
MOAB, PyDAGMC and PyStell. ParaStell must support `build_pydagmc_model`, custom
source profiles and reaction rates, and `nwl_utils.compute_nwl`.

If `nwl_utils.fire_rays` does not accept `random_seed`, apply the included
[independent-seed patch](scripts/parastell-random-seed.patch) from the ParaStell
checkout with `patch -p1`, passing the patch file on standard input. The supplied
`scripts/fusion_source.py` provides the Bosch–Hale source model directly.

For each case, run one independent job per batch, numbered from zero:

```sh
python validation/scripts/run_monte_carlo.py twisty --output-dir runs/twisty/batch0 --threads 4
```

Repeat for `batch1` through `batch99`, then aggregate:

```sh
python validation/nwl_statistics.py runs/twisty --output-dir outputs/twisty-new
```

The transport script uses packaged VMEC files, the supplied Bosch–Hale source
model, a wall at normalized toroidal flux 1.05, and independent random seeds.
New stochastic results will differ within sampling uncertainty. The saved
geometry scenes can be rendered independently of the transport calculation;
their exported PNGs preserve the publication views.
