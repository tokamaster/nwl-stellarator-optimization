#!/bin/sh
# Rebuild both Problem A cases, then evaluate and plot the new endpoints.
set -eu
if [ "$#" -ne 1 ]; then
    printf 'Usage: sh problemA/scripts/reproduce.sh NEW_OUTPUT_DIRECTORY\n' >&2
    exit 2
fi
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p -- "$(dirname -- "$1")"
mkdir -- "$1"
output_dir=$(CDPATH= cd -- "$1" && pwd)
python_cmd=${PYTHON:-python}
export PYTHONNOUSERSITE=1

run_stage() {
    stage=$1
    shift
    printf 'Problem A: %s\n' "$stage"
    "$python_cmd" "$repo_dir/problemA/scripts/optimize.py" "$stage" \
        --output-dir "$output_dir/$stage" "$@" > "$output_dir/$stage.log" 2>&1
}

run_stage reference
run_stage peaking-0p5 --input-eq "$output_dir/reference/equilibrium.h5"
run_stage peaking-1p0 --input-eq "$output_dir/peaking-0p5/equilibrium.h5"
run_stage peaking-1p25 --input-eq "$output_dir/peaking-1p0/equilibrium.h5"
run_stage qs-100 --input-eq "$output_dir/peaking-1p25/equilibrium.h5"
run_stage qs-110 --input-eq "$output_dir/qs-100/equilibrium.h5"
run_stage nwl-aware --input-eq "$output_dir/qs-110/equilibrium.h5"

for case_name in reference nwl-aware; do
    "$python_cmd" "$repo_dir/scripts/evaluate.py" A "$output_dir/$case_name/equilibrium.h5" \
        --output-dir "$output_dir/$case_name-diagnostics" > "$output_dir/$case_name-diagnostics.log" 2>&1
done
"$python_cmd" "$repo_dir/problemA/scripts/generate_comparison_figures.py" \
    --reference-dir "$output_dir/reference-diagnostics" --nwl-aware-dir "$output_dir/nwl-aware-diagnostics" \
    --output-dir "$output_dir/figures" > "$output_dir/figures.log" 2>&1
printf 'Results: %s\n' "$output_dir"
