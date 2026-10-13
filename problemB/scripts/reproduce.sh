#!/bin/sh
# Rebuild both Problem B branches, then evaluate and plot the new endpoints.
set -eu
if [ "$#" -ne 1 ]; then
    printf 'Usage: sh problemB/scripts/reproduce.sh NEW_OUTPUT_DIRECTORY\n' >&2
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
    printf 'Problem B: %s\n' "$stage"
    "$python_cmd" "$repo_dir/problemB/scripts/optimize.py" "$stage" \
        --output-dir "$output_dir/$stage" "$@" > "$output_dir/$stage.log" 2>&1
}

run_stage initialize
run_stage qs-control --input-eq "$output_dir/initialize/equilibrium.h5"
run_stage reference --input-eq "$output_dir/qs-control/equilibrium.h5"
# The cap branch starts from the initializer, independently of the QS branch.
run_stage cap-0p880 --input-eq "$output_dir/initialize/equilibrium.h5"
run_stage cap-0p865 --input-eq "$output_dir/cap-0p880/equilibrium.h5"
run_stage parent --input-eq "$output_dir/cap-0p865/equilibrium.h5"
run_stage stage-1 --input-eq "$output_dir/parent/equilibrium.h5"
run_stage stage-2 --input-eq "$output_dir/stage-1/equilibrium.h5"
run_stage nwl-aware --input-eq "$output_dir/stage-2/equilibrium.h5"

for case_name in reference nwl-aware; do
    "$python_cmd" "$repo_dir/scripts/evaluate.py" B "$output_dir/$case_name/equilibrium.h5" \
        --output-dir "$output_dir/$case_name-diagnostics" > "$output_dir/$case_name-diagnostics.log" 2>&1
done
"$python_cmd" "$repo_dir/problemB/scripts/generate_paper_comparison_figures.py" \
    --reference-dir "$output_dir/reference-diagnostics" --nwl-aware-dir "$output_dir/nwl-aware-diagnostics" \
    --output-dir "$output_dir/figures" > "$output_dir/figures.log" 2>&1
printf 'Results: %s\n' "$output_dir"
