#!/bin/bash
# One experiment-split run per array task: task N runs entry N of an experiment
# list (experiment/run.py --list). Normally submitted via submit_experiment.sh,
# which sets --array from the list length:
#
#   scripts/submit_experiment.sh experiments/models_v1.txt
#   scripts/submit_experiment.sh experiments/models_v1.txt -c 32 --mem 96G
#
# Direct use: sbatch --array=1-3 scripts/experiment_run.sbatch LIST [run.py args...]
# e.g. `--in-ram` or `--suffix _c32`. Avoid --baseline here: the baseline
# run may not have finished; experiment/table.py pairs runs afterwards.
#SBATCH --job-name=experiment_run
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out

# See run_linear.sbatch: the venv needs Sherlock's python/3.12.1 module.
module load devel python/3.12.1

set -euo pipefail

LIST="${1:?usage: experiment_run.sbatch LIST [run.py args...]}"
shift
VENV="${VENV:-$SCRATCH/core980-work/.venv}"
DATA_ROOT="${DATA_ROOT:-/oak/stanford/projects/ruizhe-share/student_core980/data/core980_h30}"
SPLIT="${SPLIT:-$SCRATCH/core980-work/experiment/v2_52w}"
# Results go to results/<split dir name>/ unless OUT is set.
OUT_ARGS=()
[[ -n "${OUT:-}" ]] && OUT_ARGS=(--out "$OUT")

source "$VENV/bin/activate"
# Ridge parallelizes through BLAS; LightGBM/MLP take threads from
# SLURM_CPUS_PER_TASK inside experiment/run.py.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) cpus=$SLURM_CPUS_PER_TASK list=$LIST task=$SLURM_ARRAY_TASK_ID start=$(date -Is)"
python -m experiment.run --split "$SPLIT" --data "$DATA_ROOT" ${OUT_ARGS[@]+"${OUT_ARGS[@]}"} \
       --list "$LIST" --index "$SLURM_ARRAY_TASK_ID" "$@"
echo "end=$(date -Is)"
