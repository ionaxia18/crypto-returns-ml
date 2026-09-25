#!/bin/bash
# Rolling run of the linear ridge baseline (models/ridge.py) on Sherlock.
#
# Submit from the repo root:
#   mkdir -p logs
#   sbatch scripts/run_linear.sbatch                                  # full 4w run
#   OUT=runs/linear_timing sbatch scripts/run_linear.sbatch --end 2022-07-10   # timing test
#
# Extra arguments are passed through to `f522kit run`. The run is resumable:
# if the job hits its time limit, resubmit the identical command.
#SBATCH --job-name=linear_4w
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

VENV="${VENV:-$SCRATCH/core980-work/venv}"   # edit if your venv lives elsewhere
DATA_ROOT="${DATA_ROOT:-/oak/stanford/projects/ruizhe-share/student_core980/data/core980_h30}"
OUT="${OUT:-runs/linear_4w}"

source "$VENV/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) out=$OUT start=$(date -Is)"
f522kit run --data "$DATA_ROOT" --model models/ridge.py:MyModel \
            --out "$OUT" --stride 4w --row-budget 1000000 --seed 0 "$@"
echo "end=$(date -Is)"
