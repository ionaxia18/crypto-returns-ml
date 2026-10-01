#!/bin/bash
# Ridge at minute strides 30/15/5/1 on the experiment split's dates, one data pass
# (experiment/checks/stride_test.py). Answers: what does thinning minutes cost APS?
#
# Submit from the repo root (CPU/memory/time are overridable on the command line):
#   mkdir -p logs results
#   sbatch scripts/stride_test.sbatch
#   sbatch -c 32 --mem 96G scripts/stride_test.sbatch --readers 8
#
# Extra arguments are passed through to `python -m experiment.checks.stride_test`.
# Memory: ~(readers + 1) x 1.3 GB of shards + 4 x 7.7 MB Gram matrices.
# The 2-week smoke run took ~6 s per training date at 16 CPUs, so 26 weeks
# should take ~20 min. A short
# --time starts sooner: SLURM backfills short jobs into idle gaps.
#SBATCH --job-name=stride_test
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# See run_linear.sbatch: the venv needs Sherlock's python/3.12.1 module.
module load devel python/3.12.1

set -euo pipefail

VENV="${VENV:-$SCRATCH/core980-work/.venv}"
DATA_ROOT="${DATA_ROOT:-/oak/stanford/projects/ruizhe-share/student_core980/data/core980_h30}"
OUT="${OUT:-results/stride_test.json}"

source "$VENV/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) cpus=$SLURM_CPUS_PER_TASK out=$OUT start=$(date -Is)"
python -m experiment.checks.stride_test --data "$DATA_ROOT" --out "$OUT" "$@"
echo "end=$(date -Is)"
