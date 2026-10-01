#!/bin/bash
# Build the experiment split (experiment/split.py) once.
#
# Submit from the repo root:
#   mkdir -p logs
#   sbatch scripts/experiment_build.sbatch
#   SPLIT=$SCRATCH/core980-work/experiment/s5 sbatch scripts/experiment_build.sbatch --minute-stride 5
#
# Extra arguments are passed through to `python -m experiment.split`.
# Disk: ~45 GB train (~11.4M rows) + ~35 GB val at the defaults (26 weeks, stride 5).
# --workers defaults to --cpus-per-task; each worker holds <= ~130 MB.
#SBATCH --job-name=experiment_build
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

# See run_linear.sbatch: the venv needs Sherlock's python/3.12.1 module.
module load devel python/3.12.1

set -euo pipefail

VENV="${VENV:-$SCRATCH/core980-work/.venv}"
DATA_ROOT="${DATA_ROOT:-/oak/stanford/projects/ruizhe-share/student_core980/data/core980_h30}"
SPLIT="${SPLIT:-$SCRATCH/core980-work/experiment/v1}"

source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) cpus=$SLURM_CPUS_PER_TASK split=$SPLIT start=$(date -Is)"
python -m experiment.split --data "$DATA_ROOT" --out "$SPLIT" "$@"
echo "end=$(date -Is)"
