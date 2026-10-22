#!/bin/bash
# One refit anchor of an official rolling run (experiment/rolling.py run).
# Submitted as an array by scripts/submit_rolling.sh; task N = anchor N.
#   sbatch --array=1-24 scripts/rolling_part.sbatch MODEL OUT [rolling.py options...]
#SBATCH --job-name=rolling_part
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out

# See run_linear.sbatch: the venv needs Sherlock's python/3.12.1 module.
module load devel python/3.12.1

set -euo pipefail

MODEL="${1:?usage: rolling_part.sbatch MODEL OUT [rolling.py options...]}"
OUT="${2:?usage: rolling_part.sbatch MODEL OUT [rolling.py options...]}"
shift 2
VENV="${VENV:-$SCRATCH/core980-work/.venv}"
DATA_ROOT="${DATA_ROOT:-/oak/stanford/projects/ruizhe-share/student_core980/data/core980_h30}"

source "$VENV/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) model=$MODEL out=$OUT anchor=$SLURM_ARRAY_TASK_ID start=$(date -Is)"
python -m experiment.rolling run --data "$DATA_ROOT" --model "$MODEL" --out "$OUT" \
       --index "$SLURM_ARRAY_TASK_ID" "$@"
echo "end=$(date -Is)"
