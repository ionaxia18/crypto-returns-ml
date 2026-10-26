#!/bin/bash
# Combine full rolling runs and score over the is window (experiment/combine_rolling.py).
#   sbatch scripts/rolling_combine.sbatch runs/cmb_ridge_lgb_4w --part ridge=runs/linear_4w:1 --part lgb=runs/lgbm_4w:1 --norm rms
#SBATCH --job-name=rolling_combine
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

module load devel python/3.12.1

set -euo pipefail

OUT="${1:?usage: rolling_combine.sbatch OUT --part alias=run[:weight] ... [--norm rms|none]}"
shift
VENV="${VENV:-$SCRATCH/core980-work/.venv}"
DATA_ROOT="${DATA_ROOT:-/oak/stanford/projects/ruizhe-share/student_core980/data/core980_h30}"

source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
python -m experiment.combine_rolling --data "$DATA_ROOT" --out "$OUT" "$@"
