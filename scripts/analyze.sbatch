#!/bin/bash
# Post-prediction analysis on one split (experiment/analyze.py). Output:
# results/<split>/analysis/ANALYSIS.md + analysis.json (small; commit them).
#   sbatch scripts/analyze.sbatch                                                     # window 1
#   SPLIT=$SCRATCH/core980-work/experiment/v3_52w_jan24 sbatch scripts/analyze.sbatch  # window 2
# Default models: ridge (reference), best LightGBM (3 seeds averaged), ridge + LightGBM.
# Pass your own: sbatch scripts/analyze.sbatch --model ridge=ridge_eqall --model mlp='h256x128_s*'
#SBATCH --job-name=analyze
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

module load devel python/3.12.1

set -euo pipefail

VENV="${VENV:-$SCRATCH/core980-work/.venv}"
DATA_ROOT="${DATA_ROOT:-/oak/stanford/projects/ruizhe-share/student_core980/data/core980_h30}"
SPLIT="${SPLIT:-$SCRATCH/core980-work/experiment/v2_52w}"
if [[ $# -eq 0 ]]; then
  set -- --model ridge=ridge_eqall --model "lgb=lgb_lr01_loss_s*" --model combo=cmb_ridge_lgb
fi

source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
python -m experiment.analyze --split "$SPLIT" --data "$DATA_ROOT" "$@"
