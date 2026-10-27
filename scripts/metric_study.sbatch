#!/bin/bash
# Which smooth measure tracks APS? (experiment/metric_study.py) Reads saved val
# predictions of every run on one split; output results/<split>/metric_study.md
# + .json (small; commit them).
#   sbatch scripts/metric_study.sbatch                                   # window 1, cross-checked on window 2
#   SPLIT=$SCRATCH/core980-work/experiment/v3_52w_jan24 sbatch scripts/metric_study.sbatch --other results/v2_52w
# Extra options pass through, e.g. --models mlp
#SBATCH --job-name=metric_study
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

module load devel python/3.12.1

set -euo pipefail

VENV="${VENV:-$SCRATCH/core980-work/.venv}"
SPLIT="${SPLIT:-$SCRATCH/core980-work/experiment/v2_52w}"
if [[ $# -eq 0 ]]; then
  set -- --other results/v3_52w_jan24
fi

source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
python -m experiment.metric_study --split "$SPLIT" "$@"
