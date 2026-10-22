#!/bin/bash
# Merge the per-anchor parts of a rolling run and score it with the kit
# against the reproduced linear baseline, on the is window only.
#   sbatch scripts/rolling_merge.sbatch OUT
#SBATCH --job-name=rolling_merge
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

module load devel python/3.12.1

set -euo pipefail

OUT="${1:?usage: rolling_merge.sbatch OUT}"
VENV="${VENV:-$SCRATCH/core980-work/.venv}"
DATA_ROOT="${DATA_ROOT:-/oak/stanford/projects/ruizhe-share/student_core980/data/core980_h30}"
BASELINE="${BASELINE:-runs/linear_4w}"

source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
python -m experiment.rolling merge --data "$DATA_ROOT" --out "$OUT"
f522kit score --data "$DATA_ROOT" --preds "$OUT" --baseline "$BASELINE" \
       --window is,2022-07-01,2024-06-30 --out "$OUT/report.json"
