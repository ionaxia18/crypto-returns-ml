#!/bin/bash
# Official rolling evaluation of one model, one parallel task per refit
# anchor (is window only; dev is never predicted), then merge + kit score.
#
#   scripts/submit_rolling.sh MODEL OUT [sbatch options...] [-- rolling.py options...]
#   scripts/submit_rolling.sh models/mlp_final.py:TwoLayer runs/mlp2_4w
#   scripts/submit_rolling.sh models/mlp_final.py:Linear   runs/mlplin_4w
#   scripts/submit_rolling.sh models/mlp_final.py:TwoLayer runs/mlp2_4w_rb15m -- --row-budget 15000000
#
# Default row budget 10M (rolling.py); report lands in OUT/report.json, with
# the paired daily-APS comparison against runs/linear_4w (BASELINE=...).
# MAX_PARALLEL=N limits concurrent anchors. Run from the repo root.
set -euo pipefail

MODEL="${1:?usage: submit_rolling.sh MODEL OUT [sbatch options...] [-- rolling.py options...]}"
OUT="${2:?usage: submit_rolling.sh MODEL OUT [sbatch options...] [-- rolling.py options...]}"
shift 2
SBATCH_ARGS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do SBATCH_ARGS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift

DATA_ROOT="${DATA_ROOT:-/oak/stanford/projects/ruizhe-share/student_core980/data/core980_h30}"
VENV="${VENV:-$SCRATCH/core980-work/.venv}"
# Counting anchors reads only the calendar (dataset.json + directory names).
module load devel python/3.12.1 >/dev/null 2>&1 || true
source "$VENV/bin/activate"
N=$(python -m experiment.rolling count --data "$DATA_ROOT")
[[ "$N" -gt 0 ]] || { echo "no anchors in the window" >&2; exit 1; }
ARRAY="1-$N${MAX_PARALLEL:+%$MAX_PARALLEL}"

mkdir -p logs
PART_ID=$(sbatch --parsable --array="$ARRAY" ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"} \
          scripts/rolling_part.sbatch "$MODEL" "$OUT" "$@")
MERGE_ID=$(sbatch --parsable --dependency="afterok:$PART_ID" scripts/rolling_merge.sbatch "$OUT")
echo "submitted $N anchors for $MODEL -> $OUT (array job $PART_ID); merge + score: job $MERGE_ID"
