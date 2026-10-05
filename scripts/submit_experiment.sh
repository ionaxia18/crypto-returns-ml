#!/bin/bash
# Submit an experiment list: build the split first if needed, then run every
# list entry as one SLURM array job.
#
#   scripts/submit_experiment.sh LIST [sbatch options...] [-- run.py options...]
#
#   scripts/submit_experiment.sh experiments/models_v1.txt
#   scripts/submit_experiment.sh experiments/models_v1.txt -c 32 --mem 96G
#   scripts/submit_experiment.sh experiments/models_v1.txt -- --in-ram   # run.py options after --
#
# - If $SPLIT has no meta.json, experiment_build.sbatch is submitted first and
#   the runs start only after it succeeds (--dependency=afterok). An existing
#   split is reused as is. sbatch options here apply to the runs, not the build.
# - CPU benchmark: submit the same list at several -c values with a suffix, e.g.
#     for c in 8 16 32; do scripts/submit_experiment.sh LIST -c $c -- --suffix _c$c; done
# - MAX_PARALLEL=N limits how many array tasks run at once (default: all).
# - SPLIT (default $SCRATCH/core980-work/experiment/v2_52w) is exported to both jobs.
#   Results go to results/<split dir name>/ (e.g. results/v2_52w/).
# - Several lists can be submitted back to back before the split exists: the
#   pending build job id is kept in $SPLIT.build_job and later submissions
#   wait on that same build instead of starting another.
#
# Run from the repo root. Needs no venv: entries are counted here the same
# way experiment/run.py reads them (text after # ignored, blank lines skipped).
set -euo pipefail

LIST="${1:?usage: submit_experiment.sh LIST [sbatch options...] [-- run.py options...]}"
shift
[[ -f "$LIST" ]] || { echo "no such list: $LIST" >&2; exit 1; }

SBATCH_ARGS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do SBATCH_ARGS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift

N=$(sed 's/#.*//' "$LIST" | grep -c '[^[:space:]]' || true)
[[ "$N" -gt 0 ]] || { echo "$LIST has no entries" >&2; exit 1; }
ARRAY="1-$N${MAX_PARALLEL:+%$MAX_PARALLEL}"

export SPLIT="${SPLIT:-$SCRATCH/core980-work/experiment/v2_52w}"
MARKER="$SPLIT.build_job"
mkdir -p logs

DEPEND=()
PENDING_ID=""
if [[ -f "$MARKER" ]]; then
  PENDING_ID=$(cat "$MARKER")
  squeue -h -j "$PENDING_ID" 2>/dev/null | grep -q . || PENDING_ID=""  # no longer queued
fi
if [[ -f "$SPLIT/meta.json" ]]; then
  echo "using existing split $SPLIT"
  rm -f "$MARKER"
elif [[ -n "$PENDING_ID" ]]; then
  echo "split $SPLIT is being built by job $PENDING_ID: runs will wait for it"
  DEPEND=(--dependency="afterok:$PENDING_ID")
elif [[ -e "$SPLIT" ]]; then
  echo "$SPLIT exists without meta.json and no build is queued: the build crashed." >&2
  echo "Check logs/experiment_build-*.out; remove $SPLIT and $MARKER to rebuild." >&2
  exit 1
else
  BUILD_ID=$(sbatch --parsable scripts/experiment_build.sbatch)
  mkdir -p "$(dirname "$SPLIT")" && echo "$BUILD_ID" > "$MARKER"
  echo "no split at $SPLIT: submitted build job $BUILD_ID"
  DEPEND=(--dependency="afterok:$BUILD_ID")
fi

echo "submitting $N runs from $LIST (array $ARRAY)"
sbatch --array="$ARRAY" ${DEPEND[@]+"${DEPEND[@]}"} ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"} \
       scripts/experiment_run.sbatch "$LIST" "$@"
