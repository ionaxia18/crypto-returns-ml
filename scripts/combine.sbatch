#!/bin/bash
# Build combined models from saved predictions (experiment/combine.py) for one
# split, then they appear in the seed-aware table like any run.
#   sbatch scripts/combine.sbatch experiments/combos_v1.json                        # window 1
#   SPLIT=$SCRATCH/core980-work/experiment/v3_52w_jan24 sbatch scripts/combine.sbatch experiments/combos_v1.json
# Extra arguments go to experiment/combine.py (e.g. --force, --only NAME ...).
#SBATCH --job-name=combine
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

module load devel python/3.12.1

set -euo pipefail

SPEC="${1:?usage: combine.sbatch SPEC [combine.py options...]}"
shift
VENV="${VENV:-$SCRATCH/core980-work/.venv}"
DATA_ROOT="${DATA_ROOT:-/oak/stanford/projects/ruizhe-share/student_core980/data/core980_h30}"
SPLIT="${SPLIT:-$SCRATCH/core980-work/experiment/v2_52w}"

source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
python -m experiment.combine --split "$SPLIT" --data "$DATA_ROOT" --spec "$SPEC" "$@"
