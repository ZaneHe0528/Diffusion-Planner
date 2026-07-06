#!/usr/bin/env bash
set -euo pipefail

###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/media/ubuntu/T9/dataset/nuplan-v1.1/trainval" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/home/ubuntu/code/hezexiang/Diffusion-Planner/dataset/maps" # prefer a local writable copy over external drives

TRAIN_SET_PATH="/home/ubuntu/code/hezexiang/Diffusion-Planner/exp/trainval_cache" # preprocess training data output directory

PYTHON_BIN="/home/ubuntu/anaconda3/envs/dp/bin/python"
TOTAL_SCENARIOS=1000000
###################################

cd "$(dirname "$0")"
mkdir -p "$TRAIN_SET_PATH"

if [[ ! -w "$NUPLAN_MAP_PATH" ]]; then
  echo "ERROR: NUPLAN_MAP_PATH is not writable: $NUPLAN_MAP_PATH" >&2
  echo "nuPlan map.gpkg may fail to open in WAL mode from a read-only directory." >&2
  echo "Copy or mount the maps directory to a writable path, then update NUPLAN_MAP_PATH." >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

"$PYTHON_BIN" -X faulthandler -u -B data_process.py \
  --data_path "$NUPLAN_DATA_PATH" \
  --map_path "$NUPLAN_MAP_PATH" \
  --save_path "$TRAIN_SET_PATH" \
  --total_scenarios "$TOTAL_SCENARIOS"
