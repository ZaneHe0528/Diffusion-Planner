#!/usr/bin/env bash
set -euo pipefail

###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/media/ubuntu/T9/dataset/nuplan-v1.1/trainval"
NUPLAN_MAP_PATH="/home/ubuntu/code/hezexiang/Diffusion-Planner/dataset/maps"
NUPLAN_DB_FILES=""

PYTHON_BIN="/home/ubuntu/anaconda3/envs/dp/bin/python"

# 0 means no total-scenario limit. Use a small number first if you only want a smoke test.
TOTAL_SCENARIOS=0
MAX_FRAMES_PER_SCENARIO=""
NUM_WORKERS=8
LIMITED_DB_FILES=16

OUTPUT_ROOT="/home/ubuntu/code/hezexiang/Diffusion-Planner/exp/gate_trainval_cache"
SAVE_PATH="${OUTPUT_ROOT}/npz"
DATA_LIST="${OUTPUT_ROOT}/diffusion_planner_gate.json"
METADATA_CSV="${OUTPUT_ROOT}/diffusion_planner_gate_metadata.csv"
SUMMARY_JSON="${OUTPUT_ROOT}/diffusion_planner_gate_summary.json"
###################################

cd "$(dirname "$0")"
mkdir -p "$SAVE_PATH"

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

CMD=(
  "$PYTHON_BIN" -X faulthandler -u -B data_process_gate.py
  --data_path "$NUPLAN_DATA_PATH"
  --map_path "$NUPLAN_MAP_PATH"
  --save_path "$SAVE_PATH"
  --data_list "$DATA_LIST"
  --metadata_csv "$METADATA_CSV"
  --summary_json "$SUMMARY_JSON"
  --total_scenarios "$TOTAL_SCENARIOS"
  --num_workers "$NUM_WORKERS"
  --limited_db_files "$LIMITED_DB_FILES"
  --scenario_subsample_ratio 0.5
  --frame_stride 1
  --no_future_gt
)

if [[ -n "$MAX_FRAMES_PER_SCENARIO" ]]; then
  CMD+=(--max_frames_per_scenario "$MAX_FRAMES_PER_SCENARIO")
fi

if [[ -n "$NUPLAN_DB_FILES" ]]; then
  CMD+=(--db_files "$NUPLAN_DB_FILES")
fi

"${CMD[@]}"
