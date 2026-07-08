#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

###################################
# User Configuration Section
###################################
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-/home/ubuntu/code/hezexiang/Diffusion-Planner/nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/home/ubuntu/data/hezexiang/nuplan/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$SCRIPT_DIR/dataset/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/home/ubuntu/code/hezexiang/Diffusion-Planner/exp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/dp-pycache-stable}"
export PYTHONDONTWRITEBYTECODE=1

# ROS injects Python 3.10 packages and native libraries on this machine, which
# can crash the conda dp Python 3.9 process during scipy/torch/lightning imports.
unset PYTHONPATH
export LD_LIBRARY_PATH="${DP_LD_LIBRARY_PATH:-/home/ubuntu/anaconda3/envs/dp/lib:${CUDA_HOME:-/usr/local/cuda-11.8}/lib64}"
mkdir -p "$MPLCONFIGDIR"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "/home/ubuntu/anaconda3/envs/dp/bin/python" ]; then
        PYTHON_BIN="/home/ubuntu/anaconda3/envs/dp/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

if [ ! -d "$NUPLAN_MAPS_ROOT" ] && [ -d "$SCRIPT_DIR/dataset/maps" ]; then
    echo "NUPLAN_MAPS_ROOT not found, using repo maps: $SCRIPT_DIR/dataset/maps" >&2
    export NUPLAN_MAPS_ROOT="$SCRIPT_DIR/dataset/maps"
fi

SPLIT="${SPLIT:-val14}"
CHALLENGE="${CHALLENGE:-closed_loop_nonreactive_agents}"
SIM_WORKER="${SIM_WORKER:-ray_distributed}"
ENABLE_SIMULATION_PROGRESS_BAR="${ENABLE_SIMULATION_PROGRESS_BAR:-true}"
RAY_THREADS_PER_NODE="${RAY_THREADS_PER_NODE:-6}"
RAY_GPUS_PER_SIM="${RAY_GPUS_PER_SIM:-0.15}"
THREAD_POOL_MAX_WORKERS="${THREAD_POOL_MAX_WORKERS:-}"
DISABLE_METRIC_SUMMARY="${DISABLE_METRIC_SUMMARY:-true}"
CPU_AFFINITY="${CPU_AFFINITY:-0-7}"
LIMIT_TOTAL_SCENARIOS="${LIMIT_TOTAL_SCENARIOS:-}"
NUM_SCENARIOS_PER_TYPE="${NUM_SCENARIOS_PER_TYPE:-}"
EXTRA_SIM_ARGS="${EXTRA_SIM_ARGS:-}"
###################################

BRANCH_NAME=diffusion_planner_release
ARGS_FILE="$SCRIPT_DIR/checkpoints/args.json"
CKPT_FILE="$SCRIPT_DIR/checkpoints/model.pth"
GATE_CKPT="$SCRIPT_DIR/0hzxcode/gate_v2_output/runs/ego_nbr_perstep/best.pt"

if [ -z "${SCENARIO_BUILDER:-}" ]; then
    if [ "$SPLIT" == "val14" ]; then
        SCENARIO_BUILDER="nuplan"
    else
        SCENARIO_BUILDER="nuplan_challenge"
    fi
fi

FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"
PLANNER="${PLANNER:-diffusion_planner}"

TRAINVAL_DIR="$NUPLAN_DATA_ROOT/nuplan-v1.1/trainval"
if [ ! -f "$ARGS_FILE" ]; then
    echo "missing args file: $ARGS_FILE" >&2
    exit 2
fi
if [ ! -f "$CKPT_FILE" ]; then
    echo "missing planner checkpoint: $CKPT_FILE" >&2
    exit 2
fi
if [ ! -f "$GATE_CKPT" ]; then
    echo "missing gate checkpoint: $GATE_CKPT" >&2
    exit 2
fi
if [ ! -d "$NUPLAN_DEVKIT_ROOT/nuplan" ]; then
    echo "missing nuPlan devkit: $NUPLAN_DEVKIT_ROOT" >&2
    exit 2
fi
if [ ! -d "$NUPLAN_MAPS_ROOT" ]; then
    echo "missing nuPlan maps directory: $NUPLAN_MAPS_ROOT" >&2
    exit 2
fi
if [ ! -d "$TRAINVAL_DIR" ] || [ -z "$(find "$TRAINVAL_DIR" -maxdepth 1 -type f -name '*.db' -print -quit)" ]; then
    echo "missing nuPlan trainval db files under: $TRAINVAL_DIR" >&2
    exit 2
fi

# 清理上一轮诊断日志
rm -f "$SCRIPT_DIR/0hzxcode/gate_v2_output/val14_closedloop/frames.jsonl"

SIM_ARGS=(
    "+simulation=$CHALLENGE"
    "planner=$PLANNER"
    "planner.diffusion_planner.config.args_file=$ARGS_FILE"
    "planner.diffusion_planner.ckpt_path=$CKPT_FILE"
    "planner.diffusion_planner.gate_ckpt_path=$GATE_CKPT"
    "planner.diffusion_planner.enable_warmstart=true"
    "scenario_builder=$SCENARIO_BUILDER"
    "scenario_filter=$SPLIT"
    "experiment_uid=$PLANNER/${SPLIT}_gate_warmstart/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S")"
    "verbose=true"
    "worker=$SIM_WORKER"
    "enable_simulation_progress_bar=$ENABLE_SIMULATION_PROGRESS_BAR"
    "hydra.searchpath=[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
)

if [ "$SIM_WORKER" = "ray_distributed" ]; then
    SIM_ARGS+=(
        "worker.threads_per_node=$RAY_THREADS_PER_NODE"
        "distributed_mode=SINGLE_NODE"
        "number_of_gpus_allocated_per_simulation=$RAY_GPUS_PER_SIM"
    )
elif [ "$SIM_WORKER" = "single_machine_thread_pool" ] && [ -n "$THREAD_POOL_MAX_WORKERS" ]; then
    SIM_ARGS+=("worker.max_workers=$THREAD_POOL_MAX_WORKERS")
fi

if [ "$DISABLE_METRIC_SUMMARY" = "true" ]; then
    SIM_ARGS+=("~main_callback.metric_summary_callback")
fi

if [ -n "$LIMIT_TOTAL_SCENARIOS" ]; then
    SIM_ARGS+=("scenario_filter.limit_total_scenarios=$LIMIT_TOTAL_SCENARIOS")
fi

if [ -n "$NUM_SCENARIOS_PER_TYPE" ]; then
    SIM_ARGS+=("scenario_filter.num_scenarios_per_type=$NUM_SCENARIOS_PER_TYPE")
fi

if [ -n "$EXTRA_SIM_ARGS" ]; then
    read -r -a EXTRA_ARGS <<< "$EXTRA_SIM_ARGS"
    SIM_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [ -n "$CPU_AFFINITY" ] && command -v taskset >/dev/null 2>&1; then
    taskset -c "$CPU_AFFINITY" "$PYTHON_BIN" -B "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" "${SIM_ARGS[@]}"
else
    "$PYTHON_BIN" -B "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" "${SIM_ARGS[@]}"
fi
