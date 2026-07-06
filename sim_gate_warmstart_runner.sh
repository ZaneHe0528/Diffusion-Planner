export CUDA_VISIBLE_DEVICES=0,1
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
export NUPLAN_DEVKIT_ROOT="/home/ubuntu/code/hezexiang/Diffusion-Planner/nuplan-devkit"
export NUPLAN_DATA_ROOT="/home/ubuntu/data/hezexiang/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/ubuntu/data/hezexiang/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/home/ubuntu/code/hezexiang/Diffusion-Planner/exp"

SPLIT="val14"
CHALLENGE="closed_loop_nonreactive_agents"
###################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BRANCH_NAME=diffusion_planner_release
ARGS_FILE="$SCRIPT_DIR/checkpoints/args.json"
CKPT_FILE="$SCRIPT_DIR/checkpoints/model.pth"
GATE_CKPT="$SCRIPT_DIR/0hzxcode/gate_v2_output/runs/ego_nbr_perstep/best.pt"

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi

FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"
PLANNER=diffusion_planner

# 清理上一轮诊断日志
rm -f "$SCRIPT_DIR/0hzxcode/gate_v2_output/val14_closedloop/frames.jsonl"

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    planner.diffusion_planner.gate_ckpt_path=$GATE_CKPT \
    planner.diffusion_planner.enable_warmstart=true \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/${SPLIT}_gate_warmstart/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=6 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"
