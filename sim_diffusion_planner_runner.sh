export CUDA_VISIBLE_DEVICES=0,1
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
# Set environment variables
# export NUPLAN_DEVKIT_ROOT="REPLACE_WITH_NUPLAN_DEVIKIT_DIR"  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
# export NUPLAN_DATA_ROOT="REPLACE_WITH_DATA_DIR"  # nuplan dataset absolute path (e.g. "/data")
# export NUPLAN_MAPS_ROOT="REPLACE_WITH_MAPS_DIR" # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
# export NUPLAN_EXP_ROOT="REPLACE_WITH_EXP_DIR" # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

export NUPLAN_DEVKIT_ROOT="/home/ubuntu/code/hezexiang/Diffusion-Planner/nuplan-devkit"
export NUPLAN_DATA_ROOT="/home/ubuntu/data/hezexiang/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/ubuntu/data/hezexiang/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/home/ubuntu/code/hezexiang/Diffusion-Planner/exp"


# Dataset split to use  选择场景集
# Options: 
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="val14"  # e.g., "val14"

# Challenge type。 NR 和 R 是仿真模式，三个场景集都有NR和R
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="closed_loop_nonreactive_agents"  # e.g., "closed_loop_nonreactive_agents"
###################################


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BRANCH_NAME=diffusion_planner_release
ARGS_FILE="$SCRIPT_DIR/checkpoints/args.json"
CKPT_FILE="$SCRIPT_DIR/checkpoints/model.pth"

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    # worker.threads_per_node=128 \ 
    worker.threads_per_node=6 \     
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"