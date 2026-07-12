# val14 gate warm-start 实验设置

本文记录 2026-07-08 跑通的 Diffusion-Planner + `gate_v2` warm-start val14 闭环评测。最终结果见 `0实验结果/DP-val14-gate-warmstart.md`。

## 实验目标

在原始 Diffusion-Planner 的闭环推理中加入轻量 gate，对相邻规划帧的轨迹变化量 `d_hat` 做预测，并根据预测难度复用上一帧扩散去噪结果，从而减少 DPM-Solver 去噪步数。实验比较：

- nuPlan `val14` / `closed_loop_nonreactive_agents` 总分变化。
- decoder / NFE 层面的推理加速。
- planner `compute_trajectory` 口径的端到端推理耗时变化。
- 主动判难和被动回退触发比例。

## 代码实现手段

### Planner 串线

入口文件：`diffusion_planner/planner/planner.py`

- 在 planner 中加载原始 Diffusion-Planner checkpoint：`checkpoints/model.pth`。
- 在 planner 中加载 gate checkpoint：`0hzxcode/gate_v2_output/runs/ego_nbr_perstep/best.pt`。
- gate 使用 raw observation 输入；Diffusion-Planner 主模型仍使用原有 normalized feature 输入，避免 gate 特征处理污染主模型输入分布。
- planner 每帧调用 gate 预测 `d_hat_m`，将其转换为 warm-start 操作参数 `level / t_start / steps`。
- 对每帧写出诊断日志到 `0hzxcode/gate_v2_output/val14_closedloop/frames.jsonl`，字段包括 `decoder_ms`、`nfe`、`d_hat_m`、`level`、`t_start`、`steps`、`forced_full`、`cache_miss`、`passive_fallback`、`d_meas_m`、`epsilon_m` 等。
- 增加 planner model cache，避免 nuPlan 构建 1118 个 scenario runner 时重复加载 Diffusion-Planner checkpoint。

### Gate 推理

相关文件：

- `0hzxcode/gate_v2/gate_model.py`
- `0hzxcode/gate_v2/inference.py`
- `0hzxcode/gate_v2/d_to_ops.py`
- `0hzxcode/gate_v2/safety.py`

实现要点：

- gate 模型输入组：`ego_history`、`neighbor_agents`。
- gate 输出 `d_hat_m`，含义是相邻规划帧轨迹变化量的预测值，本实验使用 checkpoint 中的 `d_column=perstep_max_m`。
- `model_inputs_to_gate_batch` 保持 planner 侧 batch shape，不再额外 `unsqueeze(0)` 造成邻车张量维度错误。
- `load_gate` 增加 checkpoint cache，避免每个 scenario 重复加载 gate 权重。
- `score_threshold_m` 只作为 `score_alarm` 诊断，不直接强制 full denoise；只有 `hard_threshold_m` 触发主动 full fallback。

### Warm-start 缓存与轨迹复用

相关文件：

- `0hzxcode/gate_v2/warmstart.py`
- `diffusion_planner/model/module/decoder.py`
- `diffusion_planner/model/diffusion_utils/dpm_solver_pytorch.py`

实现要点：

- 第一个 planning frame 无缓存，强制完整去噪，并要求 decoder 返回 `x0_norm` 作为后续帧缓存。
- ego 轨迹复用时先将缓存的 normalized `x0_norm` 反归一化到物理坐标，做 ego-frame shift 后再归一化，避免在 normalized 空间直接平移造成单位错误。
- 邻车轨迹按 nuPlan track token 对齐；token 未命中时用当前观测做匀速外推 fallback。
- DPM-Solver 支持从 `t_start` 开始采样，支持 `first_model_output` 注入，并通过 `nfe_counter` 统计实际模型调用次数。
- decoder 在 warm-start 路径先做一次验证前向，得到 `first_model_output` 和被动验证距离 `d_meas_m`。
- 若 `d_meas_m <= hard_threshold_m`，复用 warm-start 并注入首个模型输出，避免重复计算第一步。
- 若 `d_meas_m > hard_threshold_m`，触发 passive fallback，执行完整去噪。
- NFE 统计包含 `denoise_to_zero`。

### 安全兜底

相关文件：`0hzxcode/gate_v2/safety.py`

- 主动判难：当 `d_hat_m >= hard_threshold_m` 时，直接使用 full denoise。
- 分数告警：当 `d_hat_m >= score_threshold_m` 时仅记录 `score_alarm=true`，不强制 full denoise。
- 邻车交互保守上调：近距离且速度较高的邻车比例较高时，将 warm-start level 上调。
- 被动回退：decoder 端实际验证 `d_meas_m`，超过 `hard_threshold_m` 时 full fallback。

### 数据层改动

相关文件：

- `diffusion_planner/data_process/data_processor.py`
- `diffusion_planner/data_process/agent_process.py`
- `data_process_gate.py`

实现要点：

- `observation_adapter(..., return_neighbor_tokens=True)` 返回与邻车 slot 对齐的 track token。
- 修正 future tracked objects 的返回值解包。
- 为避免本机 native numpy / map 几何路径偶发崩溃，`agent_process.py` 与 `roadblock_utils.py` 中部分小数组逻辑改成 Python list / scalar 路径。

### 运行稳定性处理

相关文件：`sim_gate_warmstart_runner.sh`

- 使用 `PYTHON_BIN=/home/ubuntu/anaconda3/envs/dp/bin/python`。
- `unset PYTHONPATH`，避免 ROS 注入 Python 3.10 包和 native library。
- 设置 `LD_LIBRARY_PATH=/home/ubuntu/anaconda3/envs/dp/lib:${CUDA_HOME:-/usr/local/cuda-11.8}/lib64`。
- 设置 `MPLBACKEND=Agg`、`MPLCONFIGDIR=/tmp/matplotlib-cache`。
- 设置 `PYTHONPYCACHEPREFIX=/tmp/dp-pycache-stable`、`PYTHONDONTWRITEBYTECODE=1`。
- 默认关闭 `main_callback.metric_summary_callback`，避免 matplotlib/fontTools 相关问题。
- 全量最终 run 使用 `worker=sequential` 和 `CPU_AFFINITY=0`。原因：`single_machine_thread_pool` 在本机触发 native FPE，顺序模式最稳定。
- 本机 `nuplan-devkit` 做了本地稳定性 patch：本地 nuboard 写入改同步写，`run_simulation.py` / `utils_config.py` 去掉对完整 `pytorch_lightning` 的 eager import 依赖。

## Gate checkpoint 参数

checkpoint：`0hzxcode/gate_v2_output/runs/ego_nbr_perstep/best.pt`

训练参数来自 checkpoint metadata：

| 参数 | 值 |
| --- | --- |
| `d_column` | `perstep_max_m` |
| `dataset_dir` | `0hzxcode/gate_output/gate_dataset_chunks` |
| `adjacent_csv` | `0hzxcode/adjacent_traj_l2_output/per_pair_overlap_l2.csv` |
| `seed` | 3407 |
| `batch_size` | 512 |
| `epochs` | 40 |
| `lr` | 0.001 |
| `weight_decay` | 0.0001 |
| `dropout` | 0.1 |
| `group_dropout` | 0.15 |
| `val_ratio` | 0.15 |
| `test_ratio` | 0.15 |
| `hard_quantile` | 0.9 |
| `level_quantiles` | `[0.5, 0.75, 0.9]` |

模型结构来自 checkpoint config：

| 参数 | 值 |
| --- | --- |
| enabled groups | `ego_history`, `neighbor_agents` |
| `embed_dim` | 64 |
| `token_hidden` | 128 |
| `trunk_hidden` | 128 |
| `num_levels` | 4 |
| `dropout` | 0.1 |
| `group_dropout` | 0.15 |

阈值：

| 参数 | 值 |
| --- | ---: |
| `level_edges_m[0]` | 0.5246983767 |
| `level_edges_m[1]` | 1.3284558058 |
| `level_edges_m[2]` | 2.6504895687 |
| `score_threshold_m` | 0.1856259704 |
| `hard_threshold_m` | 2.6504895687 |

## Warm-start 参数

`d_hat_m` 先按 `level_edges_m` 映射到 4 个 level。默认 level 操作表来自 `0hzxcode/gate_v2/d_to_ops.py`：

| level | 默认 `t_start` 上限比例 | steps |
| ---: | ---: | ---: |
| 0 | 0.12 | 2 |
| 1 | 0.35 | 4 |
| 2 | 0.65 | 7 |
| 3 | 1.00 | 10 |

实际 level 0-2 使用 continuous `t_start`：

- 将 `d_hat_m / xy_std_m` 转成目标 sigma。
- `xy_std_m=20.0`。
- 通过 `NoiseScheduleVP(schedule=linear, beta_min=0.1, beta_max=20.0)` 反解 `t_start`。
- `t_start` 被限制在 `[1e-3, level+1 的默认上限]`。
- `base_steps=10`。
- level 3 或 hard fallback 使用 `t_start=1.0, steps=10`。

## 实验环境

| 项目 | 值 |
| --- | --- |
| 工作目录 | `/home/ubuntu/code/hezexiang/Diffusion-Planner` |
| Python | `/home/ubuntu/anaconda3/envs/dp/bin/python` |
| CUDA | `CUDA_VISIBLE_DEVICES=0,1`，实际 planner device 为 `cuda` |
| nuPlan devkit | `/home/ubuntu/code/hezexiang/Diffusion-Planner/nuplan-devkit` |
| 数据根目录 | `/media/ubuntu/T9/dataset` |
| 地图目录 | `/media/ubuntu/T9/dataset/maps` |
| trainval db 目录 | `/media/ubuntu/T9/dataset/nuplan-v1.1/trainval` |
| 实验根目录 | `/home/ubuntu/code/hezexiang/Diffusion-Planner/exp` |

## 评测设置

| 项目 | 值 |
| --- | --- |
| challenge | `closed_loop_nonreactive_agents` |
| scenario filter | `val14` |
| scenario builder | `nuplan` |
| 场景数 | 1118 |
| planner | `diffusion_planner` |
| planner checkpoint | `checkpoints/model.pth` |
| planner args | `checkpoints/args.json` |
| gate checkpoint | `0hzxcode/gate_v2_output/runs/ego_nbr_perstep/best.pt` |
| `enable_warmstart` | `true` |
| worker | `sequential` |
| CPU affinity | `0` |
| progress bar | `true` |
| metric summary callback | disabled |

最终成功命令：

```bash
NUPLAN_DATA_ROOT=/media/ubuntu/T9/dataset \
NUPLAN_MAPS_ROOT=/media/ubuntu/T9/dataset/maps \
SPLIT=val14 \
SIM_WORKER=sequential \
ENABLE_SIMULATION_PROGRESS_BAR=true \
CPU_AFFINITY=0 \
TQDM_DISABLE=1 \
PYTHONFAULTHANDLER=1 \
bash sim_gate_warmstart_runner.sh
```

生成的 Hydra 关键 override：

```text
+simulation=closed_loop_nonreactive_agents
planner=diffusion_planner
planner.diffusion_planner.config.args_file=checkpoints/args.json
planner.diffusion_planner.ckpt_path=checkpoints/model.pth
planner.diffusion_planner.gate_ckpt_path=0hzxcode/gate_v2_output/runs/ego_nbr_perstep/best.pt
planner.diffusion_planner.enable_warmstart=true
scenario_builder=nuplan
scenario_filter=val14
worker=sequential
enable_simulation_progress_bar=true
~main_callback.metric_summary_callback
```

## 实验产物

主结果目录：

```text
exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14_gate_warmstart/diffusion_planner_release/model_2026-07-08-11-20-44
```

关键文件：

| 文件 | 作用 |
| --- | --- |
| `runner_report.parquet` | 每个 scenario 的 runner runtime |
| `aggregator_metric/closed_loop_nonreactive_agents_weighted_average_metrics_2026.07.08.11.20.51.parquet` | nuPlan aggregate 指标 |
| `metrics/*.parquet` | 各项 nuPlan metric |
| `nuboard_1783480851.nuboard` | nuBoard 文件 |
| `log.txt` | nuPlan run 日志 |
| `0hzxcode/gate_v2_output/val14_closedloop/frames.jsonl` | 逐帧 gate / warm-start 诊断 |
| `0实验结果/DP-val14-gate-warmstart.md` | 汇总报告 |

汇总命令：

```bash
/home/ubuntu/anaconda3/envs/dp/bin/python -B 0hzxcode/gate_v2/summarize_val14_gate.py
```

## 验证命令

代码编译检查：

```bash
/home/ubuntu/anaconda3/envs/dp/bin/python -B -m compileall -q 0hzxcode/gate_v2 diffusion_planner data_process_gate.py
```

gate_v2 单元/集成轻量测试：

```bash
/home/ubuntu/anaconda3/envs/dp/bin/python -B 0hzxcode/gate_v2/test_gate_v2.py
```

验证结果：

- `compileall` 通过。
- `gate_v2 tests passed`。
- val14 全量闭环评测成功：1118 / 1118 success，0 failed。

## 实验结果摘要

| 指标 | 基线 DP | gate warm-start | 变化 |
| --- | ---: | ---: | ---: |
| CLS-NR 总分 | 89.59 | 85.23 | -4.36 |
| 零分场景数 | 63 / 1118 | 104 / 1118 | +41 |
| 平均 NFE | 11.00 | 4.93 | 2.23x |
| 平均 decoder 延迟 | 19.00 ms | 12.18 ms | 1.56x |
| runner mean `compute_trajectory` | 121.30 ms | 110.86 ms | 1.09x |
| runner median `compute_trajectory` | 91.95 ms | 65.37 ms | 1.41x |
| 主动 full fallback 率 | - | 0.009 | - |
| 被动 fallback 率 | - | 0.025 | - |

全局指标变化：

| 指标 | 基线 | gate warm-start | 变化 |
| --- | ---: | ---: | ---: |
| NC | 95.75 | 93.25 | -2.50 |
| DAC | 98.30 | 97.85 | -0.45 |
| DDC | 99.60 | 99.55 | -0.04 |
| MP | 99.91 | 98.84 | -1.07 |
| TTC | 90.43 | 87.03 | -3.40 |
| EP | 94.16 | 92.70 | -1.46 |
| SC | 97.26 | 96.56 | -0.70 |
| Comfort | 95.08 | 91.50 | -3.58 |

主要掉点来自 Comfort、TTC、NC 和 EP。DAC / DDC 基本保持，说明 warm-start 主要影响轨迹平滑性、碰撞时间和局部交互风险，而不是大规模越界或逆行。

按场景类型掉点最大的类别：

| 场景类型 | 基线 | gate warm-start | 变化 |
| --- | ---: | ---: | ---: |
| `starting_right_turn` | 78.14 | 69.85 | -8.29 |
| `following_lane_with_lead` | 97.96 | 89.90 | -8.06 |
| `low_magnitude_speed` | 90.57 | 83.97 | -6.60 |
| `high_lateral_acceleration` | 86.04 | 79.62 | -6.43 |
| `starting_straight_traffic_light_intersection_traversal` | 94.70 | 88.31 | -6.38 |

## 注意事项

- 本次全量 run 为稳定性使用 `worker=sequential`，wall-clock 为 6:42:08；它不能和基线报告中非顺序/并发评测的 1h15m 直接比较。
- 更可比的速度指标是 decoder 内部延迟、NFE，以及 runner `compute_trajectory` 统计。
- `score_threshold_m=0.1856259704` 较低，如果直接作为 full fallback 阈值会导致过多完整去噪；本次实现中它只保留为告警诊断。
- 当前结果说明 gate warm-start 已能显著降低 NFE，但闭环质量损失仍偏大，后续优化重点应放在 Comfort / TTC 相关场景，特别是右转、跟车、低速和高横向加速度场景。
