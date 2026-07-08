# val14 gate warm-start 评测汇总

## 诊断统计 (frames.jsonl)

- 帧数: 166806
- 平均 NFE: 4.93 (基线 11)
- NFE 加速比: 2.23x
- 平均 decoder 延迟: 12.18 ms (profile 基线 19.0 ms)
- decoder 加速比: 1.56x
- 主动判难率: 0.009
- 被动回退率: 0.025

## 仿真分数

- aggregator: `/home/ubuntu/code/hezexiang/Diffusion-Planner/exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14_gate_warmstart/diffusion_planner_release/model_2026-07-08-11-20-44/aggregator_metric/closed_loop_nonreactive_agents_weighted_average_metrics_2026.07.08.11.20.51.parquet`

- CLS-NR 总分: **85.23** (基线 89.59, Δ -4.36)
- runner 平均 compute_trajectory: 110.86 ms

## 与原始 DP val14 基线对比

基线结果来自 `0实验结果/DP-val14.md` / `model_2026-06-15-16-34-20`。

| 指标 | 基线 | gate warm-start | Δ |
| --- | ---: | ---: | ---: |
| CLS-NR 总分 | 89.59 | 85.23 | -4.36 |
| 无责碰撞 NC | 95.75 | 93.25 | -2.50 |
| 可行驶区域 DAC | 98.30 | 97.85 | -0.45 |
| 行驶方向 DDC | 99.60 | 99.55 | -0.04 |
| 是否前进 MP | 99.91 | 98.84 | -1.07 |
| 碰撞时间 TTC | 90.43 | 87.03 | -3.40 |
| 沿专家路线进度 EP | 94.16 | 92.70 | -1.46 |
| 限速合规 SC | 97.26 | 96.56 | -0.70 |
| 舒适度 C | 95.08 | 91.50 | -3.58 |

零分场景数从 63 / 1118 增加到 104 / 1118。掉点主要来自舒适度、TTC、无责碰撞和专家路线进度，而 DAC/DDC 基本保持。

## 按场景类型掉点

| 场景类型 | 基线 | gate warm-start | Δ |
| --- | ---: | ---: | ---: |
| `starting_right_turn` | 78.14 | 69.85 | -8.29 |
| `following_lane_with_lead` | 97.96 | 89.90 | -8.06 |
| `low_magnitude_speed` | 90.57 | 83.97 | -6.60 |
| `high_lateral_acceleration` | 86.04 | 79.62 | -6.43 |
| `starting_straight_traffic_light_intersection_traversal` | 94.70 | 88.31 | -6.38 |
| `starting_left_turn` | 87.66 | 82.16 | -5.49 |
| `changing_lane` | 85.65 | 81.61 | -4.04 |
| `waiting_for_pedestrian_to_cross` | 91.42 | 87.84 | -3.59 |
| `traversing_pickup_dropoff` | 75.09 | 72.00 | -3.09 |
| `high_magnitude_speed` | 93.84 | 91.12 | -2.72 |
| `near_multiple_vehicles` | 95.56 | 93.57 | -2.00 |
| `stationary_in_traffic` | 98.33 | 96.43 | -1.91 |
| `stopping_with_lead` | 96.58 | 95.47 | -1.11 |
| `behind_long_vehicle` | 97.28 | 98.55 | +1.27 |

## 推理加速

| 口径 | 基线 | gate warm-start | 加速比 |
| --- | ---: | ---: | ---: |
| 平均 NFE | 11.00 | 4.93 | 2.23x |
| 平均 decoder 延迟 | 19.00 ms | 12.18 ms | 1.56x |
| runner `compute_trajectory_runtimes_mean` | 121.30 ms | 110.86 ms | 1.09x |
| runner `compute_trajectory_runtimes_median` | 91.95 ms | 65.37 ms | 1.41x |

本次全量评测 wall-clock 为 6:42:08，但这是为了避开本机并发仿真的 native FPE 而使用 `worker=sequential` 跑出的稳定结果；它不能和基线报告中的并发/非顺序 wall-clock 直接比较。更可比的是 planner/decoder 内部 runtime。
