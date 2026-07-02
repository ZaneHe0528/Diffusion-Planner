# Gate 训练报告

- 样本数: 144446，场景数: 1118，划分: {'train': 89001, 'val': 21715, 'test': 21695}
- 启用特征组: ['ego_history', 'neighbor_agents', 'route_lanes', 'lanes', 'static_objects']，参数量: 0.231 M
- hard 阈值: 2.650 m，level 边界: [0.525, 1.62, 4.062] m

## 各划分指标

| split | RMSE | MAE | Pearson | Spearman | AUROC | hard recall | false easy | level acc | adj acc | macro F1 | top recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2.207 | 0.819 | 0.342 | 0.6745 | 0.7663 | 0.967 | 0.033 | 0.469 | 0.792 | 0.403 | 0.643 |
| val | 2.036 | 0.813 | 0.285 | 0.6120 | 0.7027 | 0.950 | 0.050 | 0.425 | 0.767 | 0.348 | 0.415 |
| test | 2.232 | 0.867 | 0.283 | 0.6414 | 0.7077 | 0.943 | 0.057 | 0.446 | 0.749 | 0.358 | 0.389 |

## 特征组重要性（test 集，按 leave-one-out ΔSpearman 排序）

| 实测排名 | 特征组 | 先验排名 | LOO ΔSpearman | LOO ΔAUROC | solo Spearman | perm ΔSpearman |
|---:|---|---:|---:|---:|---:|---:|
| 1 | ego_history | 1 | 0.0617 | 0.0188 | 0.5996 | 0.1974 |
| 2 | neighbor_agents | 2 | 0.0183 | 0.0125 | 0.5298 | 0.0683 |
| 3 | lanes | 4 | 0.0042 | 0.0091 | 0.4784 | 0.0353 |
| 4 | route_lanes | 3 | 0.0019 | 0.0102 | 0.4933 | 0.0443 |
| 5 | static_objects | 5 | -0.0060 | -0.0041 | 0.0105 | -0.0000 |

## test 按场景类型 Spearman

- near_multiple_vehicles: 0.800 (n=1678)
- behind_long_vehicle: 0.698 (n=258)
- starting_right_turn: 0.569 (n=1549)
- stationary_in_traffic: 0.537 (n=2456)
- stopping_with_lead: 0.505 (n=2195)
- low_magnitude_speed: 0.452 (n=1680)
- waiting_for_pedestrian_to_cross: 0.446 (n=647)
- high_magnitude_speed: 0.436 (n=1032)
- changing_lane: 0.424 (n=903)
- high_lateral_acceleration: 0.407 (n=1419)
- traversing_pickup_dropoff: 0.374 (n=2454)
- starting_left_turn: 0.309 (n=2324)
- starting_straight_traffic_light_intersection_traversal: 0.298 (n=2454)
- following_lane_with_lead: 0.068 (n=646)
