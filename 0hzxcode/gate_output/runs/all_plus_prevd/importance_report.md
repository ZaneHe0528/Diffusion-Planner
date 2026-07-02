# Gate 训练报告

- 样本数: 144446，场景数: 1118，划分: {'train': 89001, 'val': 21715, 'test': 21695}
- 启用特征组: ['ego_history', 'neighbor_agents', 'route_lanes', 'lanes', 'static_objects', 'prev_d']，参数量: 0.241 M
- hard 阈值: 2.650 m，level 边界: [0.525, 1.62, 4.062] m

## 各划分指标

| split | RMSE | MAE | Pearson | Spearman | AUROC | hard recall | false easy | level acc | adj acc | macro F1 | top recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2.110 | 0.779 | 0.455 | 0.7191 | 0.8230 | 0.957 | 0.043 | 0.527 | 0.878 | 0.462 | 0.536 |
| val | 1.924 | 0.745 | 0.453 | 0.7110 | 0.8135 | 0.950 | 0.050 | 0.513 | 0.883 | 0.434 | 0.428 |
| test | 2.096 | 0.790 | 0.470 | 0.7414 | 0.8135 | 0.931 | 0.069 | 0.538 | 0.867 | 0.450 | 0.469 |

## 特征组重要性（test 集，按 leave-one-out ΔSpearman 排序）

| 实测排名 | 特征组 | 先验排名 | LOO ΔSpearman | LOO ΔAUROC | solo Spearman | perm ΔSpearman |
|---:|---|---:|---:|---:|---:|---:|
| 1 | prev_d | 6 | 0.1063 | 0.1167 | 0.7281 | 0.3802 |
| 2 | ego_history | 1 | 0.0127 | 0.0049 | 0.6007 | 0.0778 |
| 3 | static_objects | 5 | -0.0015 | -0.0015 | 0.0301 | -0.0001 |
| 4 | route_lanes | 3 | -0.0020 | -0.0023 | 0.5036 | 0.0166 |
| 5 | lanes | 4 | -0.0025 | 0.0016 | 0.4284 | 0.0070 |
| 6 | neighbor_agents | 2 | -0.0035 | -0.0010 | 0.4767 | 0.0098 |

## test 按场景类型 Spearman

- behind_long_vehicle: 0.798 (n=258)
- near_multiple_vehicles: 0.796 (n=1678)
- waiting_for_pedestrian_to_cross: 0.751 (n=647)
- stationary_in_traffic: 0.694 (n=2456)
- low_magnitude_speed: 0.693 (n=1680)
- starting_right_turn: 0.658 (n=1549)
- changing_lane: 0.575 (n=903)
- traversing_pickup_dropoff: 0.560 (n=2454)
- high_lateral_acceleration: 0.559 (n=1419)
- stopping_with_lead: 0.544 (n=2195)
- high_magnitude_speed: 0.537 (n=1032)
- starting_left_turn: 0.443 (n=2324)
- starting_straight_traffic_light_intersection_traversal: 0.393 (n=2454)
- following_lane_with_lead: 0.252 (n=646)
