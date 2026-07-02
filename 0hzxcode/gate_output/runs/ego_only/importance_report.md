# Gate 训练报告

- 样本数: 144446，场景数: 1118，划分: {'train': 89001, 'val': 21715, 'test': 21695}
- 启用特征组: ['ego_history']，参数量: 0.038 M
- hard 阈值: 2.650 m，level 边界: [0.525, 1.62, 4.062] m

## 各划分指标

| split | RMSE | MAE | Pearson | Spearman | AUROC | hard recall | false easy | level acc | adj acc | macro F1 | top recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2.267 | 0.844 | 0.284 | 0.6078 | 0.7014 | 0.954 | 0.046 | 0.473 | 0.812 | 0.386 | 0.446 |
| val | 2.055 | 0.790 | 0.288 | 0.6158 | 0.6901 | 0.950 | 0.050 | 0.472 | 0.809 | 0.365 | 0.357 |
| test | 2.258 | 0.845 | 0.284 | 0.6428 | 0.6922 | 0.935 | 0.065 | 0.499 | 0.821 | 0.384 | 0.384 |

## test 按场景类型 Spearman

- near_multiple_vehicles: 0.795 (n=1678)
- waiting_for_pedestrian_to_cross: 0.740 (n=647)
- stopping_with_lead: 0.732 (n=2195)
- stationary_in_traffic: 0.621 (n=2456)
- behind_long_vehicle: 0.616 (n=258)
- starting_right_turn: 0.535 (n=1549)
- changing_lane: 0.501 (n=903)
- traversing_pickup_dropoff: 0.410 (n=2454)
- low_magnitude_speed: 0.407 (n=1680)
- high_magnitude_speed: 0.377 (n=1032)
- starting_left_turn: 0.289 (n=2324)
- high_lateral_acceleration: 0.286 (n=1419)
- starting_straight_traffic_light_intersection_traversal: 0.165 (n=2454)
- following_lane_with_lead: 0.156 (n=646)
