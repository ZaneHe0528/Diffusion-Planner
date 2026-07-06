# Gate 决策导向评测 — `best.pt`

## 主指标（面向分档/复用决策）

| split | Spearman | rmse_log1p | adj level acc | reuse@HR95 | reuse@HR99 | 校准单调 |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.7129 | 0.4267 | 0.915 | 0.372 | 0.226 | True |
| val | 0.7076 | 0.4182 | 0.916 | 0.366 | 0.245 | True |
| test | 0.7412 | 0.4338 | 0.916 | 0.368 | 0.256 | True |

## 说明

- **Spearman / adj level acc / reuse@HR95** 是主指标；Pearson/原始 RMSE 对重尾 d 不适用，见 JSON。
- **reuse@HR95**：hard_recall≥95% 约束下最大可复用率。
- **校准单调**：按 d_hat 分 10 箱，真实 d 中位数是否单调递增。

## 按场景类型 Spearman (test)

- near_multiple_vehicles: 0.847 (n=1678)
- behind_long_vehicle: 0.780 (n=258)
- waiting_for_pedestrian_to_cross: 0.766 (n=647)
- stopping_with_lead: 0.682 (n=2195)
- low_magnitude_speed: 0.680 (n=1680)
- starting_right_turn: 0.652 (n=1549)
- stationary_in_traffic: 0.633 (n=2456)
- changing_lane: 0.587 (n=903)
- traversing_pickup_dropoff: 0.570 (n=2454)
- high_magnitude_speed: 0.520 (n=1032)
- high_lateral_acceleration: 0.513 (n=1419)
- starting_left_turn: 0.454 (n=2324)
- starting_straight_traffic_light_intersection_traversal: 0.378 (n=2454)
- following_lane_with_lead: 0.210 (n=646)

