# Gate 决策导向评测 — `best.pt`

## 主指标（面向分档/复用决策）

| split | Spearman | rmse_log1p | adj level acc | reuse@HR95 | reuse@HR99 | 校准单调 |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.6589 | 0.3415 | 0.904 | 0.279 | 0.171 | True |
| val | 0.6013 | 0.3380 | 0.902 | 0.237 | 0.095 | True |
| test | 0.6351 | 0.3560 | 0.895 | 0.301 | 0.131 | True |

## 说明

- **Spearman / adj level acc / reuse@HR95** 是主指标；Pearson/原始 RMSE 对重尾 d 不适用，见 JSON。
- **reuse@HR95**：hard_recall≥95% 约束下最大可复用率。
- **校准单调**：按 d_hat 分 10 箱，真实 d 中位数是否单调递增。

## 按场景类型 Spearman (test)

- near_multiple_vehicles: 0.806 (n=1678)
- waiting_for_pedestrian_to_cross: 0.724 (n=647)
- starting_right_turn: 0.528 (n=1549)
- behind_long_vehicle: 0.499 (n=258)
- low_magnitude_speed: 0.484 (n=1680)
- changing_lane: 0.478 (n=903)
- traversing_pickup_dropoff: 0.432 (n=2454)
- stopping_with_lead: 0.376 (n=2195)
- high_magnitude_speed: 0.359 (n=1032)
- high_lateral_acceleration: 0.352 (n=1419)
- stationary_in_traffic: 0.322 (n=2456)
- starting_left_turn: 0.285 (n=2324)
- starting_straight_traffic_light_intersection_traversal: 0.227 (n=2454)
- following_lane_with_lead: -0.041 (n=646)

