# Gate 决策导向评测 — `best.pt`

## 主指标（面向分档/复用决策）

| split | Spearman | rmse_log1p | adj level acc | reuse@HR95 | reuse@HR99 | 校准单调 |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.6906 | 0.4418 | 0.912 | 0.295 | 0.186 | True |
| val | 0.5880 | 0.4689 | 0.893 | 0.197 | 0.000 | True |
| test | 0.6261 | 0.4885 | 0.886 | 0.271 | 0.000 | True |

## 说明

- **Spearman / adj level acc / reuse@HR95** 是主指标；Pearson/原始 RMSE 对重尾 d 不适用，见 JSON。
- **reuse@HR95**：hard_recall≥95% 约束下最大可复用率。
- **校准单调**：按 d_hat 分 10 箱，真实 d 中位数是否单调递增。

## 按场景类型 Spearman (test)

- near_multiple_vehicles: 0.792 (n=1678)
- waiting_for_pedestrian_to_cross: 0.705 (n=647)
- behind_long_vehicle: 0.566 (n=258)
- starting_right_turn: 0.552 (n=1549)
- low_magnitude_speed: 0.508 (n=1680)
- stopping_with_lead: 0.507 (n=2195)
- changing_lane: 0.436 (n=903)
- high_magnitude_speed: 0.425 (n=1032)
- traversing_pickup_dropoff: 0.417 (n=2454)
- high_lateral_acceleration: 0.416 (n=1419)
- starting_left_turn: 0.306 (n=2324)
- starting_straight_traffic_light_intersection_traversal: 0.221 (n=2454)
- stationary_in_traffic: 0.217 (n=2456)
- following_lane_with_lead: -0.026 (n=646)

