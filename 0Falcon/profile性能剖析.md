Profile 完成。脚本在 `profile/profile_baseline.py`（数据保存在 `profile/profile_results.json`），用的是真实 Las Vegas 地图 + 合成场景（30 辆邻车 + 5 个静态物体，ego 每步前移模拟闭环重规划）+ 随机初始化权重（无 checkpoint，但 FLOPs 与训练后完全一致，不影响延迟测量）。环境用的是 navsim conda 环境（有 nuplan-devkit + CUDA torch）。

## 结果：单次 planner 调用延迟分解（B=1，RTX 4090D）

| 阶段 | 耗时 (mean) | 占比 |
|---|---|---|
| 数据处理 `observation_adapter`（CPU） | **85.4 ms**（median 50 / p90 130） | **79.9%** |
| 归一化 | 0.2 ms | 0.2% |
| Encoder（GPU） | 2.1 ms | 1.9% |
| **扩散去噪 Decoder（GPU, steps=10, NFE=11）** | **19.3 ms** | **18.0%** |
| 合计 | ~107 ms | |

去噪步数缩减的实测扩展性（单次 DiT 前向 = 1.37 ms，延迟几乎与 NFE 严格线性）：

| steps | NFE | decoder 耗时 | decoder 加速 | 端到端加速 |
|---|---|---|---|---|
| 10（基线） | 11 | 19.3 ms | 1.0x | 1.0x |
| 5 | 6 | 10.5 ms | 1.84x | 1.09x |
| 3 | 4 | 7.0 ms | 2.76x | 1.13x |
| 2 | 3 | 5.2 ms | 3.68x | 1.15x |

数据处理内部热点（cProfile，`profile/profile_data_breakdown.py`）：约 **90% 耗在地图向量化**——`map_process`（59%，其中 `_convert_lane_to_fixed_size` 逐点调用 shapely `interpolate`，30 次调用产生 12.6 万次 shapely 调用）+ `get_neighbor_vector_set_map`（32%）；agent 处理只占 ~6%。

## 判断

这验证了我之前的担心，但结论比想象的更清晰：

1. **端到端口径下，Falcon 的天花板只有 1.15x**——数据处理（85 ms 纯 Python shapely 循环）把去噪完全淹没了。只讲端到端实时性，故事不成立。

2. **模型推理口径下，故事完全成立**：GPU 部分（encoder + decoder）共 21.4 ms，其中去噪占 90%。这也是 Diffusion-Planner 论文及同类工作报告延迟的标准口径。Falcon 若能把平均 NFE 从 11 压到 2~3，模型推理延迟从 21.4 ms → 5~7 ms，**3~4x 加速**；若 warm-start 后平均只需 1~2 步，可逼近 7x，和 Falcon 原文宣称的 2-7x 区间吻合。

3. **数据处理的 85 ms 是纯工程问题**（逐点 shapely 插值，完全可以向量化或缓存），不是方法论问题。建议你论文里明确口径为"模型推理延迟"，同时可以顺手把地图处理向量化作为工程附带改进——但那是另一件事，不影响 Falcon 创新点的评估。

4. 两个测量注意事项：① 随机权重不影响延迟，但意味着我没法测"去噪质量 vs 步数"的退化曲线，那一步需要 checkpoint 和真实数据；② 4090D 上单次 DiT 前向仅 1.37 ms，模型很小（6M 参数），所以固定开销（solver 逻辑、kernel launch）占比不小——这反而对 Falcon 有利，因为减少 NFE 同时减少这些开销。

结论：**值得做，但论文叙事必须锚定"去噪/模型推理延迟"而非端到端延迟**。下一步如果你拿到官方 checkpoint，我可以把"步数 vs 轨迹质量"的退化曲线补上，那是决定 Falcon 增益空间的最后一块拼图。

顺带说明：navsim 环境里 GDAL 与 torch 存在随机性的导入冲突（段错误），脚本里已规避（`fiona` 必须先于 `torch` 导入），偶尔仍会崩，重跑一次即可。