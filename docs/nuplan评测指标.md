nuPlan v1.1 的评测体系由 **nuplan-devkit** 定义，按仿真模式分为三大类。本仓库 Diffusion Planner 主要报的是 **闭环分数 CLS-NR / CLS-R**（0–100 分），但官方其实还有开环指标。

---

## 一、三种评测模式

| 模式 | 简称 | 说明 |
|------|------|------|
| **开环** | OLS | ego 按 log 回放，planner 输出与专家轨迹对比 |
| **闭环非反应** | CLS-NR | ego 由 LQR + 运动学自行车模型跟踪 planner 轨迹；其他交通参与者按 log 回放 |
| **闭环反应** | CLS-R | ego 同上；**车辆**由 IDM 模型反应式仿真，行人、骑行者和静态物体仍按 log 回放 |

### 常用 benchmark 划分（社区约定，非 nuPlan 官方挑战赛 split）

| 名称 | 来源 | 说明 |
|------|------|------|
| **Val14** | PDM 论文（Dauner et al., CoRL 2023） | 从 validation split 取 14 类挑战场景，每类至多 100 个，共约 1090–1118 个场景 |
| **Test14** | planTF 论文 | 从 test split 取 14 类场景 |
| **Test14-hard** | planTF 论文 | 用基线 planner 筛出的 272 个低分困难场景 |

nuPlan 官方挑战赛使用的是保留测试集，与上述社区 benchmark 不同。

---

## 二、开环指标（Challenge 1 → OLS）

以 **1 Hz** 采样，在 **3s / 5s / 8s** 三个预测 horizon 上与专家轨迹对比：

| 指标 | 英文缩写 | 类型 | 权重 | 含义 |
|------|----------|------|------|------|
| 脱靶率 / 未命中率 | MR | **乘数**（一票否决） | — | 超过 30% 的时间点轨迹偏差过大 → 场景分 = 0 |
| 平均位移误差 | ADE | 加权 | 1 | 逐点 L2 距离均值 |
| 终点位移误差 | FDE | 加权 | 1 | horizon 末端 L2 距离 |
| 平均航向误差 | AHE | 加权 | 2 | 航向角差均值 |
| 终点航向误差 | FHE | 加权 | 2 | horizon 末端航向差 |

MR 的位移阈值：3s → 6 m，5s → 8 m，8s → 16 m。

ADE / FDE / AHE / FHE 的得分为**连续值**（非 0/1 二值）：

```
score = max(0, 1 − error / threshold) ∈ [0, 1]
```

其中位移类阈值 8 m，航向类阈值 0.8 rad。只有 MR 是 0/1 乘数。

---

## 三、闭环指标（Challenge 2/3 → CLS-NR / CLS-R）

共 **8 项核心指标**，分两类：

### 乘数指标（Multiplier，可一票否决）

| 指标 | 缩写 | 含义 | 计分规则 |
|------|------|------|----------|
| **无责碰撞** | NC | 只惩罚 planner 应避免的碰撞 | 与车辆/行人/自行车碰撞 → 0；与物体（锥桶等）碰撞 1 次 → 0.5，多次 → 0 |
| **可行驶区域合规** | DAC | ego 是否在地图可行驶区域内 | 角点越界 > 0.3 m → 0，否则 1 |
| **行驶方向合规** | DDC | 是否逆行 | 逆行 > 6 m → 0；2–6 m → 0.5；否则 1 |
| **是否前进** | MP | 是否沿专家路线有效推进 | 进度比 < 20% → 0，否则 1 |

> **注意**：DDC 在 devkit **实际代码**（`closed_loop_*_weighted_average.yaml` 的 `multiple_metrics`）中是乘数指标，与 PDM 论文一致；但 devkit markdown 文档表格里误标为 weight 5 的加权指标（[Issue #273](https://github.com/motional/nuplan-devkit/issues/273)，至今未修）。跑评测时以代码为准。

### 加权指标（Weighted Average）

| 指标 | 缩写 | 权重 | 含义 |
|------|------|------|------|
| **碰撞时间** | TTC | 5 | 投影 3 s 内最小 TTC，< 0.95 s → 0，否则 1 |
| **沿专家路线进度比** | EP | 5 | ego 沿专家路线的进度 / 专家进度，裁剪到 [0, 1] |
| **限速合规** | SC | 4 | 超速持续时间和严重程度，连续值 [0, 1] |
| **舒适度** | C | 2 | 加速度、横摆角速度、jerk 等是否在阈值内 → 0 或 1 |

舒适度检查的物理量包括：
- 纵向加速度：[-4.05, 2.40] m/s²
- 横向加速度：≤ 4.89 m/s²
- 横摆角速度：≤ 0.95 rad/s
- 横摆角加速度：≤ 1.93 rad/s²
- 纵向 jerk：≤ 4.13 m/s³
- 总 jerk 模：≤ 8.37 m/s³

---

## 四、分数聚合方式

单场景分数公式（PDM 论文与 devkit 代码一致）：

```
scenario_score = (∏ multiplier_scores) × (Σ weight × score) / (Σ weight)
```

**一票否决**（场景分直接为 0）的情况：
- 与车辆或 VRU（行人/骑行者）发生有责碰撞
- 与物体多次有责碰撞
- 驶出可行驶区域
- 逆行超过 6 m
- 进度不足（MP = 0）

**0.5 倍惩罚**（加权平均后再 × 0.5）的情况：
- 与物体发生 1 次有责碰撞
- 逆行 2–6 m（未达一票否决的 6 m 阈值）

**最终分数**：所有场景分数取平均，再 × 100 显示（即 0–100 分）。本仓库 README 中 Diffusion Planner 在 Val14 上的成绩为 **CLS-NR 89.87**、**CLS-R 82.80**（带 refine 版为 94.26 / 92.90）。

---

## 五、辅助指标（nuBoard 可见，不计入总分）

- 平均速度（Mean speed）
- 位移误差（Displacement error）
- 带航向的位移误差（Displacement error with yaw）

---

## 六、与本仓库的关系

`sim_diffusion_planner_runner.sh` 跑的是 `closed_loop_nonreactive_agents` 或 `closed_loop_reactive_agents`，对应 **CLS-NR / CLS-R**。仿真结束后由 nuplan-devkit 离线算上述 8 项指标并聚合，结果在 **nuBoard** 查看。

官方文档：
- [metrics_description.md](https://github.com/motional/nuplan-devkit/blob/master/docs/metrics_description.md)
- [nuplan_metrics_aggeregation.md](https://github.com/motional/nuplan-devkit/blob/master/docs/nuplan_metrics_aggeregation.md)
