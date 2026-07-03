全部完成。汇总如下。

## 结果一览

gate 网络已设计、实现并在 val14 全量数据（144,446 帧 / 1,118 场景，按场景切分 train/val/test）上训练完毕，**0.23M 参数**（对齐 BLUE 的 0.11M 量级）。test 集结果：

| run | Spearman | AUROC(hard@P90) | RMSE | hard recall | level 相邻准确率 |
|---|---:|---:|---:|---:|---:|
| `ego_only`（运动学基线） | 0.643 | 0.692 | 2.258 | 0.935 | 0.821 |
| `all_groups`（全部原始观测） | 0.641 | 0.708 | 2.232 | 0.943 | 0.749 |
| `all_plus_prevd`（+上一帧实测 d） | **0.741** | **0.814** | 2.096 | 0.931 | 0.867 |

作为参照：旧 M-probe 的 encoding 探针是 Spearman 0.625 / AUROC 0.674（FAIL），单特征"当前速度"是 0.581 / 0.639。

## 途中发现并修复了两个旧数据 bug（重要）

1. **旧 probe 数据集帧错位约 3 秒**：旧 DB 导出没传 `ScenarioMapping`，DB 场景从事件时刻开始、20Hz；而 d 标签来自闭环仿真（事件前 3s 开始、10Hz）。我实测同一 iteration 两边差 2.95s。所以旧 M-probe 的 (encoding, d) 不是同一帧，**FAIL 结论不可靠**。新导出直接读闭环仿真日志（`new_iteration` 就是日志帧下标，天然对齐，且观测是闭环下 planner 真实输入）。
2. **旧 ego 基线恒为全零**（从推理态 `ego_current_state` 硬编码字段读的），所以"运动学 Spearman=0"是假的——修正后运动学单独就有 0.64。

## 特征重要性（代码中先验排序 + 训练后实测排序）

`gate_features.py` 的 `FEATURE_GROUPS` 按先验重要性排序，每组带理由；训练后自动输出实测对照（leave-one-out / solo / permutation 三种口径）。实测排名：**ego_history ≫ neighbor_agents > lanes ≈ route_lanes > static_objects(≈0)**，与先验一致（除 lanes/route_lanes 互换，差异很小）。关键洞察：**d 的信号大头在自车运动学**，邻车/地图只增强 hard 帧判别（AUROC +0.015）；可选组 `prev_d`（上一帧实测 d，闭环缓存近乎免费）是最强单信号，但有"训练/闭环分布漂移"风险（README 已注明需 DAgger 式回灌验证，默认关闭）。

## 关闭某组输入的三种方式

- 训练时白名单/黑名单：`--enable-groups ego_history` 或 `--disable-groups lanes static_objects`（重训消融）；
- 推理时即时关：`model(batch, disabled_groups={"lanes"})`——训练带 group-dropout（0.15），一个模型支持任意关组组合；
- level 档位边界（默认 train 分位数 0.5/0.8/0.95 → 低/中/高/满 4 档，单位米）与归一化统计都存在 checkpoint 里，闭环集成只需 `load_gate("best.pt")`。

## 产出文件

- 代码：`0hzxcode/gate/`（`gate_features.py` 特征组注册表、`export_gate_dataset_from_val14_logs.py` 导出、`gate_model.py`、`gate_dataset.py`、`train_gate.py`、`README.md` 含 trainval 扩展步骤）；
- 数据：`0hzxcode/gate_output/gate_dataset_chunks/`（1.8GB，224 chunks，可断点续传）；
- 三个 run：`0hzxcode/gate_output/runs/{ego_only,all_groups,all_plus_prevd}/`，各含 `best.pt`、`metrics.json`、`importance_report.md`（含按场景类型分解）、`training_curves.png`。

![训练曲线与测试散点](/home/ubuntu/code/hezexiang/Diffusion-Planner/0hzxcode/gate_output/runs/all_plus_prevd/training_curves.png)

一个值得注意的信号：按场景类型分解显示 `near_multiple_vehicles` Spearman 0.80 而 `following_lane_with_lead` 只有 0.07（见 `all_groups` 报告），后者 d 波动可能主要来自扩散采样噪声而非场景可预测变化——这对后续设定分箱映射和 ε 兜底策略有参考价值。