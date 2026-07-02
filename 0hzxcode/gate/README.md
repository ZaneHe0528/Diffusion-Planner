# 原始观测 Gate（BLUE 式）

仿照 BLUE 的 gate 设计（冻结主干 + 轻量 MLP gate，~0.2M 参数），但输入不用
planner encoding，而是闭环推理时**零成本可得的原始观测**，预测：

- **d_hat**：帧间变化距离（米，回归，`perstep_max_m` 口径，与阈值回退度量一致）；
- **level**：帧间变化档位（默认 4 档 低/中/高/满，边界=训练集 d 分位数
  0.5/0.8/0.95，存于 checkpoint，供映射到 warm-start 操作点 (t_s, steps)）。

## 特征组（按先验重要性排序，均可开关）

见 `gate_features.py` 的 `FEATURE_GROUPS`：

| 先验排名 | 组名 | 内容 |
|---:|---|---|
| 1 | `ego_history` | 过去 2.1s 自车状态 [21,7]（当前帧系）+ 模型内派生运动学（速度/加速度/yaw rate/曲率等 14 维） |
| 2 | `neighbor_agents` | 32 个邻居过去 2.1s 轨迹 [32,21,11]（含类型 one-hot） |
| 3 | `route_lanes` | 路线车道 [25,20,12] + 限速（含红绿灯 one-hot） |
| 4 | `lanes` | 周边车道 [70,20,12] + 限速（红绿灯状态在最后 4 维） |
| 5 | `static_objects` | 静态障碍 [5,10] |
| 6 | `prev_d`（默认关） | 上一帧实测 d [d_prev, log1p, valid]，加载时从标签派生；闭环缓存近乎免费，但训练标签来自 full-denoise 回放，warm-start 部署后分布会漂移，启用需 DAgger 式回灌验证 |

关闭方式（三选一）：

- 训练命令 `--disable-groups lanes static_objects`（重训消融，彻底排除）；
- 训练命令 `--enable-groups ego_history`（白名单，如 ego-only 对照）；
- 推理/评测期 `model(batch, disabled_groups={"lanes"})`：训练时有 group-dropout
  （默认 0.15），单个训好的模型支持推理时任意关组（重要性报告即用此机制）。

## 网络结构（`gate_model.py`）

```
每组: token 级共享 MLP(隐层128) -> masked mean/max 池化 -> 64 维组 embedding
      (ego_history / encoding 为向量组，直接 MLP -> 64)
trunk: concat 各组 embedding -> LayerNorm -> Linear(128) + SiLU   [单隐层，对齐 BLUE]
双头:  reg 头 -> 标准化 log1p(d)（Huber 损失）
      cls 头 -> K 档 level（加权 CE）
```

标准化统计、log1p(d) 标准化参数、level 边界（米）全部存进 checkpoint buffer，
闭环集成时只需加载 `best.pt` 并喂 planner 已有的 model_inputs + ego 历史。

## 数据

`export_gate_dataset_from_val14_logs.py` 直接读 **val14 闭环仿真日志**（d 标签
就来自这些日志，`new_iteration` = 日志帧下标，逐帧对齐；ego/观测是闭环下
planner 真实输入，与部署分布一致）：

```bash
/home/ubuntu/anaconda3/envs/dp/bin/python 0hzxcode/gate/export_gate_dataset_from_val14_logs.py \
  --max-parallel 6
# 输出 0hzxcode/gate_output/gate_dataset_chunks/chunk_*.npz（可断点续传，重跑会跳过已完成 chunk）
```

> 注意：旧 `m_probe_output/probe_dataset.npz` 有两个 bug，不要再用：
> 1. DB 重建未传 ScenarioMapping，与闭环日志帧错位约 3s 且采样率不同（20Hz vs 10Hz），
>    (encoding, d) 不是同一帧；
> 2. ego_features 从推理态 ego_current_state 读出，恒为全零。
> 旧 M-probe 的 FAIL 结论（AUROC 0.674）建立在错位数据上，仅供参考。

## 训练

```bash
# 全特征组
CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/dp/bin/python 0hzxcode/gate/train_gate.py \
  --output-dir 0hzxcode/gate_output/runs/all_groups

# ego-only 对照（rule-based 基线，评测矩阵第 4 行）
CUDA_VISIBLE_DEVICES=1 ... train_gate.py --enable-groups ego_history \
  --output-dir 0hzxcode/gate_output/runs/ego_only
```

产出（`--output-dir` 下）：

- `best.pt`：最优 checkpoint（按 val Spearman 选）；
- `metrics.json` / `importance_report.md`：train/val/test 指标 +
  特征组实测重要性（LOO ΔSpearman、solo、permutation）+ 按场景类型分解；
- `training_curves.png`：损失曲线 + val 指标 + test 预测散点。

评测协议与 M-probe 一致：按 scenario 划分 train/val/test（seed 3407，15%/15%），
hard = d > train P90，训练集做 BLUE 式时间冗余下采样（连续帧签名余弦 >0.99 的
段保留 max(2, ceil(sqrt(L))) 帧）。

## val14 结果（2026-07-02，test 集 21,695 帧 / 168 场景）

| run | Spearman | AUROC(hard@P90) | RMSE | MAE | hard recall | level acc | adj acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ego_only`（运动学基线） | 0.6428 | 0.6922 | 2.258 | 0.845 | 0.935 | 0.499 | 0.821 |
| `all_groups`（全部原始观测） | 0.6414 | 0.7077 | 2.232 | 0.867 | 0.943 | 0.446 | 0.749 |
| `all_plus_prevd`（+上一帧实测 d） | **0.7414** | **0.8135** | **2.096** | **0.790** | 0.931 | 0.538 | 0.867 |

诊断基线（test）：单特征当前速度 Spearman 0.581 / AUROC 0.639；
"上一帧 d 直接当预测"（持续性）Spearman 0.695 / AUROC 0.796。

解读：
- 原始观测里关于 d 的信号大头在 **ego 运动学**；邻车/车道主要提升 hard/easy
  判别（AUROC +0.015），对秩相关几乎无增益；静态障碍无贡献。
- `prev_d` 是最强单信号，与观测组合后显著超过持续性基线
  （0.741 vs 0.695，AUROC 0.814 vs 0.796）。
- 旧 M-probe 的 FAIL（encoding AUROC 0.674）建立在帧错位数据上；
  修正后原始观测 gate 已过 AUROC≥0.70 与 hard recall≥0.85 门槛。

## 扩展到完整 trainval

流程不变，前提是先有 trainval 的闭环日志与 d 标签：

1. 在 trainval 场景上跑闭环仿真（`sim_diffusion_planner_runner.sh` 换 filter）；
2. `0hzxcode/analyze_adjacent_traj_l2.py <log_root> --output-dir <out>` 算相邻帧距离；
3. `0hzxcode/m_probe/make_d_labels_from_adjacent_traj_l2.py --input <out>/per_pair_overlap_l2.csv --require-time-aligned --output <d_labels.csv>`；
4. `export_gate_dataset_from_val14_logs.py --labels-csv <d_labels.csv> --output-dir <chunks>`（脚本不依赖 val14 特定路径，读 labels csv 的 log_file 列）；
5. `train_gate.py --dataset-dir <chunks> ...`。
