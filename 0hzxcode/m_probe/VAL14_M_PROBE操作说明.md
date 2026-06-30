# val14 M-probe 操作说明

本文档说明如何用 val14 的 `*.db` 场景数据，以及 `0hzxcode/analyze_adjacent_traj_l2.py` 生成的相邻规划帧轨迹变化距离，完成 M-probe：

> 在 `(encoding, d)` 上拟合线性回归 / 小 MLP，对比 `encoding` 与 ego 运动学特征预测 `d` 的误差与相关性。

## 目标

M-probe 只验证一个问题：

`Diffusion-Planner` 冻结 encoder 的场景 `encoding` 是否含有足够信号，能事前预测当前帧相对上一帧的轨迹变化距离 `d`。

它不是训练主模型，也不是训练最终 gate。主模型使用现有 checkpoint：

```bash
checkpoints/model.pth
```

项目 Python 环境使用：

```bash
conda activate dp
```

或显式使用：

```bash
/home/ubuntu/anaconda3/envs/dp/bin/python
```

## 当前已有代码

`0hzxcode/m_probe/` 中包含：

- `make_d_labels_from_adjacent_traj_l2.py`  
  把 `adjacent_traj_l2_output/per_pair_overlap_l2.csv` 转成 `d_labels.csv`。

- `make_probe_dataset.py`  
  从已处理 `.npz` 输入缓存导出 encoder mean-pool 后的 `encoding` 和 ego 运动学特征，并与 `d_labels.csv` 按 `sample_id` 合并。

- `make_probe_dataset_from_val14_logs.py`  
  直接读取 val14 simulation log，按 `d_labels.csv` 里的同一批 `sample_id` 重建 planner 输入，导出 `(encoding, d, ego_features)`。

- `run_encoding_d_probe.py`  
  在 `probe_dataset.npz` 上跑 Ridge 回归和 hard/easy logistic probe，输出 `probe_report.md` 与 `probe_results.json`。

## 数据契约

最终 probe 数据必须是一个 `.npz`：

```text
encoding      [N, D]   frozen encoder fusion 输出的 valid-token mean-pool
d             [N]      帧间变化距离
scenario_id   [N]      按 scenario 切 train/val/test，防止相邻帧泄漏
ego_features  [N, K]   ego 速度、加速度、转角、yaw rate 等对照特征
sample_id     [N]      诊断用，必须能唯一对应 d
```

`sample_id` 是最关键的 join key。`encoding` 和 `d` 必须来自同一帧。

## Step 1：从 adjacent_traj_l2_output 生成 d_labels.csv

输入：

```bash
0hzxcode/adjacent_traj_l2_output/per_pair_overlap_l2.csv
```

这个文件由 `0hzxcode/analyze_adjacent_traj_l2.py` 从 val14 闭环仿真日志生成。它已经包含每对相邻规划帧的重叠轨迹距离。

生成标签：

```bash
/home/ubuntu/anaconda3/envs/dp/bin/python 0hzxcode/m_probe/make_d_labels_from_adjacent_traj_l2.py \
  --input 0hzxcode/adjacent_traj_l2_output/per_pair_overlap_l2.csv \
  --require-time-aligned \
  --output 0hzxcode/m_probe_output/d_labels.csv
```

默认使用：

```text
d = perstep_max_m
```

这对应计划里的定义：

```text
d = max_t ||x0_current[t, :2] - shifted_x0_previous[t, :2]||_2
```

输出格式：

```csv
sample_id,d,scenario_id,source,d_metric,scenario_type,log_name,scenario_name,old_iteration,new_iteration,...
2021.06.07...__15416947...__iter_000001,0.0233,2021.06.07...__15416947...,analyze_adjacent_traj_l2,perstep_max_m,...
```

这里的 `sample_id` 规则是：

```text
<log_name>__<scenario_name>__iter_<new_iteration:06d>
```

注意：这个 `sample_id` 对应 val14 闭环日志帧，不是 `exp/cache/mini/*.npz` 文件名。

## Step 2：导出同一批 sample_id 的 encoding

这是最容易出错的一步。

`d_labels.csv` 来自 val14 闭环日志帧，所以 encoding 也必须来自同一批帧，并且使用相同的 `sample_id`：

```text
<log_name>__<scenario_name>__iter_<new_iteration:06d>
```

不能把这个 `d_labels.csv` 直接配 `exp/cache/mini + mini.json` 使用，因为 `mini.json` 里的样本名类似：

```text
us-nv-las-vegas-strip_a8c200dd8c495d2a.npz
```

两者不是同一帧标识，强行 join 会得到 0 行或错误标签。

### 推荐方式：直接从 val14 simulation log 导出

运行：

```bash
/home/ubuntu/anaconda3/envs/dp/bin/python 0hzxcode/m_probe/make_probe_dataset_from_val14_logs.py \
  --labels-csv 0hzxcode/m_probe_output/d_labels.csv \
  --checkpoint checkpoints/model.pth \
  --config checkpoints/args.json \
  --output 0hzxcode/m_probe_output/probe_dataset.npz \
  --device cuda \
  --cpu-log-load

/home/ubuntu/anaconda3/envs/dp/bin/python 0hzxcode/m_probe/make_probe_dataset_from_val14_logs.py \
  --labels-csv 0hzxcode/m_probe_output/d_labels.csv \
  --checkpoint checkpoints/model.pth \
  --config checkpoints/args.json \
  --output 0hzxcode/m_probe_output/probe_dataset.npz \
  --device cpu \
  --cpu-log-load
```

如果只是冒烟验证：

```bash
/home/ubuntu/anaconda3/envs/dp/bin/python 0hzxcode/m_probe/make_probe_dataset_from_val14_logs.py \
  --labels-csv 0hzxcode/m_probe_output/d_labels.csv \
  --checkpoint checkpoints/model.pth \
  --config checkpoints/args.json \
  --output /tmp/probe_dataset_val14_smoke.npz \
  --device cpu \
  --max-logs 1 \
  --max-rows 30 \
  --cpu-log-load
```

脚本内部对每一帧做：

1. 从 `SimulationHistorySample` 重建 21 帧 `SimulationHistoryBuffer`；
2. 用 `DataProcessor.observation_adapter(...)` 生成 planner 输入；
3. 用 `config.observation_normalizer(inputs)` 归一化；
4. 跑 frozen encoder；
5. 对 `encoder_outputs["encoding"]` 做 valid-token mean-pool；
6. 保存：

```text
sample_id    = <log_name>__<scenario_name>__iter_<iteration:06d>
scenario_id  = <log_name>__<scenario_name>
encoding
ego_features
d
```

输出：

```bash
0hzxcode/m_probe_output/probe_dataset.npz
```

同时会写出：

```bash
0hzxcode/m_probe_output/probe_dataset.summary.json
```

注意：encoder 的 `time_len=21`，所以 `new_iteration < 20` 的早期帧会被跳过。summary 里的 `skipped_short_history` 会记录数量。

### 如果已经有与 d_labels 对齐的 `.npz` 缓存

要求：

- `data-list` 里的条目必须能和 `d_labels.csv` 的 `sample_id` 对上；
- 或者条目的 basename 能和 `sample_id` 对上；
- `d_labels.csv` 至少包含 `sample_id,d,scenario_id`。

运行：

```bash
/home/ubuntu/anaconda3/envs/dp/bin/python 0hzxcode/m_probe/make_probe_dataset.py \
  --data-dir <matched_cache_dir> \
  --data-list <matched_list.json> \
  --checkpoint checkpoints/model.pth \
  --config checkpoints/args.json \
  --labels-csv 0hzxcode/m_probe_output/d_labels.csv \
  --output 0hzxcode/m_probe_output/probe_dataset.npz \
  --device cuda \
  --max-samples 5000
```

如果只是检查 encoder 导出链路，可以临时去掉 `--labels-csv`，但导出的 `d` 会是 `NaN`，不能跑最终 probe：

```bash
/home/ubuntu/anaconda3/envs/dp/bin/python 0hzxcode/m_probe/make_probe_dataset.py \
  --data-dir exp/cache/mini \
  --data-list mini.json \
  --checkpoint checkpoints/model.pth \
  --config checkpoints/args.json \
  --output 0hzxcode/m_probe_output/probe_dataset_unlabeled.npz \
  --device cuda \
  --max-samples 5000
```

## Step 3：运行 M-probe

有了真实对齐的 `probe_dataset.npz` 后运行：

```bash
/home/ubuntu/anaconda3/envs/dp/bin/python 0hzxcode/m_probe/run_encoding_d_probe.py \
  --input 0hzxcode/m_probe_output/probe_dataset.npz \
  --output-dir 0hzxcode/m_probe_output
```

输出：

```text
0hzxcode/m_probe_output/probe_results.json
0hzxcode/m_probe_output/probe_report.md
```

默认评估内容：

- `encoding -> d` Ridge 回归；
- `ego_features -> d` Ridge 回归；
- `encoding -> hard/easy` logistic probe；
- `ego_features -> hard/easy` logistic probe。

默认 hard 标签：

```text
hard = d > train split 的 d P90
```

也可以指定绝对阈值：

```bash
--hard-threshold 3.0
```

## PASS / FAIL 标准

默认通过条件：

```text
encoding test RMSE 相对 ego_features 降低 >= 15%
encoding test Spearman >= 0.30
encoding hard/easy AUROC >= 0.70
encoding hard recall >= 0.85
```

通过：

```text
GO: train the learned gate
```

不过：

```text
STOP: keep fixed t_s warm-start; do not train learned gate yet
```

## 结果解读

重点看 `probe_report.md` 的 test metrics：

```text
| feature | RMSE | MAE | Spearman | AUROC | AP | hard recall | false easy |
```

判断逻辑：

- `encoding` 的 RMSE/MAE 应显著低于 `ego_features`；
- `encoding` 的 Spearman 应明显为正，说明排序能力可用；
- `encoding` 的 AUROC/AP 应优于 ego 运动学；
- `false easy` 越低越好，因为 hard 帧误判为 easy 会让低 `t_s` 太激进。

## 常见错误

### 1. labels CSV not found

说明 `d_labels.csv` 还没生成。先运行 Step 1。

### 2. no valid labeled rows remain

通常是 `probe_dataset.npz` 里的 `d` 全是 `NaN`，或者 `d_labels.csv` 和 encoding 缓存的 `sample_id` 对不上。

### 3. 用 mini.json 直接配 val14 d_labels

不成立。`mini.json` 样本 ID 和 val14 闭环日志帧 ID 不是同一命名空间。

### 4. 按帧随机切 train/val/test

不允许。必须按 `scenario_id` 切分，否则相邻帧泄漏会让 probe 虚高。

## 当前状态

已经可以从 `adjacent_traj_l2_output/per_pair_overlap_l2.csv` 生成 `d_labels.csv`。

下一步真正要补的是：从 val14 连续 replay 的同一批帧导出 encoding，并使用和 `d_labels.csv` 相同的 `sample_id`。
