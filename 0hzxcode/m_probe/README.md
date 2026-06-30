# M-probe: encoding -> d 可行性探针

目的：在正式训练 adaptive gate 前，先验证冻结 planner 的 `encoding` 是否真的含有可预测帧间变化距离 `d` 的信号。探针不过，就停止 learned gate 路线，退回固定 `t_s` warm-start。

## 1. 评估已有 probe 数据

输入 `.npz` 至少需要：

- `encoding`: `[N, D]`，推荐为 valid token mean-pool 后的 encoder fusion 输出。
- `d`: `[N]`，标签，必须和阈值安全网的距离定义一致。
- `scenario_id`: `[N]`，用于按 scenario 划分 train/val/test，避免相邻帧泄漏。

可选：

- `ego_features`: `[N, K]`，rule-based/运动学对照特征。
- `sample_id`: `[N]`，诊断用。

运行：

```bash
python 0hzxcode/m_probe/run_encoding_d_probe.py \
  --input 0hzxcode/m_probe_output/probe_dataset.npz \
  --output-dir 0hzxcode/m_probe_output
```

输出：

- `probe_results.json`
- `probe_report.md`

默认门槛：

- encoding Ridge 回归 test RMSE 相对 ego baseline 降低至少 15%；
- encoding 回归 Spearman >= 0.30；
- encoding hard/easy logistic AUROC >= 0.70；
- hard recall >= 0.85。



## 2. 从训练缓存导出 encoding

如果已有训练 `.npz` 缓存和一个标签 CSV，可以先导出 probe 数据：

```bash
python 0hzxcode/m_probe/make_probe_dataset.py \
  --data-dir /path/to/train/cache \
  --data-list /path/to/list.json \
  --checkpoint checkpoints/model.pth \
  --config checkpoints/args.json \
  --labels-csv 0hzxcode/m_probe_output/d_labels.csv \
  --output 0hzxcode/m_probe_output/probe_dataset.npz \
  --device cuda \
  --max-samples 5000

python 0hzxcode/m_probe/make_probe_dataset.py \
  --data-dir exp/cache/mini \
  --data-list mini.json \
  --checkpoint checkpoints/model.pth \
  --config checkpoints/args.json \
  --labels-csv 0hzxcode/m_probe_output/d_labels.csv \
  --output 0hzxcode/m_probe_output/probe_dataset.npz \
  --device cuda \
  --max-samples 5000
```

`labels-csv` 需要列：

- `sample_id`: 与 `data-list` 中的条目一致，或是条目的 basename；
- `d`: 帧间变化距离；
- `scenario_id`: 可选，但强烈建议提供。

注意：如果当前只有闭环轨迹距离 CSV，而没有对应帧的 encoder 输入/输出，这还不能验证 `encoding -> d`。不要用轨迹统计替代 encoding probe。