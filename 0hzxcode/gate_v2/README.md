# gate_v2 精简自适应门控

ego 运动学 + 邻车 (+ 可选 prev_d) -> `d_hat` -> `(t_start, steps)` warm-start。

## 训练

```bash
conda activate dp
cd /home/ubuntu/code/hezexiang/Diffusion-Planner

# perstep_max_m 口径（默认）
CUDA_VISIBLE_DEVICES=0 python 0hzxcode/gate_v2/train_gate.py \
  --d-column perstep_max_m \
  --output-dir 0hzxcode/gate_v2_output/runs/ego_nbr_perstep

# normalized_l2_xy 口径（更平滑，推荐对比）
CUDA_VISIBLE_DEVICES=0 python 0hzxcode/gate_v2/train_gate.py \
  --d-column normalized_l2_xy \
  --output-dir 0hzxcode/gate_v2_output/runs/ego_nbr_norml2

# + prev_d（闭环需缓存上一帧 d_hat，有分布漂移风险）
CUDA_VISIBLE_DEVICES=0 python 0hzxcode/gate_v2/train_gate.py \
  --enable-groups ego_history neighbor_agents prev_d \
  --d-column perstep_max_m \
  --output-dir 0hzxcode/gate_v2_output/runs/ego_nbr_prevd
```

档位边界与 hard_threshold 均按 **所选 d-column 的 train 分位数** 自动计算，不再硬编码。

## 决策导向评测（无需重训）

```bash
python 0hzxcode/gate_v2/evaluate_gate.py \
  --checkpoint 0hzxcode/gate_v2_output/runs/ego_nbr_perstep/best.pt \
  --output-dir 0hzxcode/gate_v2_output/runs/ego_nbr_perstep/eval
```

主指标：**Spearman**、**reuse@HR95**、**分箱校准单调**、**adj level acc**。Pearson/原始 RMSE 对重尾 d 仅作参考。

## 口径对比表

```bash
python 0hzxcode/gate_v2/compare_d_columns.py \
  --runs perstep=0hzxcode/gate_v2_output/runs/ego_nbr_perstep \
           norml2=0hzxcode/gate_v2_output/runs/ego_nbr_norml2 \
           prevd=0hzxcode/gate_v2_output/runs/ego_nbr_prevd \
  --output 0hzxcode/gate_v2_output/d_column_comparison.md
```

## 模块

| 文件 | 作用 |
|---|---|
| `gate_metrics.py` | 决策导向指标（Spearman、log回归、校准、复用曲线） |
| `evaluate_gate.py` | 对已训 checkpoint 出报告/曲线 |
| `gate_model.py` | LiteGate |
| `d_to_ops.py` | d_hat -> t_start/steps（边界从 checkpoint 读） |
| `safety.py` | hard recall + 邻车保守上调 |
| `inference.py` | 闭环 GateWarmStartController |
