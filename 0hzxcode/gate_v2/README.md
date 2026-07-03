# gate_v2 精简自适应门控

ego 运动学 + 邻车轻量补充 -> `d_hat` -> `(t_start, steps)` warm-start。

## 训练

```bash
conda activate dp
cd /home/ubuntu/code/hezexiang/Diffusion-Planner

CUDA_VISIBLE_DEVICES=0 python 0hzxcode/gate_v2/train_gate.py \
  --output-dir 0hzxcode/gate_v2_output/runs/ego_nbr_test
```

数据复用 `0hzxcode/gate_output/gate_dataset_chunks/`（无需重导出）。

## 闭环启用 warm-start

```python
planner = DiffusionPlanner(
    config, ckpt_path,
    past_trajectory_sampling=..., future_trajectory_sampling=...,
    device="cuda",
    gate_ckpt_path="0hzxcode/gate_v2_output/runs/ego_nbr/best.pt",
    enable_warmstart=True,
)
```



## 模块


| 文件              | 作用                             |
| --------------- | ------------------------------ |
| `gate_model.py` | LiteGate，仅回归头                  |
| `d_to_ops.py`   | d_hat -> t_start/steps 四档映射    |
| `safety.py`     | hard recall 阈值 + 邻车强交互上调       |
| `warmstart.py`  | ego-shift、再加噪、缓存               |
| `inference.py`  | `GateWarmStartController` 闭环入口 |




## 默认档位（米，perstep_max_m）

P50=0.275 / P75=0.696 / P90=1.385 -> 低/中/高/满四档，对应 steps 2/4/7/10。