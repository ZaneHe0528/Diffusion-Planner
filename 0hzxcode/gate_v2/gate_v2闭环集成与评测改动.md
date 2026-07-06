gate_v2 闭环集成代码已全部落地。受限于本机 nuPlan 数据集目录为空，全量 val14 未能在此环境执行。

## 已完成

### 1. 数据层 — 邻车 track token
- `agent_past_process` 现返回 `selected_neighbor_tokens`（与 slot 对齐的 nuPlan track token 字符串）
- `observation_adapter` 返回 `(model_inputs, selected_neighbor_tokens)`

### 2. Warm-start — token 匹配 + 匀速外推
- [`0hzxcode/gate_v2/warmstart.py`](0hzxcode/gate_v2/warmstart.py)：ego 行缓存 shift；邻车行按 token 命中对齐，未命中用当前观测匀速外推

### 3. 采样器 — 首步注入 + NFE 计数
- `dpm_solver.sample` 支持 `first_model_output`、`nfe_counter`
- `dpm_sampler` 通过 `nfe_holder` 回传 NFE

### 4. Decoder — 一步估计 + 被动回退
- warm-start 路径：在 `t_start` 做验证前向 → `d_meas` 与 `hard_threshold_m` 比较
- 通过：注入首步输出，0 额外 NFE；失败：完整去噪（验证前向计入 NFE）

### 5. Planner 串线 + 诊断
- [`diffusion_planner/planner/planner.py`](diffusion_planner/planner/planner.py)：gate 全流程 + 逐帧 append 到 `0hzxcode/gate_v2_output/val14_closedloop/frames.jsonl`

### 6. 配置与脚本
- [`diffusion_planner/config/planner/diffusion_planner.yaml`](diffusion_planner/config/planner/diffusion_planner.yaml)：`gate_ckpt_path`、`enable_warmstart`
- [`sim_gate_warmstart_runner.sh`](sim_gate_warmstart_runner.sh)：全量 val14 CLS-NR

### 7. 验证
- `python 0hzxcode/gate_v2/test_gate_v2.py` — **全部通过**

### 8. 文档
- [`CONTEXT.md`](CONTEXT.md)：lite gate、主动判难、被动回退
- [`0docs/adr/0003-raw-obs-gate-instead-of-encoding.md`](0docs/adr/0003-raw-obs-gate-instead-of-encoding.md)
- [`0实验结果/DP-val14-gate-warmstart.md`](0实验结果/DP-val14-gate-warmstart.md)

## 未执行（环境限制）

`NUPLAN_DATA_ROOT/nuplan-v1.1/trainval` 为空，闭环仿真无法启动。数据集就绪后：

```bash
conda activate dp
cd /home/ubuntu/code/hezexiang/Diffusion-Planner
bash sim_gate_warmstart_runner.sh
python 0hzxcode/gate_v2/summarize_val14_gate.py
```

将自动汇总：
- CLS-NR vs 基线 **89.59**
- 平均 NFE vs **11**
- decoder 延迟加速比
- 主动判难率 / 被动回退率

## 关键文件一览

| 文件 | 作用 |
|---|---|
| `sim_gate_warmstart_runner.sh` | 全量 val14 评测入口 |
| `0hzxcode/gate_v2/summarize_val14_gate.py` | 结果汇总 |
| `0hzxcode/gate_v2/smoke_planner_integration.py` | 无 nuPlan 数据的集成冒烟（需 GPU 加载 checkpoint） |
| `0hzxcode/gate_v2_output/val14_closedloop/frames.jsonl` | 逐帧 NFE / d_hat / 回退诊断 |