# 修复时间轴滚动后跑完整 val14

本文用于评测以下两项 warm-start 修复后的闭环效果：

1. 上一帧缓存轨迹沿时间轴向前滚动一个 0.1 s planning tick，并在末端外推补点。
2. 将上一帧局部轨迹转换到当前 ego frame 时，正确处理非零全局航向下的平移向量。

本次评测保持 gate checkpoint、DPM-Solver 步数映射和安全回退逻辑不变，重点观察时空对齐修复能否改善 Comfort、TTC、碰撞和转弯场景。

## 1. 评测配置

| 项目 | 值 |
| --- | --- |
| 工作目录 | `/home/ubuntu/code/hezexiang/Diffusion-Planner` |
| Python | `/home/ubuntu/anaconda3/envs/dp/bin/python` |
| challenge | `closed_loop_nonreactive_agents`（CLS-NR） |
| scenario filter | `val14` |
| 场景数 | 1118 |
| planner checkpoint | `checkpoints/model.pth` |
| planner args | `checkpoints/args.json` |
| gate checkpoint | `0hzxcode/gate_v2_output/runs/ego_nbr_perstep/best.pt` |
| warm-start | 启用 |
| worker | `sequential` |
| CPU affinity | CPU 0 |

选择 `sequential` 是为了规避本机并发仿真时出现过的 native FPE。上一轮完整 val14 顺序评测耗时约 6 小时 42 分，应预留约 7 小时。

### 为什么没有 `num_workers`

nuPlan 的仿真 worker 没有统一的 `num_workers` 参数，参数名取决于 worker 类型：

| `SIM_WORKER` | 并发参数 | 本教程设置 |
| --- | --- | --- |
| `sequential` | 无；固定单 worker 顺序执行 | 使用该模式，因此没有 `num_workers` |
| `single_machine_thread_pool` | `worker.max_workers` | 脚本环境变量 `THREAD_POOL_MAX_WORKERS` |
| `ray_distributed` | `worker.threads_per_node` | 脚本环境变量 `RAY_THREADS_PER_NODE` |

本教程命令中的：

```bash
SIM_WORKER=sequential
CPU_AFFINITY=0
```

表示所有 scenario 顺序执行，并把进程绑定在 CPU 0；有效仿真并发数就是 1。

如果要尝试 4 个本机线程，应改成：

```bash
SIM_WORKER=single_machine_thread_pool \
THREAD_POOL_MAX_WORKERS=4 \
CPU_AFFINITY=0-7 \
bash sim_gate_warmstart_runner.sh
```

对应的 Hydra override 是 `worker.max_workers=4`。但本机此前使用 `single_machine_thread_pool` 跑完整 val14 时出现过 native FPE，因此本次验证修复效果仍推荐 `sequential`，不要为了加速同时改变 worker 模式。

Ray 模式对应：

```bash
SIM_WORKER=ray_distributed \
RAY_THREADS_PER_NODE=6 \
RAY_GPUS_PER_SIM=0.15 \
CPU_AFFINITY=0-7 \
bash sim_gate_warmstart_runner.sh
```

这里的 `scenario_builder.max_workers` 和训练配置中的 `data_loader.num_workers` 不是 scenario 仿真并发数，不应拿来替代上述 worker 参数。

## 2. 本次代码变更

修改文件：

```text
0hzxcode/gate_v2/warmstart.py
0hzxcode/gate_v2/test_gate_v2.py
```

关键逻辑：

```text
时间滚动：new[k] = old[k + 1]

坐标变换：
p_cur = R(h_prev - h_cur) * p_prev
      + R(-h_cur) * (anchor_prev - anchor_cur)
```

缓存 ego 和 token 匹配成功的邻车轨迹滚动一格；未匹配邻车继续根据当前观测做匀速外推。末端 xy 使用最后两点线性外推，heading 使用包角后的最后一个角度变化量外推。

运行全量评测前记录当前版本，避免之后无法确认评测对应的代码：

```bash
cd /home/ubuntu/code/hezexiang/Diffusion-Planner

git rev-parse HEAD
git diff -- 0hzxcode/gate_v2/warmstart.py 0hzxcode/gate_v2/test_gate_v2.py
```

如果评测开始前又修改了这两个文件，应重新执行第 4 节的测试。

## 3. 运行前检查

进入仓库：

```bash
cd /home/ubuntu/code/hezexiang/Diffusion-Planner
```

检查模型、gate、nuPlan devkit、数据和地图：

```bash
test -x /home/ubuntu/anaconda3/envs/dp/bin/python
test -f checkpoints/model.pth
test -f checkpoints/args.json
test -f 0hzxcode/gate_v2_output/runs/ego_nbr_perstep/best.pt
test -d nuplan-devkit/nuplan
test -d /media/ubuntu/T9/dataset/nuplan-v1.1/trainval
test -d /media/ubuntu/T9/dataset/maps

find /media/ubuntu/T9/dataset/nuplan-v1.1/trainval \
  -maxdepth 1 -type f -name '*.db' -print -quit
```

最后一条命令必须打印至少一个 `.db` 文件。本机 `/home/ubuntu/data/hezexiang/nuplan/dataset` 虽然目录存在，但当前检查没有发现 trainval DB，因此本教程显式使用已经验证有 DB 的 `/media/ubuntu/T9/dataset`。

检查磁盘空间和是否已有仿真进程：

```bash
df -h /home/ubuntu/code/hezexiang/Diffusion-Planner/exp
pgrep -af 'run_simulation.py|sim_gate_warmstart_runner.sh' || true
nvidia-smi
```

不要同时启动第二个 gate warm-start 评测。`sim_gate_warmstart_runner.sh` 每次启动都会清空：

```text
0hzxcode/gate_v2_output/val14_closedloop/frames.jsonl
```

## 4. 全量前的快速验证

### 4.1 单元测试

```bash
env -u PYTHONPATH \
  /home/ubuntu/anaconda3/envs/dp/bin/python \
  -m pytest -q 0hzxcode/gate_v2/test_gate_v2.py
```

当前预期：

```text
13 passed
```

### 4.2 planner + gate 集成冒烟

```bash
env -u PYTHONPATH \
  /home/ubuntu/anaconda3/envs/dp/bin/python \
  0hzxcode/gate_v2/smoke_planner_integration.py
```

当前预期结尾：

```text
warmstart nfe 5
planner+gate integration smoke passed
```

### 4.3 语法检查

```bash
env -u PYTHONPATH \
  /home/ubuntu/anaconda3/envs/dp/bin/python \
  -m py_compile \
  0hzxcode/gate_v2/warmstart.py \
  0hzxcode/gate_v2/test_gate_v2.py
```

三项都通过后再启动完整 val14。

## 5. 启动完整 val14

### 推荐：前台运行并保存终端日志

```bash
cd /home/ubuntu/code/hezexiang/Diffusion-Planner

mkdir -p 0hzxcode/gate_v2_output/val14_closedloop

set -o pipefail
NUPLAN_DATA_ROOT=/media/ubuntu/T9/dataset \
NUPLAN_MAPS_ROOT=/media/ubuntu/T9/dataset/maps \
SPLIT=val14 \
CHALLENGE=closed_loop_nonreactive_agents \
SIM_WORKER=sequential \
ENABLE_SIMULATION_PROGRESS_BAR=true \
CPU_AFFINITY=0 \
TQDM_DISABLE=1 \
PYTHONFAULTHANDLER=1 \
bash sim_gate_warmstart_runner.sh 2>&1 | \
tee 0hzxcode/gate_v2_output/val14_closedloop/time_roll_fix_run.log
```

脚本会自动设置或覆盖：

```text
planner=diffusion_planner
planner.diffusion_planner.gate_ckpt_path=.../ego_nbr_perstep/best.pt
planner.diffusion_planner.enable_warmstart=true
scenario_builder=nuplan
scenario_filter=val14
worker=sequential
~main_callback.metric_summary_callback
```

实验目录会带启动时间戳，形如：

```text
exp/exp/simulation/closed_loop_nonreactive_agents/
  diffusion_planner/val14_gate_warmstart/
  diffusion_planner_release/model_YYYY-MM-DD-HH-MM-SS/
```

### 后台运行方式

需要关闭终端后继续运行时使用：

```bash
cd /home/ubuntu/code/hezexiang/Diffusion-Planner
mkdir -p 0hzxcode/gate_v2_output/val14_closedloop

nohup env \
  NUPLAN_DATA_ROOT=/media/ubuntu/T9/dataset \
  NUPLAN_MAPS_ROOT=/media/ubuntu/T9/dataset/maps \
  SPLIT=val14 \
  CHALLENGE=closed_loop_nonreactive_agents \
  SIM_WORKER=sequential \
  ENABLE_SIMULATION_PROGRESS_BAR=true \
  CPU_AFFINITY=0 \
  TQDM_DISABLE=1 \
  PYTHONFAULTHANDLER=1 \
  bash sim_gate_warmstart_runner.sh \
  > 0hzxcode/gate_v2_output/val14_closedloop/time_roll_fix_run.log 2>&1 &

echo $! | tee 0hzxcode/gate_v2_output/val14_closedloop/time_roll_fix_run.pid
```

## 6. 运行中监控

查看主日志：

```bash
tail -f 0hzxcode/gate_v2_output/val14_closedloop/time_roll_fix_run.log
```

查看进程和 GPU：

```bash
pgrep -af 'run_simulation.py|sim_gate_warmstart_runner.sh'
nvidia-smi
```

查看已记录的 planning frame 数：

```bash
wc -l 0hzxcode/gate_v2_output/val14_closedloop/frames.jsonl
```

注意：逐帧数不是已完成 scenario 数，不能用它直接估计 `1118` 个场景的完成比例。scenario 进度以主日志为准。

出现异常时先保存以下内容：

```bash
tail -200 0hzxcode/gate_v2_output/val14_closedloop/time_roll_fix_run.log
cp 0hzxcode/gate_v2_output/val14_closedloop/frames.jsonl \
   /tmp/frames-time-roll-fix-partial.jsonl
```

不要直接重新启动脚本，否则现有 `frames.jsonl` 会被删除。

## 7. 判断是否完整结束

进程退出后检查 shell 退出码；前台 `tee` 运行方式因为设置了 `set -o pipefail`，脚本失败会返回非零。

检查是否生成完整结果：

```bash
find exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner \
  -path '*val14_gate_warmstart*' \
  -name 'runner_report.parquet' \
  -printf '%T@ %p\n' | sort -nr | head

find exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner \
  -path '*val14_gate_warmstart*' \
  -path '*/aggregator_metric/*.parquet' \
  -printf '%T@ %p\n' | sort -nr | head
```

应当同时存在最新的：

- `runner_report.parquet`
- `aggregator_metric/*.parquet`
- `metrics/*.parquet`
- `.nuboard` 文件
- `0hzxcode/gate_v2_output/val14_closedloop/frames.jsonl`

主日志应显示 1118 个场景完成且没有失败。如果场景数不足，不要把该结果与完整基线比较。

## 8. 生成本次独立汇总

不要使用汇总脚本的默认输出文件名，否则会覆盖旧的 `0实验结果/DP-val14-gate-warmstart.md`。

```bash
cd /home/ubuntu/code/hezexiang/Diffusion-Planner

env -u PYTHONPATH \
  /home/ubuntu/anaconda3/envs/dp/bin/python -B \
  0hzxcode/gate_v2/summarize_val14_gate.py \
  --out '0实验结果/DP-val14-gate-warmstart-time-roll-fix.md'
```

汇总脚本会选择 `val14_gate_warmstart` 下修改时间最新的 aggregator，并输出：

- CLS-NR 总分及相对 89.59 基线的变化。
- 平均 NFE 和相对 11 NFE 基线的加速比。
- 平均 decoder 延迟。
- 主动判难率和被动回退率。
- runner 平均 `compute_trajectory` 时间。

运行后检查汇总引用的是本次最新 aggregator：

```bash
sed -n '1,100p' '0实验结果/DP-val14-gate-warmstart-time-roll-fix.md'
```

如果其中的 aggregator 路径不是本次时间戳目录，不要使用该分数，应通过 `--exp-root` 指向正确实验根目录后重新汇总。

## 9. 需要重点比较的指标

对照结果：

| 实验 | CLS-NR | 平均 NFE | decoder 延迟 |
| --- | ---: | ---: | ---: |
| 原始 Diffusion-Planner 基线 | 89.59 | 11.00 | 19.00 ms |
| 修复前 gate warm-start | 85.23 | 4.93 | 12.18 ms |
| 本次时空对齐修复 | 待填写 | 待填写 | 待填写 |

修复前主要掉点：

| 指标 | 原始基线 | 修复前 warm-start |
| --- | ---: | ---: |
| NC | 95.75 | 93.25 |
| TTC | 90.43 | 87.03 |
| EP | 94.16 | 92.70 |
| Comfort | 95.08 | 91.50 |

本次优先检查：

1. Comfort 是否恢复：时间滚动修复应减少速度、加速度和 jerk 的帧间滞后。
2. TTC 和 NC 是否恢复：正确的邻车时间/坐标对齐应改善交互轨迹。
3. `starting_right_turn`、`starting_left_turn`、`high_lateral_acceleration` 是否改善：非零航向平移修复对这些类别最敏感。
4. `following_lane_with_lead` 和 `low_magnitude_speed` 是否改善：时间轴错位曾可能导致跟车距离及低速轨迹滞后。
5. 平均 NFE 是否仍接近 4.93：本次没有修改 gate 档位和 steps 映射，若 NFE 大幅改变，需要检查被动回退率是否变化。

## 10. 结果记录模板

评测结束后填写：

```text
开始时间：
结束时间：
wall-clock：
git commit：
是否有未提交 diff：
实验目录：
aggregator 文件：
成功 / 失败场景数：

CLS-NR：
平均 NFE：
平均 decoder ms：
runner mean compute_trajectory ms：
主动 full fallback 率：
被动 fallback 率：

NC：
TTC：
EP：
Comfort：

starting_right_turn：
starting_left_turn：
high_lateral_acceleration：
following_lane_with_lead：
low_magnitude_speed：
```

## 11. 解释结果时的限制

- 本次同时修复了时间滚动和非零航向坐标变换，因此完整 val14 只能测二者的组合收益，不能区分单项贡献。
- `worker=sequential` 的 wall-clock 不能与并发基线直接比较；优先比较 NFE、decoder 延迟和 runner `compute_trajectory`。
- 如果 CLS-NR 仍明显低于基线，应继续检查 warm-start 的 index-0 首次前向污染、过小 `t_start` 和 2-step 档位，而不是立即把问题归因于短区间 4-step solver。
- full val14 是闭环最终结论；单元测试和集成冒烟只证明实现链路正确，不代表规划分数一定提升。
