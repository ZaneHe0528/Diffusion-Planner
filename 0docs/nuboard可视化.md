`run_nuboard.ipynb` 是 **nuPlan 仿真结果可视化工具 nuBoard** 的启动脚本，用来查看 `sim_diffusion_planner_runner.sh` 跑完闭环仿真后的指标和场景回放。

## 整体流程

```
1. 先跑仿真  →  bash sim_diffusion_planner_runner.sh
2. 再配置 notebook  →  填路径
3. 依次运行所有 cell  →  浏览器打开 nuBoard
```

---

## 前置条件

1. 已完成环境安装（nuplan-devkit + diffusion_planner）
2. 已跑过闭环仿真，结果目录下有 `.nuboard` 文件

你当前已有一次仿真结果，例如：

```
/home/ubuntu/code/hezexiang/Diffusion-Planner/exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14/diffusion_planner_release/model_2026-06-15-16-34-20/
```

其中包含 `nuboard_*.nuboard` 文件。

---

## 使用步骤

### 1. 修改配置（Cell 2）

把占位符换成你机器上的**绝对路径**：

| 变量 | 含义 | 你当前的参考值 |
|------|------|----------------|
| `RESULT_FOLDER` | 仿真结果目录 | `/home/ubuntu/code/hezexiang/Diffusion-Planner/exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14/diffusion_planner_release/model_2026-06-15-16-34-20` |
| `NUPLAN_DEVKIT_ROOT` | nuplan-devkit 根目录 | `/home/ubuntu/code/hezexiang/Diffusion-Planner/nuplan-devkit` |
| `NUPLAN_DATA_ROOT` | 数据集根目录 | `/home/ubuntu/data/hezexiang/nuplan/dataset` |
| `NUPLAN_MAPS_ROOT` | 地图目录 | `/home/ubuntu/data/hezexiang/nuplan/dataset/maps` |
| `NUPLAN_EXP_ROOT` | 实验输出根目录 | `/home/ubuntu/code/hezexiang/Diffusion-Planner/exp` |

`RESULT_FOLDER` 可以填：
- **单次实验目录**（上面那种带时间戳的路径），或
- **更高层目录**（如 `.../val14`），notebook 会自动递归查找含 `.nuboard` 的子目录

### 2. 检查 CONFIG_PATH

Notebook 里写的是：

```python
CONFIG_PATH = '../nuplan-devkit/nuplan/planning/script/config/nuboard'
```

Jupyter 的工作目录一般是项目根目录 `Diffusion-Planner/`，此时 `../nuplan-devkit` 会指到**上一级目录**，路径不对。

应改为：

```python
CONFIG_PATH = 'nuplan-devkit/nuplan/planning/script/config/nuboard'
```

### 3. 依次运行所有 Cell

在项目根目录启动 Jupyter：

```bash
cd /home/ubuntu/code/hezexiang/Diffusion-Planner
jupyter lab run_nuboard.ipynb
# 或
jupyter notebook run_nuboard.ipynb
```

按顺序执行 Cell 0 → 4。最后一个 Cell 会启动 nuBoard Web 服务。

### 4. 打开浏览器

默认端口是 **6599**，在浏览器访问：

```
http://localhost:6599
```

如果是远程服务器，需要做端口转发，例如：

```bash
ssh -L 6599:localhost:6599 user@your-server
```

---

## nuBoard 三个页面

| 页面 | 作用 |
|------|------|
| **Overview** | 各场景类型、各 planner 的聚合指标汇总 |
| **Histograms** | 指标分布直方图（碰撞、限速、舒适度等） |
| **Scenarios** | 单个场景仿真回放 + 逐帧指标 |

详细指标说明见项目内 [`docs/nuplan评测指标.md`](docs/nuplan评测指标.md)。

---

## 常见问题

**Q: 找不到 `.nuboard` 文件？**  
说明仿真还没跑完，或 `RESULT_FOLDER` 填错了。可用下面命令确认：

```bash
find /home/ubuntu/code/hezexiang/Diffusion-Planner/exp -name "*.nuboard"
```

**Q: 端口被占用？**  
在 Cell 3 里改 `port_number=6599` 为其他端口（如 `6600`）。

**Q: 不用 Jupyter 能启动吗？**  
可以，命令行等价方式：

```bash
export NUPLAN_DEVKIT_ROOT="/home/ubuntu/code/hezexiang/Diffusion-Planner/nuplan-devkit"
export NUPLAN_DATA_ROOT="/home/ubuntu/data/hezexiang/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/ubuntu/data/hezexiang/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/home/ubuntu/code/hezexiang/Diffusion-Planner/exp"

python nuplan-devkit/nuplan/planning/script/run_nuboard.py \
  simulation_path='["/path/to/your/simulation/result/folder"]' \
  port_number=6599
```

---

如果你愿意，我可以直接帮你把 notebook 里的路径改成你当前环境的正确值（含 `CONFIG_PATH` 修正）。