我先把仓库里的 nuPlan 评测链路和扩散采样实现看了一遍（`diffusion_planner/planner/planner.py`、`model/module/decoder.py`、`diffusion_utils/sampling.py`、`sim_diffusion_planner_runner.sh`），下面给你完整的分析和判断。

## 一、nuPlan 闭环评测是怎么实现的

入口是 `sim_diffusion_planner_runner.sh` 调用 nuplan-devkit 的 `run_simulation.py`，挑战类型是 `closed_loop_nonreactive_agents` 或 `closed_loop_reactive_agents`。机制如下：

1. **仿真循环（同步、10Hz）**：每个场景 15 秒，仿真器以 0.1s 为步长迭代。每一步调用一次 planner 的 `compute_planner_trajectory()`，即你的模型每个场景会被调用约 150 次。本仓库的实现在这里：

```117:130:diffusion_planner/planner/planner.py
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)

        inputs = self.observation_normalizer(inputs)        
        _, outputs = self._planner(inputs)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory
```

2. **闭环的"闭"**：planner 输出 8s 的未来轨迹，但 ego 不直接瞬移到轨迹上，而是由两级控制器（LQR 跟踪器 + 运动学自行车模型）跟踪该轨迹推进 0.1s，下一步 planner 看到的是被自己上一步决策影响后的新状态。nonreactive 模式下其他车按 log 回放；reactive 模式下其他车由 IDM 模型驱动、会对 ego 做出反应。

3. **打分**：仿真结束后离线计算指标（无责碰撞、可行驶区域、TTC、进度、舒适度、限速等），加权聚合成 CLS-NR / CLS-R 分数，用 nuBoard 查看。

4. **一个对你至关重要的事实**：nuPlan 闭环是**同步仿真、模拟时间**——仿真器会无限等待 planner 算完，推理快慢**不影响闭环分数**。planner 每步的耗时会被记录在 runner report 里，可以单独统计。

## 二、Falcon 的创新点能否用在这里

**能，而且匹配度相当好。** 逐条对一下 Falcon 的假设：

- **时序依赖性假设**：Falcon 依赖"相邻决策步的动作序列高度重叠"。nuPlan 里 replan 间隔 0.1s、预测 horizon 8s，相邻两帧轨迹重叠 79/80，比机器人任务（T_a/T_p 通常是 8/16）的重叠度高得多。假设成立得更强。
- **training-free / 即插即用**：本仓库用 DPM-Solver++（`sampling.py` 里 `steps=10, order=2`），Falcon 论文明确说能和 DPM-Solver 叠加。`DPM_Solver.sample()` 支持 `t_start` 参数，所以"从中间噪声水平开始去噪"在工程上是现成的：把上一帧的部分去噪轨迹（或对上一帧的 x0 重新加噪到水平 k）作为初值，从 `t_start < T` 开始走更少的步数。
- **改动位置集中**：在 `DiffusionPlanner` 类里维护一个 latent buffer（实例属性），在 `decoder.forward` 的推理分支把 `xT = randn(...)` 换成可选的 warm-start 初值即可，不动训练代码。

但有三个驾驶场景特有的问题，是 Falcon 原文（机械臂任务）没有的，需要你处理：

1. **坐标系对齐**：模型在 ego 局部坐标系下预测（且经过 `state_normalizer` 归一化）。两帧之间 ego 动了，上一帧的轨迹必须先反归一化 → SE(2) 变换到当前 ego 系 → 沿时间轴平移 0.1s → 再归一化，才能作为参考/初值。这是最主要的工程量。
2. **联合去噪邻车**：decoder 同时去噪 ego + `predicted_neighbor_num` 个邻车（P 个 token），相邻帧的邻车集合可能变化，需要按 track token 匹配，匹配不上的 token 退回纯噪声初始化。
3. **模态切换风险**：warm-start 会让模型偏向上一帧的模态。驾驶中突发事件（前车急刹、红灯跳变）时这是危险的——Falcon 的 thresholding 机制正是干这个的（参考轨迹与一步估计差距超过 ε 就回退到完整去噪），这个机制你必须保留，不能只做朴素的 warm-start。

**一个需要泼冷水的点**：基线已经只有 10 步 DPM-Solver（约 20 次 NFE），不是 DDPM 的 100 步，加速天花板有限——大概能压到 2~5 步。而且每次调用的总耗时里，编码器 + 数据预处理（`observation_adapter` 里的地图查询往往是 nuPlan 仿真的真正瓶颈）不会被 Falcon 加速。**动手前先 profile 一次**：如果去噪只占单步延迟的 30%，那 5 倍去噪加速也只换来 ~25% 端到端提速，故事就不够好；如果占 70% 以上，这个方向就值得做。

另外注意叙事方式：由于闭环是同步仿真，你的论文/实验结论只能是"**在闭环分数不降的前提下，单步推理延迟降低 N 倍**"，而不是"实时性提高带来分数提升"。

## 三、Bench2Drive+CARLA 还是 nuPlan：我的判断

**选 nuPlan，GPT 的建议是对的**，理由很具体：

| | nuPlan（本仓库） | Bench2Drive + CARLA |
|---|---|---|
| 任务形态 | 轨迹规划（向量化输入），Diffusion-Planner 原生支持 | 端到端（传感器输入），需换成 DiffusionDrive 之类的模型 |
| 工程量 | 改 planner + decoder 两处，buffer + 坐标变换，约几百行 | 整套模型、训练、评测链路全部重来 |
| 基线对照 | 本仓库自带 val14 / test14-hard 基线分数，直接可比 | 需要自己复现扩散基线 |
| 实时性论证 | 同样是同步仿真，靠单独报告延迟 | 同样是同步模式（Leaderboard 协议），也不会因为慢而扣分 |

注意最后一行：你想用 CARLA 的一个潜在动机可能是"闭环跑起来更真实、能体现实时性"，但 Bench2Drive 评测同样运行在同步模式下，推理慢只是让仿真墙钟时间变长，**并不会反映到驾驶分数里**。所以 CARLA 在"证明实时性价值"这件事上相比 nuPlan 没有额外优势，却要多付出一个数量级的工程成本。

## 四、建议的验证路径

1. Profile 基线：单步延迟拆成 数据处理 / encoder / 去噪 三段 → 确认去噪占比够高。
2. 实现朴素 warm-start（上一帧 x0 加噪到固定水平 k，从 `t_start` 去噪）→ 验证：闭环分数不掉、延迟下降。
3. 加上 Falcon 的 thresholding + 回退机制 → 验证：test14-hard（含更多突发场景）上分数恢复到基线水平。
4. 报告：CLS-NR / CLS-R 分数 + NFE + 单步 GPU 延迟，对比 steps=10 基线和直接降步数（steps=2~5 不带 warm-start）的退化曲线。

一个需要你自己确认的点：扩散 planner 的 warm-start/流式去噪思路在驾驶领域已有一些相近工作（如 Streaming Diffusion Policy 在机器人侧），动手前建议快速调研一下 2025–2026 年有没有人已经在 nuPlan 上做过同样的事，确认你的增量（比如 thresholding 机制 + 联合邻车去噪的帧间复用）站得住。

如果你决定走 nuPlan 这条路，我可以帮你先做 profile 或者直接搭 warm-start 的原型。