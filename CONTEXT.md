# Diffusion-Planner × Falcon 帧间复用

将 Falcon（训练无关的部分去噪复用加速）适配到 nuPlan 闭环下的 Diffusion-Planner，目标是闭环分数不降的前提下大幅降低去噪 NFE。方法定位为"面向驾驶的适配版"，非忠实移植。

## Language

**帧间复用（warm-start）**:
利用上一规划帧的去噪结果初始化当前帧的去噪过程，从中间噪声水平而非纯高斯噪声开始采样。
_Avoid_: 缓存复用、热启动（口语可用，文档统一用"帧间复用"）

**参考轨迹（reference trajectory）**:
上一帧预测轨迹中与当前帧时间窗重叠的部分，经坐标对齐后作为当前帧的预期答案，用于判定复用是否安全。
_Avoid_: 历史轨迹（指 ego 已行驶过的轨迹，是另一概念）

**帧间变化距离（inter-frame trajectory change）**:
相邻两帧 ego 轨迹在重叠段（79/80 点）上、物理空间（米）逐点欧氏距离的 max。衡量"当前帧最优计划相对上一帧预测移动了多少"。是自适应门控的回归目标，离线用完整去噪测量；与阈值回退的 d_meas 同量纲同语义，后者在线用一步估计测量——二者是同一个量的两种估计。
_Avoid_: 归一化欧氏距离（口语可用；正式口径为物理空间米、max 聚合，非归一化空间）

**一步估计（one-step estimation）**:
以当前观测为条件，模型对一个带噪轨迹直接给出的干净轨迹预测（x_start 模型一次前向即得）。
_Avoid_: Tweedie 估计（原文术语，本项目模型类型下两者等价）

**阈值回退（threshold fallback）**:
一步估计与参考轨迹的距离超过阈值 ε 时，放弃帧间复用，退回从纯噪声完整去噪。安全关键机制，突发场景（前车急刹、红灯跳变）依赖它切换模态。
_Avoid_: 探索（Falcon 原文的随机探索率 δ 已被本机制取代，不引入）

**重加噪初始化（re-noise warm-start）**:
帧间复用的具体构造方式：缓存上一帧最终干净轨迹，坐标对齐后用前向扩散公式加噪到中间噪声水平，从该水平开始少步去噪。不缓存去噪中间状态。
_Avoid_: latent buffer（Falcon 原文机制，本项目不保留）

**场景门控（scene gate）**:
挂在冻结 planner 的场景嵌入（encoder fusion 输出 `encoding`）上的轻量 MLP，回归预测当前帧的帧间变化距离 d_hat，据此自适应设定重加噪起始水平 t_s（最高档 = 完整去噪）。沿用 BLUE 的"冻结主干 + 现成 hidden state 上训练小 gate"思路；标签来自离线统计，无需人工标注。详见 `0docs/adr/0002`。**当前未实现**；保留为历史/对照方案。
_Avoid_: 语言门控（BLUE 原文 gate 决定"是否生成语言"；本项目 gate 回归 d_hat → t_s，是回归而非二分类）

**轻量门控（lite gate）**:
gate_v2 已实现版本：挂在**原始观测**（ego 运动学 + 邻车 token）上的 <0.1M MLP，回归 d_hat → 分箱映射 (t_s, steps)。不读 encoder `encoding`。详见 `0docs/adr/0003`。
_Avoid_: 与 scene gate 混称（二者输入不同）

**主动判难（active hard routing）**:
gate 前置判定 d_hat 超过 score 阈值时，直接完整去噪、跳过 warm-start 尝试。
_Avoid_: 被动回退（后者是一步估计失败后的兜底）

**被动回退（passive fallback）**:
warm-start 已构造后，一步估计测得 d_meas 超过 hard_threshold_m 时，放弃复用并完整去噪。
_Avoid_: 主动判难（前者在 warm-start 之前就短路）

**适配版核心三件套**:
帧间复用 + 一步估计 + 阈值回退。Falcon 原文的多帧 latent buffer、温度 softmax 采样、探索率 δ 均不保留。

**叙事口径**:
论文报告"模型推理（去噪）延迟 / NFE"，不是端到端延迟（数据处理 85ms 是纯工程瓶颈，与方法无关）；闭环为同步仿真，分数与推理速度无关，结论形式为"分数不降 + NFE 降 N 倍"。
