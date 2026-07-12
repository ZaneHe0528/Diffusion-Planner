给我讲解Diffusion-Planner中DPM-solver的逻辑，并且分析降低NFE次数会造成什么影响，如何减少次数造成的影响?

这里实际使用的是 DPM-Solver++ 二阶多步求解器。它把扩散反向过程看作一个 ODE，用约 10 次 DiT 前向代替传统扩散模型的数百次迭代。

## 1. Diffusion-Planner 在生成什么

模型一次联合生成：

- ego 未来 80 个轨迹点；
- 10 辆邻车未来轨迹；
- 每个点为 \((x, y, \cos\theta, \sin\theta)\)。

训练时，归一化后的真实轨迹 \(x_0\) 按连续 VP-SDE 加噪：

\[
x_t = \alpha(t) x_0 + \sigma(t)\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
\]

其中 \(\beta(t)\) 从 0.1 线性增长到 20。训练随机采样 \(t\in[10^{-3},1]\)，DiT 学习从任意噪声时刻恢复干净轨迹：

\[
\hat{x}_0 = \mu_\theta(x_t, t, C)
\]

当前 checkpoint 使用的是 x_start 参数化，即模型直接预测干净轨迹，而不是预测噪声，见 `diffusion_planner/loss.py:54`。

## 2. 推理调用链

主要代码路径是：

1. Encoder 编码车辆历史、地图和路线条件。
2. 未来轨迹初始化为 `randn * 0.5`，这就是论文中的低温采样：减少随机性，但也会降低多样性。
3. 当前状态作为轨迹 index 0，并在每个求解步骤后被强制写回。
4. DPM-Solver++ 从 \(t=1\) 积分到 \(t=10^{-3}\)。
5. 最后再做一次 `denoise_to_zero`，得到 \(t=0\) 的轨迹。
6. 反归一化并取 ego 的未来 80 点。

入口位于 `diffusion_planner/model/module/decoder.py:103`，求解器配置位于 `diffusion_planner/model/diffusion_utils/sampling.py:55`：

```text
steps=diffusion_steps
order=2
skip_type="logSNR"
method="multistep"
denoise_to_zero=True
```

其含义是：

- `dpmsolver++`：用数据预测 \(\hat{x}_0\) 形式求解；
- `order=2`：二阶；
- `multistep`：复用前两个时间点的模型输出；
- `logSNR`：时间点在 half-logSNR 空间均匀，不是在普通时间 \(t\) 上均匀；
- `lower_order_final=True`：少于 10 步时，最后几步主动降阶以提高稳定性。

DPM-Solver 的核心思想是解析求解扩散 ODE 的线性部分，只数值近似神经网络相关项；DPM-Solver++ 又针对数据预测和引导采样增强了稳定性。DPM-Solver 论文 (https://arxiv.org/abs/2206.00927)、DPM-Solver++ 论文 (https://arxiv.org/abs/2211.01095)。

## 3. 二阶 multistep 的直觉

第一步没有历史信息，只能做一阶更新。之后保存最近两个模型预测：

\[
\hat{x}_{0,i-1},\quad \hat{x}_{0,i}
\]

通过二者差值估计 \(\hat{x}_0\) 随 logSNR 的变化趋势，再加入二阶修正。这样每个新时间点只增加一次 DiT 前向，却获得比 DDIM 一阶方法更高的精度。

简化理解：

\[
\text{新状态} = \text{解析传播(旧状态)} + \text{当前 }\hat{x}_0\text{ 贡献} + (\hat{x}_{0,i}-\hat{x}_{0,i-1})\text{ 的斜率修正}
\]

## 4. steps 与真实 NFE

这是当前代码最容易误解的地方。

因为开启了 `denoise_to_zero=True`：

\[
\boxed{\text{真实 NFE} = \text{diffusion\_steps} + 1}
\]

所以：

| solver steps | 真实 NFE |
|-------------:|--------:|
| 10 | 11 |
| 8 | 9 |
| 6 | 7 |
| 4 | 5 |
| 2 | 3 |

二阶 multistep 要求 `steps >= 2`，见 `diffusion_planner/model/diffusion_utils/dpm_solver_pytorch.py:1177`。

当前 warm-start 路径还会先做一次验证预测，但该预测会被复用为 solver 的第一个历史模型输出，所以成功 warm-start 时总量通常仍是 `steps + 1`。如果验证失败后回退完整采样，则可能达到 12 NFE。

## 5. 降低 NFE 的影响

### 数值积分误差增大

求解区间不变、网格点减少，意味着单步 logSNR 跨度增大。结果是：

- 二阶截断误差增大；
- DiT 本身的预测误差被大步外推放大；
- 极低 NFE 时，一阶初始化和末端降阶步骤占比变高；
- 轨迹可能没有充分回到训练数据流形。

### 轨迹质量退化

不一定只表现为 ADE/FDE 增大，更可能表现为：

- 横向抖动和曲率不连续；
- 速度、加速度、jerk 变差；
- 停车位置或跟车距离偏移；
- 转弯、换道等少数模式被抹平；
- 邻车预测误差影响 ego 的交互决策。

低温采样能减少随机波动，但不能消除确定性的积分误差。

### 闭环影响被放大

规划器每帧重新规划，并由下游控制器执行轨迹前段。因此单帧小误差会改变下一帧 ego 状态和交通交互，形成：

\[
\text{采样误差} \rightarrow \text{控制误差} \rightarrow \text{下一帧输入分布变化} \rightarrow \text{后续规划继续偏移}
\]

所以单帧 L2 看起来差异很小，闭环 TTC、碰撞和舒适度仍可能明显掉分。

仓库已有 val14 gate warm-start 实验：

| 指标 | 基线 | 低 NFE 组合 |
|------|-----:|----------:|
| 平均 NFE | 11.00 | 4.93 |
| decoder 延迟 | 19.00 ms | 12.18 ms |
| CLS-NR | 89.59 | 85.23 |
| 舒适度 | 95.08 | 91.50 |
| TTC | 90.43 | 87.03 |

即 NFE 加速 \(2.23\times\)，decoder 只加速 \(1.56\times\)，CLS-NR 下降 4.36。主要损失来自舒适度、TTC、无责碰撞和路线进度，见 `0实验结果/DP-val14-gate-warmstart.md`。

不过该实验同时改变了初值、\(t_{\text{start}}\) 和动态步数，所以不能把全部掉分归因于 NFE。

## 6. 如何减少低 NFE 的影响

按投入产出比，我建议：

1. **先测试去掉最后一次 NFE。**
   保持 10 个 solver 网格，只将 `denoise_to_zero=False`，实际从 11 降到 10 NFE。它不会粗化主积分区间，是风险最低的消融，但必须验证末端轨迹质量。

2. **做纯 steps sweep。**
   固定 checkpoint、随机种子和初始噪声，依次测 10/8/6/5/4/3/2。建议先从 8、6 步开始，不要直接降到 2。

3. **优先缩短求解区间。**
   利用上一帧轨迹 warm-start，将其坐标变换、邻车 token 对齐后加噪到较小 \(t_{\text{start}}\)。
   “短区间 4 步”通常比“从纯噪声跨越完整区间的 4 步”合理。

4. **动态分配 NFE。**
   稳态直行、静止、简单跟车可以少步；以下场景提高 NFE：
   - 转弯和换道；
   - 高横向加速度；
   - 交通灯起步；
   - 多车近距交互；
   - 行人和复杂跟车。

5. **建立安全回退。**
   不只比较 warm-start 轨迹 L2，还应检查：
   - 最小碰撞距离和 TTC；
   - drivable-area violation；
   - 曲率、加速度和 jerk；
   - ego 前几秒轨迹偏差；
   - warm-start 首次 \(\hat{x}_0\) 与参考轨迹的差异。

   超阈值时回退完整 11 NFE。

6. **极低 NFE 需要训练配合。**
   如果目标是稳定达到 1–4 NFE，仅调 solver 通常不够。更合适的是少步数网格微调、progressive distillation、consistency model，或者以 10-step 输出作为 teacher 蒸馏。Diffusion Planner 论文也将 consistency 与蒸馏列为进一步加速方向。

我的工程判断是：6–8 solver steps 适合作为纯降步数起点；4 步以下应结合 warm-start、动态门控和安全回退；稳定做到 1–4 NFE 则应考虑蒸馏。
