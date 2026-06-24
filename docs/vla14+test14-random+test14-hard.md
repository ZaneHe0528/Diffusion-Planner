这三个选项是 nuPlan 社区常用的 **闭环评测场景集（benchmark split）**，不是官方挑战赛 split，而是论文里约定俗成的划分。区别主要在 **数据来源、场景数量、难度** 三方面。

## 总览对比


| Split             | 数据来源                       | 场景数                 | 难度  | 典型用途           |
| ----------------- | -------------------------- | ------------------- | --- | -------------- |
| **val14**         | validation split（trainval） | ~1118               | 中等  | 开发调试、与论文基线对比   |
| **test14-random** | test split                 | 14 类 × 20 = **280** | 中等  | test 集上的随机抽样评测 |
| **test14-hard**   | test split                 | **272** 个固定困难场景     | 高   | 压力测试、考察极端/突发情况 |


---

## 1. `val14`

- 来自 **PDM 论文**（CoRL 2023）定义的 benchmark。
- 从 **validation split** 中选取 **14 类** 典型挑战场景（左转、右转、跟车、变道、等行人等），每类最多 100 个，共约 1118 个场景。
- 配置里用固定 `scenario_tokens` 列表 + `log_names: ${splitter.log_splits.val}`。
- 脚本里会走 `scenario_builder=nuplan`（读 `trainval` 数据）。

**适合**：日常开发、快速迭代、和 Diffusion-Planner README 里的基线分数（CLS-NR 89.87 / CLS-R 82.80）直接对比。

---

## 2. `test14-random`

- 来自 **planTF 论文**。
- 同样 14 类场景，但从 **test split** 中 **每类随机抽 20 个**（`num_scenarios_per_type: 20`，无固定 token 列表）。
- 脚本里走 `scenario_builder=nuplan_challenge`（读 `test` 数据）。

**适合**：在 **未见过的 test 集** 上做泛化评测，场景相对均衡、数量较少（280 个），跑起来比 val14 快。

---

## 3. `test14-hard`

- 同样来自 planTF，也在 **test split** 上。
- 不是随机抽样，而是用基线 planner 筛出的 **272 个低分困难场景**（固定 `scenario_tokens` 列表）。
- 也走 `nuplan_challenge`。

**适合**：专门测 planner 在 **困难/突发场景** 下的表现；改进算法时，这里和 CLS-R 上的差距往往更明显（你们文档里也提到 test14-hard 更适合验证 threshold 回退等机制）。

---

## 脚本里的额外差异

```38:42:sim_diffusion_planner_runner.sh
if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
```

- `val14` → 用 validation 数据（`trainval`）
- `test14-random` / `test14-hard` → 用 test 数据（`nuplan-v1.1/test/`）

---

## 怎么选

- **第一次跑 / 和论文比分数** → `val14`
- **看 test 集泛化** → `test14-random`
- **专门压测困难场景** → `test14-hard`

三者评测指标相同（CLS-NR / CLS-R 的 8 项闭环指标），区别只在 **测哪些场景、从哪个数据 split 取**。