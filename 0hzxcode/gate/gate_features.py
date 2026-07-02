#!/usr/bin/env python3
"""Gate 输入特征组注册表（按先验重要程度排序，支持逐组开/关）。

设计对齐 BLUE（冻结主干 + 轻量 gate），但输入不用 planner encoding，
而是闭环推理时可零成本获得的**原始观测**。每个特征组：
  - 在 gate 网络里是独立编码分支，可单独开/关（训练期 group-dropout，
    推理期直接置零该组 embedding，或训练时彻底排除）；
  - rank = 先验重要性（1 最重要）。训练脚本会输出实测重要性
    （leave-one-group-out + permutation）与先验排序的对照表。

先验排序理由（预测目标是帧间规划轨迹变化距离 d / 变化档位 level）：
  1 ego_history      过去 2.1s 自车运动学（21 帧 x,y,heading,vx,vy,ax,ay，当前帧系）。
                     d 是相邻两帧规划轨迹差，自车速度/加减速/转向直接决定轨迹
                     跨度与重规划幅度（停车帧 d≈0）。planner 推理不用它，但
                     history buffer 里现成可得。
  2 neighbor_agents  周边动态体（车/人/自行车）过去 2.1s 轨迹 + 类型 one-hot。
                     交互对象的出现/消失/急动是重规划的主要外因。
  3 route_lanes      导航路线车道（几何 + 红绿灯 one-hot + 限速）。
                     前方转弯、路口决定轨迹形状变化的可预期部分。
  4 lanes            周边全部车道（几何 + 红绿灯 one-hot + 限速）。
                     场景结构复杂度与信号变化；红绿灯状态在最后 4 维。
  5 static_objects   静态障碍（锥桶/路障/标志等）+ 类型 one-hot。通常稀少，
                     只影响局部绕行。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GroupSpec:
    """一个可开关的观测特征组。

    keys: 该组在数据集 npz 中占用的数组键 -> 每帧 shape（不含 batch 维）。
    derived: True 表示不存在于 npz，由 gate_dataset 加载时派生。
    default_enabled: False 的组默认不参与训练（需显式开启）。
    """

    name: str
    rank: int
    keys: dict = field(default_factory=dict)
    description: str = ""
    derived: bool = False
    default_enabled: bool = True


# 按先验重要性排序（rank 1 最重要）。修改顺序/增删组时保持 rank 与列表顺序一致。
FEATURE_GROUPS: list[GroupSpec] = [
    GroupSpec(
        name="ego_history",
        rank=1,
        keys={"ego_history": (21, 7)},
        description="过去 2.1s 自车状态（当前帧系）：x,y,heading,vx,vy,ax,ay",
    ),
    GroupSpec(
        name="neighbor_agents",
        rank=2,
        keys={"neighbor_agents_past": (32, 21, 11)},
        description="最近 32 个动态体过去 2.1s：x,y,cos,sin,vx,vy,w,l + 类型 one-hot(3)",
    ),
    GroupSpec(
        name="route_lanes",
        rank=3,
        keys={
            "route_lanes": (25, 20, 12),
            "route_lanes_speed_limit": (25, 1),
            "route_lanes_has_speed_limit": (25, 1),
        },
        description="路线车道 25 条 x 20 点：中心线+边界向量+红绿灯 one-hot(4)，附限速",
    ),
    GroupSpec(
        name="lanes",
        rank=4,
        keys={
            "lanes": (70, 20, 12),
            "lanes_speed_limit": (70, 1),
            "lanes_has_speed_limit": (70, 1),
        },
        description="周边车道 70 条 x 20 点：几何+红绿灯 one-hot(4)，附限速",
    ),
    GroupSpec(
        name="static_objects",
        rank=5,
        keys={"static_objects": (5, 10)},
        description="静态障碍 5 个：x,y,cos,sin,w,l + 类型 one-hot(4)",
    ),
    # 可选组（默认关）：不属于"原始观测"，而是闭环缓存里近乎免费的上一帧实测 d。
    # 离线诊断显示它单独 Spearman≈0.69/AUROC≈0.80，是最强单一信号；
    # 但注意：训练标签来自 full-denoise 回放，warm-start 部署后其分布会漂移
    # （计划风险表中的"训练/闭环分布不匹配"），启用需配合 DAgger 式回灌验证。
    GroupSpec(
        name="prev_d",
        rank=6,
        keys={"prev_d": (3,)},
        description="上一帧实测 d：[d_prev, log1p(d_prev), valid]，场景首帧 valid=0",
        derived=True,
        default_enabled=False,
    ),
]

GROUP_BY_NAME: dict[str, GroupSpec] = {g.name: g for g in FEATURE_GROUPS}

# 数据集 npz 中除特征组外必须存在的标签/元数据键。
LABEL_KEYS = ("d", "sample_id", "scenario_id", "scenario_type")


def resolve_enabled_groups(enable: list[str] | None, disable: list[str] | None) -> list[str]:
    """根据白名单/黑名单解析启用的特征组，保持注册表顺序。

    不给 --enable-groups 时默认启用所有 default_enabled 组（prev_d 等可选组需显式开启）。
    """
    names = [g.name for g in FEATURE_GROUPS]
    if enable:
        unknown = set(enable) - set(names)
        if unknown:
            raise ValueError(f"unknown groups in --enable-groups: {sorted(unknown)}")
        enabled = [n for n in names if n in set(enable)]
    else:
        enabled = [g.name for g in FEATURE_GROUPS if g.default_enabled]
    if disable:
        unknown = set(disable) - set(names)
        if unknown:
            raise ValueError(f"unknown groups in --disable-groups: {sorted(unknown)}")
        enabled = [n for n in enabled if n not in set(disable)]
    if not enabled:
        raise ValueError("all feature groups disabled; gate needs at least one input group")
    return enabled
