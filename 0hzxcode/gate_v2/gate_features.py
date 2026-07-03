#!/usr/bin/env python3
"""gate_v2 特征组：仅 ego 运动学 + 邻车轻量补充。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GroupSpec:
    name: str
    rank: int
    keys: dict = field(default_factory=dict)
    description: str = ""


FEATURE_GROUPS: list[GroupSpec] = [
    GroupSpec(
        name="ego_history",
        rank=1,
        keys={"ego_history": (21, 7)},
        description="过去 2.1s 自车状态（当前帧系）+ 派生运动学",
    ),
    GroupSpec(
        name="neighbor_agents",
        rank=2,
        keys={"neighbor_agents_past": (32, 21, 11)},
        description="32 邻车过去 2.1s 轨迹，补 hard 交互帧",
    ),
]

GROUP_BY_NAME: dict[str, GroupSpec] = {g.name: g for g in FEATURE_GROUPS}
LABEL_KEYS = ("d", "sample_id", "scenario_id", "scenario_type")
DEFAULT_ENABLED = [g.name for g in FEATURE_GROUPS]


def resolve_enabled_groups(enable: list[str] | None, disable: list[str] | None) -> list[str]:
    names = [g.name for g in FEATURE_GROUPS]
    if enable:
        unknown = set(enable) - set(names)
        if unknown:
            raise ValueError(f"unknown groups in --enable-groups: {sorted(unknown)}")
        enabled = [n for n in names if n in set(enable)]
    else:
        enabled = list(DEFAULT_ENABLED)
    if disable:
        unknown = set(disable) - set(names)
        if unknown:
            raise ValueError(f"unknown groups in --disable-groups: {sorted(unknown)}")
        enabled = [n for n in enabled if n not in set(disable)]
    if not enabled:
        raise ValueError("gate_v2 needs at least one feature group")
    return enabled
