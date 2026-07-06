# 门控输入改用原始观测（ego 运动学 + 邻车）而非场景嵌入 encoding

ADR-0002 规划在 encoder fusion 输出 `encoding` 上挂 scene gate。gate_v2 离线对比显示 `ego_only` 与 `all_groups` 的 Spearman 几乎相同（≈0.643 vs 0.641），静态地图/障碍对秩相关无增益；而原始 ego 运动学已显式编码复杂度信号，且闭环侧无需额外 encoder 前向即可取特征。

决策：闭环集成的 gate_v2（lite gate）使用 **ego_history + neighbor_agents** 作为输入，不使用 `encoding`。scene gate（encoding 版）保留为未实现的历史方案与潜在对照。

理由：(1) 离线指标不支撑为 encoding 路径付出额外集成复杂度。(2) ego 运动学在 val14 逐帧标签上已具备可用 Spearman（perstep 口径 ≈0.63）。(3) 邻车轻量组主要提升 hard 帧 AUROC，作为安全上调档位的补充信号。(4) 避免 M-probe 未通过时整条线止损——lite gate 已可独立交付。

被否决的备选：(a) 坚持 encoding-only scene gate——离线探针与 gate_v2 对比均未显示显著优势；(b) 同时在线融合 encoding + 原始观测——参数量与闭环工程复杂度上升，首轮集成不采用。

后果：术语上区分 **scene gate**（encoding，未实现）与 **lite gate**（原始观测，已实现）；`0BLUE` 计划中的 M-probe 门槛对 lite gate 路径不再适用。
