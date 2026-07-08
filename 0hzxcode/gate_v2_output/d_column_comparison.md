# gate d 口径 / 模型选型对比 (test)


| run     | d_column         | Spearman | rmse_log1p | adj acc | AUROC  | reuse@HR95 | 校准单调 |
| ------- | ---------------- | -------- | ---------- | ------- | ------ | ---------- | ---- |
| perstep | perstep_max_m    | 0.6261   | 0.4885     | 0.886   | 0.6982 | 0.271      | True |
| norml2  | normalized_l2_xy | 0.6351   | 0.3560     | 0.895   | 0.7141 | 0.301      | True |
| prevd   | perstep_max_m    | 0.7412   | 0.4338     | 0.916   | 0.8108 | 0.368      | True |


主指标: **Spearman** + **reuse@HR95**(hard_recall≥95% 下可复用率) + **校准单调**。