# M-probe report

Decision: **FAIL**

STOP: keep fixed t_s warm-start; do not train learned gate yet

## Data

- rows: 144441
- scenarios: 1118
- split rows: {'train': 101031, 'val': 21715, 'test': 21695}
- hard threshold d: 2.63074
- hard positive rate: 0.0986

## Test metrics

| feature | RMSE | MAE | Spearman | AUROC | AP | hard recall | false easy |
|---|---:|---:|---:|---:|---:|---:|---:|
| encoding | 2.2483 | 0.874706 | 0.6250 | 0.6742 | 0.1539 | 0.9519 | 0.0481 |
| ego_features | 2.32129 | 1.01854 | 0.0000 | 0.5000 | 0.0927 | 1.0000 | 0.0000 |

## Checks

- spearman_ok: True
- auroc_ok: False
- hard_recall_ok: True
- rmse_improvement_vs_ego: 0.03144305484067467
- rmse_improvement_ok: False
