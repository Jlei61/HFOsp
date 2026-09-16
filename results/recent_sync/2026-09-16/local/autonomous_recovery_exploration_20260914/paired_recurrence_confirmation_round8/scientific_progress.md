# 自主恢复：实时观测审阅

已分析 6 条，完成 6 条。所有轨迹均无 Z/M reset、无时刻触发的外部抑制。

| 条件 | 观测时长(s) | 高态起点(s) | 自主恢复确认(s) | 末10s全E / 最强core均率(Hz) | 当前判定 |
|---|---:|---|---|---|---|
| resource_rho0.25_k0_tau10_s9108401 | 60.0 | [13.66] | [] | 476.38 / 480.18 | HIGH_WITHOUT_VERIFIED_RECOVERY |
| resource_rho0.25_k0_tau10_s9108402 | 60.0 | [13.86] | [] | 476.38 / 480.19 | HIGH_WITHOUT_VERIFIED_RECOVERY |
| resource_rho0.25_k200_tau10_s9108402 | 30.0 | [13.86, 23.89] | [22.33] | 60.84 / 75.88 | AUTONOMOUS_RECURRENCE_CANDIDATE |
| resource_rho0.25_k50_tau10_s9108402 | 60.0 | [13.86] | [] | 92.94 / 107.39 | HIGH_WITHOUT_VERIFIED_RECOVERY |
| resource_rho0_k200_tau10_s9108401 | 30.0 | [11.72, 22.71, 26.45] | [22.41, 25.16, 29.37] | 80.82 / 94.78 | AUTONOMOUS_RECURRENCE_CANDIDATE |
| resource_rho0_k50_tau10_s9108401 | 60.0 | [11.72] | [] | 500.00 / 500.00 | HIGH_WITHOUT_VERIFIED_RECOVERY |

每个条件先用一条配对开发噪声定位机制；需要第二噪声确认。两秒低态是筛查门，不代替有限事件、Z重新积累和完整空间场验收。无进入只表示在保存时长内未达到固定200Hz/200ms定义，不是永不发作。

汇总图：红三角为高态进入，绿圆为全E与双核均通过的自主恢复；中图实心点为全E均率、空心点为较强核均率；右图为末时刻平均Z。
