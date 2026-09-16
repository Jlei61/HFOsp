# 自主恢复：实时观测审阅

已分析 12 条，完成 12 条。所有轨迹均无 Z/M reset、无时刻触发的外部抑制。

| 条件 | 观测时长(s) | 高态起点(s) | 自主恢复确认(s) | 末10s全E / 最强core均率(Hz) | 当前判定 |
|---|---:|---|---|---|---|
| add_g0.25_s9108401 | 60.0 | [] | [] | 6.41 / 22.64 | NO_HIGH_ENTRY |
| add_g0.5_s9108401 | 60.0 | [] | [] | 2.43 / 8.69 | NO_HIGH_ENTRY |
| mix_g0.25_eta0.005_s9108401 | 60.0 | [3.66] | [] | 500.00 / 500.00 | HIGH_WITHOUT_VERIFIED_RECOVERY |
| mix_g0.25_eta0.02_s9108401 | 60.0 | [5.82] | [] | 500.00 / 500.00 | HIGH_WITHOUT_VERIFIED_RECOVERY |
| mix_g0.5_eta0.005_s9108401 | 60.0 | [1.39] | [] | 500.00 / 500.00 | HIGH_WITHOUT_VERIFIED_RECOVERY |
| mix_g0.5_eta0.02_s9108401 | 60.0 | [1.41] | [] | 500.00 / 500.00 | HIGH_WITHOUT_VERIFIED_RECOVERY |
| mix_g0.75_eta0.005_s9108401 | 60.0 | [1.19] | [] | 500.00 / 500.00 | HIGH_WITHOUT_VERIFIED_RECOVERY |
| mix_g0.75_eta0.02_s9108401 | 60.0 | [1.19] | [] | 500.00 / 500.00 | HIGH_WITHOUT_VERIFIED_RECOVERY |
| native_fastZ_s9108401 | 120.0 | [] | [] | 1.39 / 1.86 | NO_HIGH_ENTRY |
| native_longM_s9108401 | 120.0 | [10.87] | [] | 500.00 / 500.00 | HIGH_WITHOUT_VERIFIED_RECOVERY |
| native_midM_s9108401 | 120.0 | [] | [] | 2.19 / 3.17 | NO_HIGH_ENTRY |
| native_weak_s9108401 | 60.0 | [9.87] | [] | 500.00 / 500.00 | HIGH_WITHOUT_VERIFIED_RECOVERY |

每个条件先用一条配对开发噪声定位机制；需要第二噪声确认。两秒低态是筛查门，不代替有限事件、Z重新积累和完整空间场验收。无进入只表示在保存时长内未达到固定200Hz/200ms定义，不是永不发作。

汇总图：红三角为高态进入，绿圆为全E与双核均通过的自主恢复；中图实心点为全E均率、空心点为较强核均率；右图为末时刻平均Z。
