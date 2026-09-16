# 闭环筛查结果

本轮目标为同一连续轨迹的高活动→反复短间期事件→再次高活动。以下是模型内时序筛查，尚不等于患者传播、发作形态或跨噪声验证。

| γ | K倍率 | τK(s) | 时长(s) | 前期短事件 | 高活动次数 | 退出次数 | 分类 |
|---|---|---|---|---|---|---|---|
| 0.5 | 1.5 | 1 | 54 | 0 | 1 | 0 | HIGH_WITHOUT_EXIT |
| 0.5 | 1.5 | 2 | 54 | 0 | 1 | 0 | HIGH_WITHOUT_EXIT |
| 0.1667 | 0.1 | 1 | 58 | 67 | 1 | 0 | HIGH_WITHOUT_EXIT |
| 0.1667 | 0.1 | 5 | 58 | 108 | 0 | 0 | NO_HIGH_OBSERVED |
| 0 | 0.1 | 1 | 58 | 215 | 0 | 0 | NO_HIGH_OBSERVED |
| 0 | 0.1 | 5 | 58 | 101 | 0 | 0 | NO_HIGH_OBSERVED |

旧27条轨迹按新标准重审，无通过时序闭环者；原46s延长仅有高活动—低活动—高活动，之间没有合格短事件。旧结果没有被覆盖。

若本轮有TEMPORAL_LOOP_PASS，必须继续目视检查原生raster/空间传播，并确认初始短事件保留和不同噪声下的行为；若没有则明确失败或观察窗截尾，不更改阈值、不自动扩大搜索。

![fig5_g0.166667_k0.1_tau1_s9108401](/data/hfosp/topic4_sef_hfo/fig5_interictal_recurrence_20260915/figures/fig5_g0.166667_k0.1_tau1_s9108401.png)

![fig5_g0.166667_k0.1_tau5_s9108401](/data/hfosp/topic4_sef_hfo/fig5_interictal_recurrence_20260915/figures/fig5_g0.166667_k0.1_tau5_s9108401.png)
