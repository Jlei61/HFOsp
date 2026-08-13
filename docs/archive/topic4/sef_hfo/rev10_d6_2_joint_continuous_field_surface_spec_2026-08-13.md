# Topic 4 rev10-D6.2：连续场两方向组合响应面

## 科学问题

D6.1 在六张新网络上留下两个不同信号：`d6_f09_sin_p0p4` 改善自然 KMeans 与方向标签的一致性，`d6_f05_sin_m0p8` 改善 contact-split patient cross-fit，但没有单个候选同时可靠改善两者。D6.2 只回答：这两个局部连续场方向的组合，能否在同一张网络的自发事件中同时形成数据支持的 K=2 结构和患者 A/B 几何。

## 冻结机制

- 场仍是 18 系数 tensor cubic B-spline 的单个连续 `h(x,y)`，无 `K`、component 或 peak 参数。
- 每个候选都经原有精确 field-mass 投影；总质量不变。
- 两个方向分别为 D6.1 的 `f09-warm` 与 `f05-warm`，在 D6.2 新网络运行前冻结。
- 七个坐标：`(0,0)`、`(1,0)`、`(0,1)`、`(0.5,0.5)`、`(1,1)`、`(1,0.5)`、`(0.5,1)`。
- 固定 D5.2 spatial OU、common detector、网络拓扑和 16 s 时长；Edge exact no-op，`beta`、慢变量和 optimizer 均关闭。
- 新网络种子为 1361–1366；network seed 是独立统计单位。

## 读数与裁定

不设新的加权总目标，也不以 pooled event 数作推断。逐候选报告：

1. 每网络自然 KMeans 与 cross-fit direction label 的 balanced alignment；
2. contact-split patient matrix 的最弱 signed margin；
3. held-out GMM `K=2-K=1` log-likelihood；
4. A/B recruitment 误差、OOD 和 runaway；
5. 候选减 warm baseline 的 paired network bootstrap。

探索性 joint signal 只要求两项 paired delta 的 bootstrap q05 均大于 0、至少 4/6 网络支持 K=2、且无 runaway。Recruitment 完整报告但不再增加 hard gate。该规则识别局部联合信号，不等于完整患者间期活动通过。

## 结论边界

本轮使用 patient-training target 且没有新 patient blind unit。任何阳性结果只说明：固定连续场局部二维子空间在新网络 realization 上存在联合自然 repertoire 信号；不能称为患者泛化、完整间期活动复现、core 因果、Edge 机制、optimizer 优越或发作生命周期。
