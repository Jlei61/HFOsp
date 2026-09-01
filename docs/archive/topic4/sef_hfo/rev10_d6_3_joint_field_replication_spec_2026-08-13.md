# Topic 4 rev10-D6.3：近边界连续场的新网络复制

## 科学问题

D6.2 的等权方向组合 `(a,b)=(0.5,0.5)` 在六张网络上同时满足 patient cross-fit 正下界和 4/6 的 K=2 密度支持；自然 KMeans paired improvement 为 `+0.133`，但 bootstrap q05 为 `-0.006`，没有通过 joint signal。D6.3 只判断这一近边界结果能否在更多完全新网络上复制，不再搜索场。

## 冻结合同

- 两臂：warm `edge_noop` 与 `d62_a0p5_b0p5`。
- 候选是同一个 18 系数连续 B-spline `h(x,y)`，无 component/peak/core-count 参数。
- 12 张新网络 `1401–1412`，paired network/OU seeds，固定 16 s。
- common detector、field mass、OU、拓扑和观察模型不变；Edge exact no-op，`beta`、Z/M 和 optimizer 关闭。
- D6.2 的 6 张网络不进入复制 gate。

## 正式复制规则

仅在 D6.3 的 12 张新网络上要求：自然 KMeans paired delta 与 patient cross-fit paired delta 的 network-bootstrap q05 均大于 0；候选至少 8/12 网络满足 held-out GMM `K=2>K=1`；无 runaway。Recruitment 连续报告但不是 hard gate。

通过也只说明这一冻结连续场相对 warm field 在 development networks 上稳定改善自然双模式与患者几何；不等于完整患者间期活动、patient blind generalization、core 因果、Edge/慢变量机制或发作生命周期。
