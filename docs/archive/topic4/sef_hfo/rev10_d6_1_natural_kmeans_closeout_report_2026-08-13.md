# Topic 4 rev10-D6.1 自然比例 KMeans closeout 报告

## 一句话裁定

```text
D6_KMEANS_CLOSEOUT_COMPLETE
/
PREFROZEN_PRIMARY_NOT_REPLICATED
/
ORTHOGONAL_CONTINUOUS_FIELD_SENSITIVITY_PARTIAL
/
PATIENT_ALIGNED_NATURAL_REPERTOIRE_UNRESOLVED
```

连续自由场的低频局部方向能够分别改变自然 KMeans 方向一致性和 contact-split patient geometry，但没有同一个候选同时稳定改善这两层，也没有形成跨网络稳定的 K=2 density evidence。因此 D6 支持“连续场局部敏感性”，不支持“患者对齐双模式 repertoire 已复制”。

## 工程完成度

- 5 个预冻结连续场 × 6 张 fresh networks × 16 s，30/30 workers 成功。
- source worker commit：`b4ee5e5ded4940903435cb00d39ad9b474dd6c15`。
- 无 runaway；每个 worker 使用 exact edge no-op、beta closed、冻结 topology、冻结 spatial OU 和同一绝对 detector。
- RSS sentinel peak `12,887,832 KiB`，按可用内存自动限制为 5 workers；每 worker cgroup `MemoryHigh=20G`、`MemoryMax=24G`，无 OOM/worker failure。
- 30 个 JSON/NPZ 均通过 config、manifest、array 和 runtime-module provenance 检查。

## 正式 primary

预冻结 primary `d6_f03_sin_p0p4` 未复制：

- natural KMeans equal-network alignment：`0.673`，baseline `0.643`；paired delta `+0.030`，network bootstrap q05–q95 `[-0.098, +0.172]`，仅 2/6 网络为正。
- contact-split cross-fit patient margin paired delta：`-0.091 [-0.235, +0.056]`。
- recruitment error 相对 baseline 变差 `0.067 [0.033, 0.100]`。
- K=2 相对 K=1 held-out likelihood：0/6 网络为正。
- OOD `0.605`，高于 baseline `0.481`。

因此 Fig.4 canonical 两图展示的是一次正式未复制，而不是成功候选。

## 两个正交的探索性方向

### `d6_f09_sin_p0p4`：自然方向一致性候选

- natural alignment `0.757 [0.656, 0.867]`；paired baseline delta `+0.114 [-0.047, +0.266]`，4/6 网络为正。
- cross-fit patient margin `0.385`，baseline `0.397`；paired delta `-0.012 [-0.192, +0.103]`。
- K=2 over K=1 仅 2/6；OOD `0.519`；无 runaway。

它说明某个连续场方向可提高 KMeans 与方向标签的一致性，但不能证明患者 geometry 或稳定双峰 density 同时改善。

### `d6_f05_sin_m0p8`：cross-fit patient geometry 候选

- natural alignment paired delta 仅 `+0.016 [-0.075, +0.112]`，虽 4/6 为正但幅度很小。
- cross-fit patient margin `0.476 [0.382, 0.562]`；paired delta `+0.079 [0.005, 0.156]`，4/6 为正。
- recruitment worst-mode error `0.161`，差于 baseline `0.128`；K=2 over K=1 仅 2/6。

它说明患者 rank geometry 可被连续场方向改善，但该改善没有同步转化为自然双簇或 recruitment 改善。

## KMeans 与模式比例边界

- patient train 原 A/B 比例为 `0.309/0.691`。
- `f03` 的 cross-fit consensus 约 `45/47`，接近均衡，未恢复 patient occupancy。
- `f09` 为 `56/76`，B 比例约 `0.576`，仍低于 patient `0.691`。
- `f05` 为 `51/101`，B 比例约 `0.664`，比例较接近患者，但 recruitment 和自然 KMeans 未同时改善。
- pooled KMeans 仅作描述；正式汇总先在每张网络内计算，再等权聚合。

## 图

正式 primary：

- `figures/fig4a_spatial_ou_direct_readout.{png,pdf}`
- `figures/fig4b_spatial_ou_kmeans_consistency.{png,pdf}`

讨论用 fresh descriptive best，不替代 primary：

- `figures/descriptive_fresh_best/fig4a_spatial_ou_direct_readout.{png,pdf}`
- `figures/descriptive_fresh_best/fig4b_spatial_ou_kmeans_consistency.{png,pdf}`

Fig.4B 最右矩阵使用两杆内交替 contact folds：一组 contacts 分配模式，互斥 contacts 评价 patient Spearman geometry，再交换两组并对网络等权。图中 pooled KMeans heatmap 和 cluster profile 仍是描述性视图。

## 下一步边界

D6 的 KMeans 层已经完成，不应继续增加 K、core 数、edge、beta 或换 optimizer 来追这个单指标。下一轮应把自然 KMeans alignment、cross-fit patient geometry、recruitment/occupancy 和完整 shaft-aware distribution 作为分开的低维响应，先判断 `f09` 与 `f05` 两个自由场方向的小范围连续组合能否得到同一候选的共同改善；若组合仍失败，再把问题归到 field-only family capacity，而不是继续扩大 KMeans 搜索。
