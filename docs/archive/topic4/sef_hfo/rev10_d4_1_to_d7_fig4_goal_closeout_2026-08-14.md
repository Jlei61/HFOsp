# Topic 4 rev10 D4.1→D7 与 Fig.4 验收闭环

## 总体裁定

```text
FORCED_ROUTE_CAPACITY_BROAD_TIMING_SUPPORTED
/
SPONTANEOUS_DIRECTION_ACCESS_PARTIAL_PASS
/
NATURAL_KMEANS_PATIENT_BENCHMARK_FAIL
/
ACTIVE_ZM_SAFETY_FAIL
/
FIG4_REPLACEMENT_NOT_AUTHORIZED
```

这条实验线已完成预定的强制路由确认、平移不变自发可达性实验和两张 Fig.4 风格验收图，但结果没有达到 Fig.4 科学验收。最安全的结论是：冻结 Node scaffold 在强制触发下能携带 A/B 两条传播方向；平移不变 spatial OU 能让两种监督方向在同一网络内自发出现，但自然 KMeans 与患者模式的一致性没有跨网络稳定复现。当前不能说 data-driven SNN 复现了患者间期活动。

## 逐项验收

### 1. D4.1 新网络强制路由剂量确认

6 张新网络、A/B 两个冻结 source 和 20/40/80/160 E-neuron packet 全部运行完成。正式 40 ms latency gate 在最大剂量得到 A `5/6`、B `4/6` clean，因此预注册状态保持 `REV10D4_1_FRESH_NETWORK_FORCED_AB_ROUTE_NOT_CONFIRMED`。

冻结后的 timing audit 解释了主要边界：两个缺失 B response 均在 `141 ms` 起始，只比 `100–140 ms` 窗口晚 1 ms；放宽到完整 paired window 后，A/B 都是 `6/6` returned、`6/6` expected direction 和 `6/6` joint-shaft recruitment，无 sham overlap。A 仍有一张网络略超 OOD boundary。因此可保留的机制状态是 `FORMAL_GATE_FAIL_BUT_BROAD_ROUTE_TIMING_SUPPORTED`，不是完整 formal pass。

### 2. 连续、平移不变、观察位置无关的自发机制

D5 的 local spatial OU 只调制外部 E afferent rate，使用周期二维网格、各向同性核和零空间均值；不读取 contact/shaft/patient target，不放置固定 source/core，也不改变 recurrent topology。matched permutation 保留每次更新的神经元值多重集和时间谱，只破坏空间邻接。

D5.2 在 6/6 新网络中产生同网络监督 A+B；患者 prototype Spearman 矩阵为：

```text
[[+0.679, -0.871],
 [-0.539, +0.800]]
```

但 canonical masked-rank KMeans direction purity 只有 `0.674`，低于 patient matched q05 `0.884`。活动负担也明显增加：time-above-detector 从 off 的 `0.060` 升到 local 的 `0.268`。D5.4 在新 selection networks 上没有复现 purity 改善。因此该机制只通过“方向可达性”，没有通过“自然双模式患者一致性”。

### 3. 连续自由场修复

D6 使用 whole-sheet Fourier residual 投影到单个连续 B-spline `h(x,y)`；没有 Gaussian components、`K`、peak count 或 contact-conditioned basis。D6.2 的近边界 `(0.5,0.5)` 候选在 6 张网络上改善 patient cross-fit，但自然 KMeans paired q05 略跨 0。

D6.3 将该候选冻结后在 12 张新网络复制：自然 alignment paired delta `+0.057 [−0.021,0.137]`，patient cross-fit delta `−0.107 [−0.270,0.077]`，支持 K=2 的网络仅 `4/12`。因此局部连续场联合信号没有复制。这一结果不来自 optimizer 未收敛，因为 D6.2/D6.3 直接测试冻结方向，没有运行 optimizer。

### 4. active Z+M 共享方程

D7 将同一 49-field observation-invariant library 接到 active Z+M reference，每条轨迹至少允许运行 20 s，并将 late runaway 判为无效。98/98 worker 完整且 provenance clean，但全部在 `5.83–10.29 s` runaway；没有安全候选进入 KMeans selection。固定 Z/M reference 不能作为当前 data-driven SNN 的稳定机制 baseline。

### 5. Fig.4 两图合同

唯一生成并视觉验收的候选图是 D6.3：

- `fig4a_spatial_ou_direct_readout`：连续场、同网络自发 TA/TB onset density 和 15-contact 直接波形。
- `fig4b_spatial_ou_kmeans_consistency`：masked-rank KMeans、rank distributions、模型/患者 prototypes 和 cross-fit matrix。

两图满足“只看同网络自发波形 + KMeans/患者一致性”的展示合同，PNG 为 600 dpi、PDF 单页、producer provenance clean。图中 pooled cross-fit matrix 为 `[[+0.80,-0.72],[-0.48,+0.35]]`，第二模式匹配弱；更重要的是 D6.3 network-level replication fail。因此两图只能标记 `DIAGNOSTIC_ONLY`，不得替换主文 Fig.4。

## 当前能否说明复现患者间期活动

不能。当前只支持：

> 患者数据约束的连续 Node scaffold 加平移不变空间涨落，能够在新的 SNN 网络中产生同网络的两个监督传播方向；其中部分 prototype geometry 与患者 A/B 同号，但自然分群强度和患者一致性没有跨网络稳定复制。

不支持完整患者事件分布、patient blind generalization、临床 SEEG 波形/频谱、core 因果、Edge 机制或发作生命周期。

## 对“优化器还是机制”的判断

当前主要不是优化器问题。D6.2/D6.3 没有依赖 optimizer，仍出现 discovery→replication 反转；D7 的失败发生在安全门，任何目标函数或优化器都不能把 runaway 轨迹变成有效候选。更准确的剩余问题是：当前 safe slow-off dynamics 缺乏跨网络稳定的自然 route-basin separation，而当前 Z/M dynamics 又不安全。

若另开下一机制线，应停止继续调同一 Z/M 参数或在当前局部 `h` 方向上加算力。下一问题应单独预注册为“安全、返回型的状态依赖 route competition”，先证明同网络自然 K=2 跨网络稳定，再读取患者相似性；它不属于本轮 Fig.4 已完成实验合同。
