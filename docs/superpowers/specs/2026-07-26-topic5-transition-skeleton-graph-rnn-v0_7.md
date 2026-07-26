# Topic 5 / Figure 6：多路径 transition-skeleton graph RNN（v0.7）

**日期**：2026-07-26
**状态**：v0.6 coarse-axis screen 后的执行合同
**继承**：34人、masked rank、chronological 80/20、LOSO、target sealed、
clinical onset `[0,10] s`、BB `1–150 Hz`

## 1. 修改原因

v0.6 证明真实患者轴相对患者内轴打乱可改善 next-set NLL 和轴向方向分布，但不能
稳定改善完整 contact path 或 pairwise precedence。原因是 coarse graph 只按轴坐标
连接相邻触点，没有表达训练事件中实际重复出现的多条 contact-to-contact 路径。

## 2. v0.7 图先验

每位患者仅使用 chronological train80：

1. 保留 v0.6 的无符号患者轴；
2. 对每个事件的相邻 rank set，统计所有 source-contact 到
   next-rank-contact 的触点对；
3. 不判断 A/B，也不为事件赋模板标签；无论实际遍历方向如何，都把同一触点对折叠
   到轴坐标递增的 canonical edge；
4. 所有 canonical edges 形成一个带权多路径 skeleton；
5. forward graph 是 skeleton 的递增轴定向，reverse graph 是其严格转置；
6. 20% coarse-axis adjacency 只作为连通性 floor，80% 权重来自 train80 transition
   skeleton；
7. graph 以 spectral norm 缩放，传播增益由共享 RNN 学习。

RNN 输入层不得再次做 row-normalization。`A[target, source]` 保留构图阶段定义的
source→next-contact 条件边权；rank 1 使用 `0.5*(A_forward+A_reverse)`。重复按
target 行归一化会破坏 transition 权重，属于实现错误。

heldout20 只用于图构建完成后的审计和最终模型评估，不能反向修改图。发作数据禁止
读取。

## 3. 已通过的先验门

- 34/34 split-half skeleton cosine ≥0.8，中位数约0.982；
- 34/34 heldout transition NLL 优于 axis-only；
- 33/34 heldout transition NLL 优于同密度均匀图；
- 34/34 真实边权优于等密度 weight-shuffle，FDR后仍显著；
- train–heldout 边权 cosine 中位数约0.975，34/34 优于 weight-shuffle；
- 34/34 forward/reverse 为严格转置；
- 构图未读取 heldout 或 ictal target。

## 4. 模型与 rank

继承 v0.6 的受约束 dynamics：

- rank 0：无 recurrent state；
- rank 1：双向 skeleton 合并为对称图；
- rank 2：显式 forward/reverse 两通道，并有对称方向竞争；
- rank 3：rank 2 + global recruitment state；
- rank 4：rank 3 + local surround-suppression state。

主 screen 先比较 rank 1/2/4；rank 0 使用同合同静态对照。local patient offset 只能
进入静态 contact hazard，不能改变 skeleton 或 transition。

## 5. Screen gate

三位预先固定患者 × 三 seeds：

1. rank 1/2/4 相对 rank 0 的 next-set NLL；
2. participation、conditional rank、precedence；
3. label-free whole-path sliced Wasserstein；
4. 真实 skeleton vs weight-shuffle skeleton；
5. direction/global/surround/inhibition lesions；
6. 跨 seed 稳定性。

只有至少一个 structured rank 在 precedence 和 whole-path 上稳定优于 rank 0、
coarse-axis/weight-shuffle，并且相应 lesion 使结果变差，才启动34人正式 LOSO。

## 6. 发作期边界

screen 或34人间期门未通过时，禁止读取发作 target。通过后冻结模型与 rank，才评估
clinical onset `[0,10] s` 的 `1–150 Hz` baseline-robust-z 静态能量场。
