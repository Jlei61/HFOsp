# Topic 5 / Figure 6：event-persistent path-mode graph RNN（v0.9）

**日期**：2026-07-26
**状态**：执行合同
**继承**：34人、masked contact rank、chronological 80/20、shared-core LOSO、
heldout train80 local calibration、target sealed

## 1. 为什么需要 v0.9

v0.7 的 train80 多路径骨架稳定，并改善 heldout next-contact NLL；相对同密度边权
打乱，也改善 participation 和 label-free whole-path distance。但是它没有改善
pairwise precedence。失败模式说明平均图包含真实路径信息，但同一事件内没有保持一个
一致的路径/方向状态。

v0.9 不再增加普通 hidden rank。一个事件在开始时选择一次 latent component
`(path mode k, direction ±)`，该 component 持续到 STOP，事件中途禁止切换。

## 2. 输入与封存

- 输入只来自 `dataset_v0_4` 的 contact-rank 间期事件。
- 每位患者只用 chronological train80 构造路径模式。
- heldout20 不参与构图、选择 K、初始化或调参。
- 禁止使用 A/B label、KMeans template label、IEI 或 seizure target。
- clinical onset `[0,10] s`、`1–150 Hz` baseline-robust-z 发作场保持封存，直到
  34人间期门通过。

## 3. 患者特异路径模式

每个 train80 事件先转换为 canonical increasing-axis transition vector。反向事件的
边折叠到同一递增轴表示，因此 path identity 与方向分离。

对所有事件的归一化 transition vector 累积 edge-by-edge co-occurrence matrix，再做
非负分解得到 K 个非负 path bases：

- K=1 必须等于 v0.7 aggregate transition skeleton；
- K=2/3/4 表示同一患者内可重复出现的多条路径；
- 每个 base 与 20% coarse-axis floor 混合，并作 spectral normalization；
- forward operator 为 base 的递增轴方向，reverse operator 为严格转置；
- mode prior 由 train80 事件对各 base 的非负相似度汇总得到；
- 分解固定 seed，不随 RNN seed 改变。

## 4. 模型与无未来信息的训练

每个 path mode 有 forward/reverse 两个 component，二者共享全部动力学参数。对一个
事件，component 在开始时从 train80 mode prior × 等概率方向先验中选择，并保持到
事件结束。

训练不生成 hard mode label。每个 component 独立计算 teacher-forced sequence
likelihood；模型以 component prior 对完整序列 likelihood 做 log-sum-exp
marginalization。任一步的预测只能用先验和此前已经观察到的 rank 更新 component
posterior，禁止用未来 rank 选择当前 mode。

患者 local offset 只能进入静态 contact hazard，不能改变 path graph、mode prior 或
共享动力学。

## 5. 结构对照

- `no_history`：无 recurrent state；
- `merged_path`：v0.7 的单个对称 aggregate graph；
- `intact`：K 个 event-persistent forward/reverse path pairs；
- `weight_shuffle`：每个 mode 内打乱 edge weight，保留患者轴、密度和 K；
- `mode_shuffle`：对每条 edge 独立打乱其 mode assignment，保留 aggregate edge
  mass、患者轴和 K，但破坏同一 mode 内的 coherent path；
- lesions：去除最常用 mode、去除 forward 或 reverse components、去除 shared
  inhibition、把 K modes collapse 成 aggregate graph。

## 6. 低成本筛选门

预先固定患者：`epilepsiae_1073`、`epilepsiae_1146`、
`yuquan_chenziyang`；seeds：`20260726/27/28`。

K=1/2/3/4 均运行。候选 K 必须同时满足：

1. 相对 `no_history`、`merged_path`、`weight_shuffle`，precedence MAE 和
   whole-path distance 均在至少 6/9 patient-seed 改善；
2. 上述每个比较均至少 2/3 患者的三-seed中位 benefit > 0；
3. K≥2 时，相对 `mode_shuffle` 也满足同一门；
4. 跨 seed 指标排序稳定：每名患者分别对 K=1–4 排序，三个 seed pair、三名患者
   合并后的 Spearman 中位数对 precedence 和 whole-path 均须 ≥0.40；
5. K=1 用对称的“去掉一个方向”平均 lesion，K≥2 用 mode-collapse 或
   dominant-mode lesion；对应 lesion 的 precedence 与 whole-path 均须满足同样的
   6/9 patient-seed 和 2/3 patient 门；
6. 选择满足全部门的最小 K。

只改善 next-contact NLL、participation 或单一 whole-path 指标不算通过。

## 7. 34人正式门

筛选通过后冻结 K、先验构造、模型、损失和训练器。正式运行要求：

- 34人 × 3 seeds；
- outer 33人 shared-core 训练；
- heldout 患者只用其 train80 校准 local offset；
- exact coverage，不以随机 steps 代替；
- patient-level seed median；
- 对每个主对照，precedence 和 whole-path 的 median benefit > 0、改善患者
  >17/34、方向性 Wilcoxon FDR q<0.05；
- 对应 lesion 同方向恶化。

全部通过后才解封发作场。未通过则停止在可复现 bounded negative，不进行无界调参。

## 8. 最终交付

模型、构图、训练/恢复/监控/汇总代码；单元测试与 falsifier；运行清单、fingerprint、
checkpoint、资源峰值；mode occupancy、事件内 posterior trajectory、方向/模式/抑制
lesion、节点分布；paper-ready 六块图及 README/metadata/PDF；论文 Methods、Results
和 claim boundary。
