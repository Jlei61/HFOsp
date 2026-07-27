### transition_signal_decomposition.png

A 先判断一阶 Markov 增益在控制同 shaft 和欧氏局部距离后是否仍存在。B 把train-only conditional log-hazard residual 分成对称和反对称部分。C 检验物理轴 residual 以及由 observed source 连续决定符号的方向项。D 比较ordered multi-step prefix 与 last-rank Markov。

**关注点**：只有 A 的跨局部残差、C 的 source-conditioned axis 和 D 的多步历史同时在 heldout 患者中成立，才允许建立下一版 recurrent model。

### transition_signal_decomposition.pdf

与 PNG 内容相同的矢量版。

**关注点**：所有模型共享 event/prefix/contact denominator 和 STOP；图中零线表示复杂项没有带来 heldout 增益。

### transition_signal_decomposition_paper_ready.png

Paper-facing 四块版本。A 去掉数学上等价的重复 Markov 参数化，加入 20 位合格患者的正式跨-shaft conditional-likelihood 结果；B 区分 symmetric residual 与额外 skew；C 同时展示 axis residual 和 source-conditioned modulation，并用蓝/橙点标出 axis coefficient 正负；D 比较 source-only、last-rank 和 ordered history。

**关注点**：数据支持跨局部几何的 transition signal 和多步历史，但 source-conditioned 增益很小；14/22 患者的 axis coefficient 为负，因此安全结论是“axis-aligned anisotropy”，不能直接写成“沿物理轴增强传播”。

### transition_signal_decomposition_paper_ready.pdf

与 paper-ready PNG 内容相同的矢量版，适合 Supplementary Figure 排版。

**关注点**：该图允许设计下一版最小 recurrent observation model，但本身不是 shared pathological scaffold 的机制证明，也不开放 early-ictal target。
