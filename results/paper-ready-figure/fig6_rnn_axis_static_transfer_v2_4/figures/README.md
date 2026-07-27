### fig6_rnn_axis_static_transfer_v2_4.png

这张六面板图汇总修正版 RNN 的完整科学链。A–B 冻结分析顺序和分母；C 显示在原分析已支持 A/B 共线轴的 9 人中，RNN 选择轴相对候选轴中位数的 alignment margin 仍为负（中位数 −0.205），因此 seed 间完全重复并不等于恢复病理轴；D 显示 selected-axis 对 next-contact 有小幅 held-out 增益（中位数 0.00798），但 source term 无稳定增益。

E 以冻结 overlap 队列中排序第一位患者作 display-only 示例，展示 empirical train80 与 full RNN 的节点 participation/rank distribution；队列层面 full RNN 保留了粗粒度 participation（中位 Spearman ρ=0.922）和 expected-rank order（ρ=0.742），但概率分布仍比 isotropic 结果远离真实 empirical distribution。F 是 14 人、clinical onset 后 `[0,10] s`、1–150 Hz 静态能量场的 source-free LOSO readout：full RNN 的 all-contact null margin 中位数为 −0.153，Gate S/H/X 全部失败；empirical rank distribution 的正向结果仅为 post-gate、未校正趋势，bootstrap CI 仍跨 0。

同目录 PDF 由同一代码和输入重新生成，metadata 记录 panel 来源、gate 状态及示例选择规则。逐发作 exact clinical-onset source 仍缺失，因此本图不包含 source-conditioned ictal rollout。

**关注点**：这是一张 bounded-negative / supplementary closeout 图。它支持“间期 rank 序列可自监督预测、节点级 rank 概貌可被低容量 RNN 近似”，但不支持“RNN 自动恢复患者 A/B 病理轴”或“当前 RNN 可解释发作早期能量场”；也不能反过来否定论文已有的 empirical A/B axis 和 interictal–early-ictal field 结果。
