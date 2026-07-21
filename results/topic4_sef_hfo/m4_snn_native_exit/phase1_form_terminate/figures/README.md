# Phase-1 form-then-terminate（先测成形、再介入）

问：那个已经**成形**的 M4 宽有界持续态，能不能被"活动持续够久才积累"的局部恢复电流干净地终止、并回到能重新产生原有间期事件的态？（对比被杀的 estfork：那是盲设 2500ms 成形；这里 t_form 是从无-p anchor **测**出来的。）

### formed_state_diagnostic_p1anchor_seed1.png
无-p 的 M4 anchor（T=12000, seed1）的成形诊断。三列左：总/核/周发放率、q_core/q_surround/q_mean、S_G+active-area；右：沿轴+横轴 kymograph；右下：t_form + 敏感性。**读出**：宽有界态**确实成形**——核率→~170Hz、周率→~74Hz、S_G→0.42、q_core→0.05、q_surround→0.10、面积→0.62，kymograph 两轴填满并稳住。数据驱动 **t_form=1600ms、敏感性完全稳（spread 0）**，全变量 85% plateau 在 ~2150ms。
**关注点**：这个有界态是**宽条带**（q_core≈q_surround≈地板、几乎无核-周梯度，正是 spec §9 "broad ~60% stripe, NOT a localized core"）——所以成形判据**不能**要"核-周 q 梯度"（会错杀真态），改判"宽招募+宽耗竭"。虚线=onset。

### arms_intervene_seed1.png
两条干预臂（不对称 p，η=80/150，**onset=2300ms**：85%成形在 ~2150ms、τ_up=3000 使电流 ~3370ms 才真起效、稳在 plateau 之后）。三行=率 / q_I / S_G+p。**判决=两条都 termination-no-go / fragment**。失败机制（图上清楚可见）：
1. 电流把成形有界态**压到 0**（~4000–5000ms）——它**能压下**；
2. 活动一停，**S_G 塌到 0**（S_G 是活动驱动、快衰减）→ **containment 消失**；
3. q_I 回灌到 ~0.76；但慢衰减的 p 电流（τ_down=12000, Φ(p)·η≈13–24mV）**过度压制**——不是回到 ~3/s 的间期 IED 列，而是只在 11.8/13/14.5s 冒**几个稀疏击穿 burst**（核在 q_I 回灌后再点火，非恢复原 IED）。
**关注点**：两严格合同都过（介入前逐字一致=True、onset 处确在成形态=True），所以这是对**已成形态**的干净判决。压下有效、但"停不住/会反弹"+过度压制 → **不是 lifecycle**。指向：(a) containment-memory H（在 q_I 回灌期保住 divisive 兜底、不靠活动驱动的 S_G）；(b) 持续时间门控 h/θ_p（让短 IED 不充 p、释放后能恢复）。⚠️击穿 burst 空间上仍是宽条带、非原紧凑 IED；seed 1；后段电流 prevention-contaminated（见 prevgated80：门控电流仍把 onset 后 IED 从 29 压到 11）。
