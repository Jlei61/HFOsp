# rev10-D6.1 执行计划

1. 对 D6 49 个场做零仿真自然比例 KMeans 与 contact-split cross-fit 重评分。
2. 在 fresh seeds 不可见时冻结 5 个不同候选和全部输入 hash。
3. 运行 5 候选 × 6 fresh networks × 16 s；先用单 worker RSS sentinel，再按内存上限并行。
4. 聚合每张网络的自然 KMeans、mode proportion、招募规模、K=2 对 K=1 held-out likelihood、cluster extent association 和 cross-fit patient matrix。
5. 做 paired candidate-minus-warm baseline 的 network bootstrap；平衡 A/B purity 仅作 secondary。
6. 输出标准 Fig.4 两图：直接波形和 KMeans/patient consistency；在完整患者 benchmark 未过时明确标记 diagnostic only。
7. D6.1 后再决定是否做低维连续场组合或转向 distributional objective；本轮不开放 edge、beta、topology、slow variables 或 optimizer comparison。
