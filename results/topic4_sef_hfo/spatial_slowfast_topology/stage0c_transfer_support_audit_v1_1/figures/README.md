### stage0c_transfer_support_audit_v1_1.png

这张图是 v1 transfer 验证失败后的纯数值修复审计，不改变候选点或模型方程。左上是 stable exact reference；右上用固定 probe_rest 对比 v1 fine（虚线）与 extra-fine（实线）；左下给出 extra-fine authoritative 分类；右下列出数值门。它不包含 slow lifecycle 或空间耦合。

**关注点**：extra-fine 是否先通过 v1.1 新锁定的 conservative 0.25 Hz / 2% 门，以及是否留下至少两初态支持的有限对象。
