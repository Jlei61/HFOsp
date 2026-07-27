### v2_2_1_closeout_diagnostics.png

A 比较同一 22 人 heldout20 中 Markov、局部各向同性传播模型和轴向传播模型相对 node-bias 的 NLL 改善。B 检查冻结 checkpoint 是否系统性高估下一rank set 的大小。C 检查 local/axis kernel 是否共线，以及 full 与isotropic 的有效算子实际相差多少。D 区分“优化重复”与“结构可辨识”：横轴是 learned axis 与植入点云 PCA1 的关系，纵轴是移除 axis mixing 后heldout logit 的实际变化。

**关注点**：Markov 阳性是否独立于传播模型、损失是否主要来自 set-size/negative contacts，以及稳定的 axis 参数是否只反映固定植入几何。

### v2_2_1_closeout_diagnostics.pdf

与 PNG 内容一致的矢量版本，用于补充材料排版。

**关注点**：所有数值均来自冻结结果复算；未重训、未读取 early-ictal target。
