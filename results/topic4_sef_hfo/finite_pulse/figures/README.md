# Step 0b/0d 结果图（LIF rate field，真实场）

> 当前 SEF-HFO 模型（LIF-derived rate field）的结果图，由 `scripts/plot_sef_hfo_step0_results.py` **重跑 canonical `integrate_lif_field` 取真实空间场**画出（不是汇总标量）。被取代的 sigmoid scaffold 旧图在 `../../_sigmoid_scaffold_SUPERSEDED/`。

### step0b_lif_self_limited.png

核心闸门图。上排四张**真实发放率场快照**：t=8ms 在刺激盘点着 → t=24ms 沿 θ_EE 轴推进 → t=44ms 传遍 patch → t=90ms 熄灭回静息（自终止）。下排左 = front(t) + 活跃比例(t) 时程：波前单调推进、活跃比例升起后**回零** = 自终止；下排右 = 响应随脉冲幅度：亚阈熄灭 / 过阈波前推进**饱和成平台**（全或无、与幅度无关）/ 峰值活跃比例始终在 runaway 阈值 0.5 以下（有界）。

**关注点**：四张快照要看出"点着 → 定向铺开 → 消失"；时程的活跃比例**回零** = 自终止；幅度响应的"阈值 + 平台 + 不过 runaway" = 全或无可激脉冲。sigmoid 场做不到（旧 scaffold 全熄灭）。

### step0d_anisotropy.png

承重判据图。四张**峰值场快照**：θ_EE = 0°/45°/90° 时活跃区明显**沿白色虚线（= 连接各向异性轴）拉长、随轴旋转**（front 拉长比 ≈ 4.2）；isotropic 对照 = 圆斑、无方向（比 = 1.0）。

**关注点**：一眼看出传播形状跟连接各向异性轴走、随轴旋转；isotropic 圆斑证明方向来自**连接**、不是网格/电极几何。这是把 SEF-HFO 与"几何采样伪象"分开的承重判据，PASS。
