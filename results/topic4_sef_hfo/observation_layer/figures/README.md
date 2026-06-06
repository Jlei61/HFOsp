# results/topic4_sef_hfo/observation_layer/figures

### increment1_gate.png

增量 1「已知方向玩具波合同门」结果。左图：对 0/30/60/90/135° 五个已知传播方向的合成
平滑行波，虚拟电极读出的通道先后顺序与真方向投影顺序的 Spearman（虚线 = 通过线 0.9）。
右图：endpoint-centroid 轴与真方向的夹角误差（虚线 = 25° 容差）。两图都越过线 = 观测
层能把已知方向干净读回来。负对照（C1 居中径向 = 无轴、C2 同起同落幅度差 = 无假序）的
裁决在 `../increment1_toywave/gate_verdict.json`。

**关注点**：五个 wave 必须全部 Spearman ≥ 0.9 且角误差 < 25°；任何一个不过 = 观测层
被几何污染，停下来调链路（绝不放宽阈值）。
