当前 2026-10-26 展示：D/E 均固定 1.5–2.9 s，E 包含前三次完整局部 burst（1.660–1.690、2.260–2.290、2.710–2.740 s）；下面的全段统计仍指 1.5–10 s。E 地图为 E/I 示意底图、灰色 non-core 背景与真实 core 阈值叠层；D/E 波形和 spikes 的 NPZ 均未改变。

# 原生 SNN D/E 数据与当前展示

D 来自 `run04_d0_poisson/D0/trajectory.npz`，不是上轮静默 `run03_d0`。零 core 的阈值、全局权重和拓扑不变，所有 E/I 接受独立 Poisson 到达，平均外源率保持原静默 D 的 ν=1.317463/ms；无局部 OU、局部随机输入区、刺激 kick 或慢变量。E 的 `E_native.npz` 与上轮完全一致。

分析 1.5–10 s：D 全网 E/I 0.093827/0.318221 Hz，固定 ROI 0.097090/0.306745 Hz，既有全网/ROI burst 判据均为 0 次；E 仍有 26 次完整局部 burst。Raster 显示同一 ROI 的全部 847 E+191 I，原始 ID 排序。无选择性抽取细胞或时间片。

LFP (a.u.) 由同一 LFPRecorder 在同一位置计算周围 E 的 |I_E|+|I_I| 空间加权均值；这是模型电流读出，不是实测细胞外电压。波形按真实样本逐点绘制，D/E 分别使用约 10.08–18.49 与 −65.36–1143.79 的纵轴，避免小幅背景不可见。地图 D 只有 E/I 符号与读出加号，无 core/ROI 圆；E 保留实际阈值场与 core。

D 中全 E 的平均兴奋电流为 13.1192、抑制为 0.8121，I/E≈0.0619；只能描述稀疏亚阈值噪声背景，不能叫严格的 E/I 电流平衡态。D/E 是不同输入工作状态，不能解释为纯粹改变 core 数量的单因素因果对照。仅一个拓扑和噪声种子，无独立重复的总体结论。

当前运行设计：`native_de_data/D_poisson_design.json`；输入/阈值/拓扑身份：`D_poisson_applied_physics.json`；主交付设计：`native_de_data/design.json`；统计及事件：`observables.json`、`firing_rates.csv`、`events.csv`。`D0_*` 保留前轮历史来源；`diagnostic_threshold_D_native.npz` 为旧阈值消融，未绘制。

当前脚本为统一 SNN 工作区 `experiments/fig4_native_de_20261024/run_native_d0_poisson.py --reference-run experiments/fig4_native_de_20261024/run02 --output <新的运行目录>`，使用 cuda_env Python，OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS=1。`export_native_background.py` 更新 D，保留 E；`validate_native_de.py` 检查真实 spikes、LFP、固定窗口、零 core、稀疏和 E 数据保留。无需重跑仿真即可从现有 NPZ 重绘。
