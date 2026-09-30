# v3 Z/M 空间 rate 模型：方程定义（同一版用于积分、平衡、线性化、周期边值与变分）

代码：`scripts/topic4_zm_onset_rate_v3/`（`model_v3.py` 静态层、`dynamics_v3.py` 时域层与 CUDA 积分器、`transfer_spline.py` 传递样条、`lif_mc.py` 有色噪声 LIF 蒙特卡洛）。网络、几何、连接、阈值、延迟与 Z/M 参数与 v2 完全相同（`results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20`，graph identity 与原 Fig.5 参考一致）。**改动仅限率响应闭合**：静态传递函数与线性响应滤波。

## 1. 单位与索引

时间 ms，电压 mV，率 spikes/ms/cell。群体 g = 1..935（E 535 个 = 20×20 格 × 阈值块，I 400 个 = 格）；E 掩码 E_g；原细胞数 n_g；阈值 θ_g（I 恒 18 mV；核内 E 14.2–18 mV）；τ_m = 20 (E) / 10 (I) ms；不应期 2 / 1 ms；V_reset = 11 mV。突触：AMPA τ_r=0.7, τ_d=3.5；GABA τ_r=1.0, τ_d=20.61 ms。面积修正 area_X = 0.1/[τ_r,X(1−e^{−0.1/τ_r,X})]。外部泊松率 ν_g（基线 1.3175 /ms；配对运行时取原生记录的逐格逐毫秒值）。

## 2. 延迟算子

A, B：AMPA / GABA 一阶矩算子（每个靶细胞到达权重均值，按 0.1 ms 延迟 bin 分解，最大 358 bin）；QA, QB：二阶矩算子（逐个物理权重平方和的均值）。对率历史 r(t)：
`A[r]_g(t) = Σ_d Σ_h A_{g h}^{(d)} r_h(t − d·0.1 ms)` 等。

## 3. 状态（每群体 14 个）与方程

突触均值/方差滤波（与 v2 相同）：
```
τ_r,A q̇a = τ_m area_A A[r] − qa      τ_d,A i̇a = qa − ia
τ_r,G q̇g = τ_m area_G B[r] − qg      τ_d,G i̇g = qg − ig
(τ_A/2) v̇a = τ_m area_A² QA[r] − va   (τ_G/2) v̇g = τ_m area_G² QB[r] − vg
```
瞬时输入矩：`μ = ia − z·ig − m + p_μ`, `v_E = va + p_v`, `v_I = z²·vg`，其中 `p_μ = τ_m area_A J_ext ν_g`, `p_v = τ_m area_A² J_ext² ν_g`（私有外部泊松输入的扩散矩）。

响应滤波（本版新增，参数由 §5 标定；状态槽 0 的 μ_f 保留但未使用）：
```
τ_s μ̇_s = μ − μ_s ;  τ_cE v̇_Ef = v_E − v_Ef ;  τ_cI v̇_If = v_I − v_If ;  τ_vE v̇_Ev = v_E − v_Ev ;  τ_vI v̇_Iv = v_I − v_Iv
μ_eff = α μ + (1−α) μ_s + η_E (v_E − v_Ef) + η_I (v_I − v_If)
v_E,eff = a_E v_E + (1−a_E) v_Ev ,  v_I,eff = a_I v_I + (1−a_I) v_Iv   （φ 内截断 ≥0）
r = φ_pop(μ_eff, v_E,eff, v_I,eff; θ_g)
```
权重 (α, a_E, a_I, η_E, η_I) 为瞬时工作点 (μ, v_E, v_I; θ) 的光滑表函数。
M（η_M·M，mV）：`τ_M ṁ = 0.5 E_g r − m`，τ_M=1000 ms（η_M=0.0005/spike 与原生逐次 +1 规则一致）。
Z（仅 E）：`τ_Z ż = Φ((θ_Z − ig)/s) − z`，θ_Z = 95.19851312666987 mV，τ_Z = 5000 ms，s² = τ_m vg/(2 τ_G)（未缩放 GABA 电流的瞬时扩散方差）。Φ 为标准正态分布函数。这是原生逐细胞规则 `1[I_I < θ_Z]` 的群体高斯期望；群体均值 z 的方程对指示函数的均值精确成立，近似只在 P(I_I<θ_Z) 的高斯形式（原生逐细胞记录直接检验见 `diagnostics/native_z_closure_summary.json`：3–12.5 s 全局耗减驱动误差 <0.5%）。

## 4. 静态传递函数 φ

对每个群体（E 或 I 参数），φ 是与原生引擎离散化（0.1 ms；先突触衰减再跳变；膜 V←I_net+(V−I_net)e^{−dt/τ_m}；不应期计数）完全一致的 LIF 在高斯扩散输入（AMPA/GABA 各自双指数滤波、精确离散协方差）下的稳态率，由蒙特卡洛表给出：
- 坐标：x=(μ−V_reset)/(θ−V_reset)，σ_c=√v_c/(θ−V_reset)；阈值缩放不变性已数值精确验证。
- 网格：x 79 点（asinh 均匀，−25…218）、σ_E 25/22 点、σ_I 27/25 点（E/I）；每点 512 条共随机数路径 × 2 s（率 <30 Hz 的点加密到 2048 条）。
- 插值：log(max(r,10⁻⁴ Hz)) 的三次张量 B 样条（σ 轴镜像保证对 v 的导数在 v=0 有限；x>60 与确定性率 1/(τ_ref+τ_m ln(x/(x−1))) 光滑混合）。值、一阶、二阶导数解析；CUDA 与 CPU 实现逐位一致（相对差 <1e-12）。
- 与 v2 冻结闭合的差异及归因：`diagnostics/static_closure_attribution.json`。

## 5. 响应滤波参数（标定结果，2026-09-18 冻结，`response_closure/closure.json` + `.npz`，SHA256 见 `FROZEN_SHA256.txt`）

结构 V3（拟合过程与候选结构对比见 `response_closure/points_{E,I}.json`、脚本 `fit_response_closure_B.py`、`refit_variance_variants.py`、`refit_variance_v3.py`）：
- 均值通道：H_μ(λ) = α + (1−α)/(1+λτ_s)（快极点在拟合中退化到 0.05 ms 下界，故取直通分量）。
- 方差通道 c∈{E,I}：R_c(λ) = φ_vc[a_c + (1−a_c)/(1+λτ_vc)] + φ_μ η_c λτ_c/(1+λτ_c)。a_c∈[−1,1]（a_c=−1 配短 τ_vc 是纯延迟的一阶 Padé 形式，对应低率处“幅度≈1、相位滞后”的实测形状）；η_c（mV/mV²）是方差快瞬变以等效均值偏移进入率的增益。
- 全局极点（ms）：E：τ_s=13.43, τ_vE=3.87, τ_cE=10.96, τ_vI=7.32, τ_cI=16.11；I：τ_s=8.18, τ_vE=2.40, τ_cE=5.70, τ_vI=4.34, τ_cI=8.65。
- 权重 (α, a_E, a_I, η_E, η_I) 为工作点 (x, σ_E, σ_I) 的三次张量样条表（粗网格 E 16×6×6、I 15×5×6；σ 轴镜像保证 ∂/∂v 在 v=0 有限；网格点权重经 σ=0.6 格点的高斯平滑以约束表梯度——未平滑/未镜像时线性化切向方程在 (v−v_f)·∂a/∂v 项上刚性发散）。
- 拟合带 2–40 Hz；80 Hz 不拟合。均值驱动高率点在 60–80 Hz 出现的相位超前/共振不在闭合内。
- 样本内拟合质量（归一化复误差中位数，E/I）：均值 0.047/0.033；方差_E 0.044/0.030；方差_I 0.077/0.047（`closure.json` summary，平滑前）。保留验证结果见 `dynamic_assay/validation_result.json`。方差通道在网络回路增益中的权重：间期态 ~0.10–0.13、高活动态 <0.005（`diagnostics/variance_channel_network_weight.json`）。

## 6. 线性化

平衡点 r*（D 或 Z 场给定）：特征矩阵
`M(λ) = I + diag(0.5 E φ_μ H_μ(λ)/(1+λτ_M)) − K(λ)`，
`K(λ) = diag(g_μ τ_m)[area_A h_A(λ) A(λ) − diag(z area_G h_G(λ)) B(λ)] + diag(g_E τ_m area_A² h_vA) QA(λ) + diag(g_I z² τ_m area_G² h_vG) QB(λ) + (动态 Z 项)`，
g_μ = φ_μ H_μ，g_c = φ_vc[a_c+(1−a_c)/(1+λτ_vc)] + φ_μ η_c λτ_c/(1+λτ_c)；A(λ) 等为延迟算子的拉普拉斯像（Σ_d e^{−λ d} A^{(d)}），h_A = 1/[(1+λτ_r,A)(1+λτ_d,A)]，h_vA = 1/(1+λτ_A/2)。权重对状态的导数在平衡点处不出现（乘以零的失衡量）；在周期轨道的变分方程中保留（`floquet_v3.py`）。动态 Z 项：δz = E[(∂z_∞/∂ig) h_G τ_m area_G B(λ) + (∂z_∞/∂vg) h_vG τ_m area_G² QB(λ)]δr/(1+λτ_Z)，进入 μ（−ig δz）与 v_I（2 z vg δz）。M(0) = −(静态 Jacobian)（数值核验为 0 差）。

## 7. D 与 Z 场

D = 1 − Σ_E n_g z_g / Σ_E n_g。规定路径：原生 9.420 s 逐细胞 Z 场的幂变换 z_i^a 投影到群体（与 v2 相同）。阶段 B 另用实际轨迹的 Z 场（`set_Z` / `set_Z_cells`），D 仍为真实加权均值。
