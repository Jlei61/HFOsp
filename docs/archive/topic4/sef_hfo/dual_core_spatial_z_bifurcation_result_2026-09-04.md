# data-driven dual-core 的 spatial-Z 分岔结果（2026-09-04）

## 结论先行

在冻结的 `dualcore_s39 + Joint=1.25` 底物上，把原来均匀的 frozen `q` 改为沿两个 core 和 surround 分区的
`Z(x)` 后，1 mm 粗粒化确定性快子系统存在经过双重核验的 **saddle-node fold chain**。这不是旧的连续自由场，
也不是把两个 core 合并成一个均匀 `q` 后得到的曲线。

最重要的机制结论是：这个分岔结构控制 **runaway 是否成为确定性必然**，但不精确控制持续 OU 噪声下的实际
转变时刻。OU-on SNN 的中位转变发生在低根和高根仍共存的区域，因此噪声可以先触发 basin crossing；若 Z
继续耗竭到低支 fold，低根消失，runaway 才变成确定性快系统唯一可达的大态方向。

## 1. 冻结底物与空间 Z 定义

- SNN 底物：`dualcore_s39 + Joint=1.25`，`g_EE=0.625`，`g_EtoI=1.25`，E→E ellipse angle 45°、aspect 2。
- 拓扑种子：2542；32,000 E + 8,000 I；两个 core 合计 1,499 个 E 细胞。
- core A/B 中心：`(1.54, 1.23)` 与 `(18.61, 2.42) mm`；按精确二值 core membership 投影到 10×10 网格。
- 确定性 reduction：同一张 realized graph、E 阈值场、外部平均驱动和 LIF transfer。这里 sheet 为
  20 mm、10×10 网格，所以每格宽度是 **2 mm**。早期口头所称“1 mm coarse”在本实现中不正确，
  本文与主文档一律按实际 `cell_width_mm=2.0` 报告。
- mixed cell 的抑制输入均值使用格内 `E[Z]`，抑制输入方差使用 `E[Z²]`；不能把后者错写成 `E[Z]²`。
- spatial-Z 主路径：

  ```text
  Z_A = 1 - s
  Z_B = 1 - s
  Z_surround = 1 - 0.70 s
  ```

  `s` 越大表示抑制效能越弱。`0.70` 来自 rev21 OU-on 转变时 surround/core 耗竭比的预先冻结近似；归档后
  从 100 个 active run 重新逐文件提取得到实际中位比为 0.713，没有再用它回调 fold。

这里的 “data-driven” 只修饰 **两个 core 的几何和冻结 SNN 底物**。`Z_A/Z_B/Z_surround` 不是从患者测出的
抑制场，而是模型状态在该几何上的空间投影。

## 2. 怎样认定是真 fold

每个候选都必须同时满足：

1. pseudo-arclength continuation 的 `ds/dℓ` 变号，即分支在控制参数上折返；
2. 同一 fixed-point residual Jacobian 有一个实特征值穿过 0；
3. 两个证据落在同一对 continuation 样本中；
4. 两侧 fixed point 都是物理根，残差通过阈值。

因此，普通 parameter sweep 中“求解器突然跳支”不算分岔证据。

## 3. 两个承重边界

### 3.1 低支消失：runaway-entry fold

- `s = 0.3375913801`
- `Z_A=Z_B=0.6624086`，`Z_surround=0.7636860`
- fold 上 mean E rate 仍为 `0.0625 Hz`；这是低根消失，不是先连续爬到高率。
- tangent bracket：`+0.12982 → -0.07403`
- Jacobian 实零模 bracket：`-7.26e-5 → +4.11e-5`
- critical E mode 的能量 `93.69%` 落在 core A；峰位于 `(1,1) mm`。

这说明低态不是全片同时软化，而是先由 core A 的局灶模式失去恢复力。超过该边界后，低 fixed point 不再存在；
在这个确定性快子系统里，这一 fold 直接控制 “runaway or not”。

### 3.2 招募支恢复：recruited-recovery fold

- `s = 0.0905892333`
- `Z_A=Z_B=0.9094108`，`Z_surround=0.9365875`
- fold 上 mean E rate `100.45 Hz`；regional E rate 为 core A `31.03 Hz`、core B `95.09 Hz`、surround
  `102.22 Hz`。
- tangent bracket：`-6.61e-4 → +2.07e-3`
- Jacobian 实零模 bracket：`+6.46e-5 → -2.02e-4`
- critical E mode 主要在 core B + surround：能量占比 `42.0% + 58.0%`，core A 近乎不参与。

从持续招募态往回恢复抑制时，这个 fold 标记一条 recruited branch 的消失。完整 continuation 另外还有 4 个
折点；它们对应不同空间块依次加入/退出，而不是数值噪声。双核系统因此是 spatial recruitment snaking/fold
chain，不宜再画成唯一光滑 S 曲线。

## 4. 与 OU-on SNN 的关系

对 rev21 `coarse + timescale` 的 100 个 operational-detector-positive run，逐个在 `t_op` 最近的 slow-state
样本读数：

- `Z_core = 0.71264`（q10–q90 `0.68110–0.75386`）
- `Z_surround = 0.79522`（q10–q90 `0.77957–0.81108`）
- 对应主路径 `s_SNN = 1-Z_core = 0.28736`
- 该路径预测 `Z_surround=0.79885`，与实测中位 0.79522 接近；最大 onset-to-saved-sample 误差 0.50 ms。

`s_SNN=0.28736` 位于 `0.09059 < s < 0.33759` 的 root-coexistence 区。就在这个同一 spatial-Z 点，确定性
reduction 同时有：

- 低根 mean E `0.0503 Hz`
- recruited tonic 根 mean E `352.83 Hz`

所以 OU-on SNN 不是等到低支 saddle-node 才跳；它在双稳/多稳区中提前被噪声推过 basin boundary。这也解释
了 topology×dynamics seed interaction：fold 给出 susceptibility landscape，OU realization 决定哪一次、从哪个
core 穿过去。

为避免只凭“两个根”就称双稳，在 `s_SNN` 处另算 zero-delay、operating-variance-frozen dynamic Jacobian：低根与
tonic 根的最大实部分别为 `-0.02237` 与 `-0.02941 ms^-1`，在这一明确标注的 sensitivity 下两者都稳定。这不是
含传导延迟的完整稳定性定理。

## 5. 两个 core 是否必须同步耗竭

固定 `Z_surround=0.80`，对 `Z_A×Z_B = 0.55–0.95` 做 11×11、每格 4 个初值的有限 root catalog：

- recruited tonic 根在当前网格全部可找到；
- 低根是否仍存在呈 L 形边界：core A 降至约 `0.63–0.67`，或 core B 降至约 `0.55–0.59`，任一侧单独足够低
  都可移除低根；
- core A 的门槛更早，和 runaway-entry 零模 93.69% 定位 core A 相符。

因此，**两个 core 不需要 homogeneous、也不需要严格同步耗竭**。对称 `Z_A=Z_B` 只是用于定义一条可比较的
主 continuation path；真正的空间系统允许单核先触发，随后通过 realized graph 招募另一核和 surround。

## 6. 科学边界与下一步

当前可以写：冻结 data-driven 双核底物的确定性 fast subsystem 存在空间 saddle-node fold chain；其中低支 fold
是 runaway 的确定性边界，OU 噪声可在 coexistence 区提前触发持续高态。

当前不能写：

- 患者组织存在同样的 Z 场或同样数值的临床发作阈值；
- 这是热力学意义的 phase transition；
- 这是 delay-aware 的完整稳定性定理；
- 两个 core 在 SNN 中始终以对称 Z 演化；当前 recorder 只保存 union-core Z，不能反推 `Z_A(t)` 与 `Z_B(t)`；
- 该结果已跨 coarse resolution / topology seed 复现。

本轮 frozen fast subsystem 只 continuation 了 Z，`eta_m=0`；M 与 OU 仍保留在 full SNN 的动态证据里。下一步最小
验证是：增加 per-core Z recorder，做 3 topology 的 coarse-resolution sensitivity，并把 full-SNN transition 在
`(Z_A,Z_B,Z_surround)` 相图上逐时刻投影；在此之前不把本图直接升格为正式 Fig.5 panel。

## 7. 产出

- 模型构建：`scripts/build_topic4_dual_core_spatial_z_meanfield.py`
- spatial-Z 数学与 continuation：`src/topic4_dual_core_spatial_z.py`
- 分岔 runner：`scripts/run_topic4_dual_core_spatial_z_bifurcation.py`
- 诊断图：`scripts/plot_topic4_dual_core_spatial_z_bifurcation.py`
- 机器结果：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/bifurcation/dualcore_spatial_z_bifurcation.{json,npz}`
- 诊断图：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/bifurcation/figures/dualcore_spatial_z_bifurcation_diagnostic.{png,pdf}`
- 图说明与 metadata：同目录 `README.md`、`dualcore_spatial_z_bifurcation_diagnostic.metadata.json`
- 测试：`tests/test_topic4_patient_zm_meanfield.py` + `tests/test_topic4_dual_core_spatial_z.py`，14/14 PASS。
