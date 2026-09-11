# data-driven dual-core 的 spatial-Z 分岔结果（2026-09-04）

## 结论先行

在冻结的 `dualcore_s39 + Joint=1.25` 底物上，把原来均匀的 frozen `q` 改为沿两个 core 和 surround 分区的
`Z(x)` 后，2 mm 粗粒化确定性快子系统存在经过双重核验的 **saddle-node fold chain**。这不是旧的连续自由场，
也不是把两个 core 合并成一个均匀 `q` 后得到的曲线。

最重要的机制结论是：低支 fold 控制 **低 fixed point 是否仍存在**，但不精确控制持续 OU 噪声下的实际
转变时刻。OU-on SNN 的中位转变发生在多个 fixed points 仍共存的区域；若 Z 继续耗竭到低支 fold，低根才
消失。含真实 delays 的工作截面显示高率 fixed points 可线性失稳，因此当前不把这个过程简化成两个稳定根之间
的 basin crossing，也不声称 fold 后高态是唯一吸引子。

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

### 3.1 低支消失：low-state fold（旧名 runaway-entry，已撤回）

- `s = 0.3375913801`
- `Z_A=Z_B=0.6624086`，`Z_surround=0.7636860`
- fold 上 mean E rate 仍为 `0.0625 Hz`；这是低根消失，不是先连续爬到高率。
- tangent bracket：`+0.12982 → -0.07403`
- Jacobian 实零模 bracket：`-7.26e-5 → +4.11e-5`
- critical E mode 的能量 `93.69%` 落在 core A；峰位于 `(1,1) mm`。

这说明低态不是全片同时软化，而是先由 core A 的局灶模式失去恢复力。超过该边界后，低 fixed point 不再存在；
但原生延迟轨迹随后可以进入 bounded spatially localized attractor，而不必进入全片 runaway。因此旧标签
`runaway-entry fold` 过强：它只控制低 fixed point 是否存在，不能直接控制 “runaway or not”。

### 3.2 招募支恢复：recruited-recovery fold

- `s = 0.0905892333`
- `Z_A=Z_B=0.9094108`，`Z_surround=0.9365875`
- fold 上 mean E rate `100.45 Hz`；regional E rate 为 core A `31.03 Hz`、core B `95.09 Hz`、surround
  `102.22 Hz`。
- tangent bracket：`-6.61e-4 → +2.07e-3`
- Jacobian 实零模 bracket：`+6.46e-5 → -2.02e-4`
- critical E mode 主要在 core B + surround：能量占比 `42.0% + 58.0%`，core A 近乎不参与。

从持续招募态往回恢复抑制时，这个 fold 标记 global-recruited family 的边界之一。续接补长并经精细 restart
穿过密集折返后，该 family 在当前计算区间共有 14 个折点；它们对应不同空间块依次加入/退出，而不是一条唯一
光滑 S 曲线。

### 3.3 Fig.5C 必须画成多分支图谱，而不是单一 S 曲线

补长 continuation 后，原来深红线其实混合了两个不同的 full-state family：

- 从 runaway-entry 的零模配对根出发，续接 `8002` 个物理 fixed points，在 `s=0.144–0.355` 内检测到
  6 个折点；该族主要是 core-A-localized spatial states。
- 从 outer tonic root 出发，经 4 次 bit-identical restart 拼接 `21,332` 个校正点，在
  `s=0.0906–0.2489` 内检测到 14 个折点；该族从 global-recruited state 进入以不同空间块加入/退出为特征的
  recruitment snaking。

完整 200 维 E/I 状态按相同 `s`（容差 `2e-5`）比较时，两族最近仍相差 `7.29 Hz` RMS，远大于
`1e-3 Hz` 的相同根阈值。因此当前证据支持“它们是已算区间内不同的 branch loci”，不支持把灰线和棕线手工
补成一条。反过来，这也不是二者在尚未续接区间永不相连的全局拓扑证明。

正式 Fig.5C 现在把所有 continuation loci 画成实线、全部折点画空心圆；**线型不再编码稳定性**。这是有意删除
旧版的 deep-red dashed/dotted grammar：旧线型只来自 zero-delay、operating-variance-frozen Jacobian，不能
外推到含真实传导延迟和 OU 噪声的系统。

## 4. 与 OU-on SNN 的关系

对 rev21 `coarse + timescale` 的 100 个 operational-detector-positive run，逐个在 `t_op` 最近的 slow-state
样本读数：

- `Z_core = 0.71264`（q10–q90 `0.68110–0.75386`）
- `Z_surround = 0.79522`（q10–q90 `0.77957–0.81108`）
- 对应主路径 `s_SNN = 1-Z_core = 0.28736`
- 该路径预测 `Z_surround=0.79885`，与实测中位 0.79522 接近；最大 onset-to-saved-sample 误差 0.50 ms。

`s_SNN=0.28736` 位于 `0.09059 < s < 0.33759` 的 fixed-point coexistence 区。扩大 multi-start 后，同一
spatial-Z 截面实际找到 6 个不同的 full-state roots；按 mean E 排序为 `0.0503, 0.0874, 0.0913, 40.20,
41.25, 352.83 Hz`。其中两个低率 saddle 分别偏 core A / core B，两个约 40 Hz 的根在 core A 已高率但在
core B 和 surround 的招募程度不同。投影到 Fig.5C 的 core-A 均值后，这 6 个状态只剩 4 个可分高度，这正是
空间异质根不能被一条 homogeneous S 曲线代表的原因。

稳定性另外按实际网络 delay bins 重算。粗粒化算子保留全部 realized delays（`dt=0.1 ms`，最长
`35.4 ms`）；native-step tangent map 同时包含 transfer 的 mean gain 与当前 diffusion variance 的即时增益
`∂Φ/∂σ · ∂σ/∂r`。power iteration 使用 3 个 seed，并要求每个 seed 的最后两个 post-burn-in 时间段同号才
分类。在这个**单一工作截面**：

- 低根为 stable，增长率中位数 `-0.02259 ms^-1`；
- 其余 5 个根均为 unstable，增长率中位数范围 `+0.01006` 到 `+0.05713 ms^-1`；
- 特别是 zero-delay 下看似稳定的 352.83 Hz tonic fixed point，在加入真实 delays 后不再可标为稳定。

再用原 40,000-cell `SpatialOUDrive` 生成 OU 场、投影到同一 2-mm 网格，并在非线性延迟模型中逐步重算递归和
外源方差。3 个 OU seed × 6 个初始根共 18 次、300 ms 轨迹中，7/18 的 tail 最近根仍是初始根；两个高率代表
根各为 3/3 保持在邻近高活动区域。这个结果说明线性 fixed-point instability 不等于 300 ms 内离开高态；它可能
落到附近的延迟驱动振荡/空间模式，但当前没有用收敛的复谱证明 Hopf，也没有建立高态吸引子的完整拓扑。

因此可写的关系是：OU-on SNN 在低根 saddle-node 消失前已进入持续高活动 regime；fold 组织 susceptibility，
但实际转变时刻由噪声与空间状态共同决定。不能再简写成“两个稳定 fixed points 之间的 basin crossing”。

## 5. 两个 core 是否必须同步耗竭

固定 `Z_surround=0.80`，对 `Z_A×Z_B = 0.55–0.95` 做 11×11、每格 4 个初值的有限 root catalog：

- recruited tonic 根在当前网格全部可找到；
- 低根是否仍存在呈 L 形边界：core A 降至约 `0.63–0.67`，或 core B 降至约 `0.55–0.59`，任一侧单独足够低
  都可移除低根；
- core A 的门槛更早，和 runaway-entry 零模 93.69% 定位 core A 相符。

因此，**两个 core 不需要 homogeneous、也不需要严格同步耗竭**。对称 `Z_A=Z_B` 只是用于定义一条可比较的
主 continuation path；真正的空间系统允许单核先触发，随后通过 realized graph 招募另一核和 surround。

## 6. Fig.5C/D 的尺度与响应语义核验

### 6.1 C 不是单神经元分岔

C 的底层是保留 realized incoming weights、格内经验 E-threshold distribution 以及 core/surround mixed-cell
组成差异的 2 mm E/I population reduction。因此系统是空间异质的，但图中纵轴只是把 core A 内各 coarse unit 的
E rate 按 E 细胞数加权平均。准确表述是 **heterogeneous spatial system 中的 core-A local population
saddle-node readout**；不是每个 E 神经元各自发生同一个分岔，也不是异质性分布图。

C 现在同时画两个实际 continuation family：从 tonic 外根出发的 global-recruited family，以及从低支零模配对根
出发的 core-A-localized family。两族均为细实线，空心圆只表示 `ds/dℓ` 变号的 continuation folds；没有 inset、
没有 deep-red stability line，也没有手工 connector。OU-on 中位截面的 6 个 full-state roots 以同一竖线上的
投影点标出，并注明投影后只有 4 个高度。

### 6.2 D 必须是同位置、同剂量的状态响应

正式 D 已恢复作者指定语义：在同一 `rev21_ts_tz3000_ta500`、topology 2542 / dynamics 2642 SNN 轨迹上，
先 exact replay 并在低态 `1000.0 ms` 与 early-ictal `2615.4 ms` 保存 checkpoint。重放的 `rate_E` 与原始
rev21 worker 在 26,155 个时间步上逐点完全一致。随后固定 4×4 分层随机的 16 个位置（seed 20260820）和
16-cell 弱脉冲，每个位置做 paired probe–sham；图中是 0–50 ms、去掉强制注入 frame 后的 descendant-only
signed spike difference，再对 16 个位置等权平均。所有位置均保留。

站点级结果显示该响应是 hotspot-dominated，而不是均匀易感性：

- low activity：mean `1770.4`、median `9.5` excess spikes/50 ms；最大站点占绝对响应 `63.0%`；16/16
  站点满足低态 E1 evaluability，未出现 probe-only ignition；
- early ictal：mean `2012.3`、median `-3.0`；最大站点占绝对响应 `87.7%`。此时 sham 已经处于高态，
  所以 0/16 满足“可再测试 ignition”的 E1 条件；这不删除站点，也不妨碍 D 作为 high-state incremental
  response contrast，但禁止把它写成发作触发概率。

因此，新 dual-core D 没有复现旧 `joint_04_control seed1801` 图中“early-runaway 响应近乎消失”的形状。
它显示的是响应热点从低态的中上部斜带重排到 early-ictal 的 sheet 上缘；总增量均值未下降。当前 D 在 panel
语义上正确，但科学证据仍是 single realized trajectory、single frozen dose 的探索性状态对比。

原先占用 D 的 runaway-entry zero-mode map 与 `Z_A×Z_B` finite root catalog 已改为
`fig5-supp-spatial-z-mechanism`，只作为分岔定位的机制补图。

## 7. `Z × tau_GABA` 振荡相搜索与数值收敛审计（2026-09-05）

上面的 continuation 只回答 fixed point 与 fold 的几何，不能回答高态是不是持续 30–80 Hz 振荡。为此在同一
topology 2542、同一 10×10 spatial-Z reduction 上增加非线性 realized-delay trajectory assay，扫描
`s × tau_d_GABA`，并把结局预先分成 low、localized/intermediate、tonic-recruited 与
oscillatory-recruited。最后一类必须同时满足：tail mean E rate ≥120 Hz，core A/core B/surround 各 ≥30 Hz，
整段主频 30–80 Hz、cycle-averaged peak-to-trough/mean ≥0.20，而且四个 250-ms 窗至少三个保持 30–80 Hz
且深度 ≥0.15。

数值收敛审计否定了初步 discovery 单格，但更完整的原生网格找到一个不同的窄窗：

- `dt=0.5 ms`、1.5 s 的粗扫描曾有 3 个 high-initial 单格通过；
- `dt=0.2 ms`、2 s 的 `s=0.24–0.28 × tau_GABA=5–9 ms` 精扫只剩一个通过点：
  `s=0.28, tau_GABA=8 ms`，31 Hz、mean 281.7 Hz、depth 0.223，4/4 窗通过；
- 同一点回到原生 `dt=0.1 ms` 后为 32 Hz、mean 301.3 Hz、depth 0.172，四窗稳定但只能归为 tonic；
- 另外四个原生点也全部是 tonic：`(s,tau)=(0.26,4),(0.26,6),(0.30,4),(0.30,6)` 的 depth 分别为
  `0.156, 0.197, 0.154, 0.135`。最接近的 0.197 点还有一窗明显塌陷，不能按阈值四舍五入为通过。
- 随后在原生 `dt=0.1 ms` 上完整重算 `s=0.24–0.28 × tau_GABA=5–9 ms` 的 25 个 high-basin 点、每点
  2 s：21 localized/intermediate、3 tonic、1 oscillatory。新通过点为 `s=0.27, tau_GABA=9 ms`，30 Hz、
  mean 286.5 Hz、depth 0.2068，4/4 窗通过，regional rate 为 `368.6/359.1/282.6 Hz`。
- 在该点周围用 `s=0.2675/0.2700/0.2725 × tau_GABA=8.5–9.5 ms` 做 15 格原生 zoom，只有
  `(0.2700,9.0)` 与 `(0.2725,8.5)` 两格通过；其余为 5 localized、8 tonic。通过点沿 localized→tonic
  边界的斜 ridge 排列，depth 仅 `0.203–0.207`，相邻格即可跌破 0.20。
- 把 `(0.27,9 ms)` 延长到 5 s 后，1–5 s 的四个连续 1-s 窗均保持 30 Hz，mean
  `286.4–286.8 Hz`，depth `0.2066–0.2068`；同参数 low-initial 5-s 轨迹保持低态。因此这里是低 basin 与
  sustained oscillatory-recruited high basin 的共存窗，不是低支 fold 后的唯一 runaway。

在此基础上又补了真正与 Z/M 语义一致的二维动力学图：固定 `tau_d_GABA=9 ms`，横轴扫描 spatial-Z
depletion path `s=0.24–0.36`，纵轴先扫动态 M 增益 `eta_m/rev21=0–2`，再补 4 与 8 倍边界，每格用原生
`dt=0.1 ms`、动态 `tau_M=500 ms`，并分别从 low/pre-fold 与 recruited 初态积分 2 s，共 77 格。合并后
low/pre-fold 初态为 58 low、19 localized/intermediate、0 global recruited；recruited 初态为 16 localized、
59 tonic、2 oscillatory。两个 oscillatory 格都在 `s=0.27`，M scale 为 0 与 0.5；rev21 M scale=1 的同一点
为 mean `273.9 Hz`、30 Hz、depth `0.185` 的 tonic-recruited state。提高到 4–8 倍 M 后，`s=0.27` 被压回
localized，而 `s≥0.28` 的 recruited basin 仍为 tonic；即 M 能把 recruited boundary 从约 0.27 推到 0.28，
但未生成新的宽振荡区。

low-root fold 仍在 `s=0.337591` 附近；由于 low state 几乎不放电，`m≈0`，在当前 0–2 倍 M 增益内其位置
几乎不受 M 影响。更关键的是，fold 后从标准化 pre-fold 初态出发并不会直接进入 global runaway：`s=0.34`
与 `0.36` 延长到 5 s 后分别稳定在 mean `67.6/73.8 Hz` 的 spatially localized state，core A/B 很高但
surround 只有 `51.5/57.6 Hz`。因此 fold 控制的是 **low fixed point 消失与离开 low basin**，不是
“全片 runaway 必然发生”的边界。该二维结果应称 native delayed Z×M regime/basin map，不称热力学相图。

随后把最接近区域中的 `tau_d_GABA=8 ms` 放回完整 40,000-cell SNN，保留 topology 2542、dynamics 2641、
dual-core field、continuous OU 与 Z/M 方程，只改原来 18 ms 的 GABA decay。该单种子在 `2544.7 ms` 到达
operational runaway，末 1 s mean E rate `262.4 Hz`，core A/core B/surround 分别为
`187.4/454.5/259.3 Hz`，所以是全区域高率招募；群体率主峰虽为 `52 Hz`，但 cycle modulation depth 只有
`0.064`，四个 250-ms population-depth 窗为 `0/4` 通过。逐触点 persistence 进一步同时要求 terminal 1 s
内至少 3/4 个 250-ms 窗和 7/10 个 100-ms 窗的主峰落在 30–80 Hz 且该带 RMS 高于自身 baseline，结果为
`9/15`：ICL `9/11`，SCL `0/4`。18-ms 同底物对照 terminal 为 `0/15`。

再把 GABA decay 缩到 4 ms，同种子 SNN 在 `2506.5 ms` runaway，末窗 mean `255.8 Hz`、群体主峰 `59 Hz`、
depth `0.079`，regional rate `442.5/457.5/246.3 Hz`。整段 1-s 指标看似 15/15 主峰入带、14/15 RMS 增强；
在 terminal 1 s 上只用 250-ms 粗窗甚至会得到 15/15，但缩到 100-ms 小窗后，四个 SCL 只通过 2–4/10，
联合 persistence 又回到 ICL `11/11`、SCL `0/4`。因此 SCL 是间歇高频 burst，不是持续 readout；单个整窗
PSD 或只有 250-ms 分窗都会产生假阳性，正式 A 必须使用多尺度 persistence gate。

另用预先按旧 18-ms 轨迹选择的 `si=0.8, M scale=0.5` 在 `tau_d_GABA=9 ms` 做动态 Z/M full-SNN 检查。
它在 `2982 ms` runaway，末 1 s mean `256.4 Hz`、主峰 `54 Hz`、depth `0.0365`，regional rate
`300.6/457.4/250.2 Hz`，但 terminal persistence 仍为 ICL `11/11`、SCL `0/4`。其 final mean Z 降到
`0.667`，说明改变快抑制时标同时改变了动态 Z 的工作点；不能用 18-ms 条件下 final Z≈0.72 直接宣称它复现了
frozen-Z coarse ridge。

最后把同一 `si=0.7` 条件继续推到 `tau_d_GABA=2 ms`，作为“仅缩短快抑制是否还会改善”的方向性边界。
它在 `2509.7 ms` runaway，末 1 s mean `205.0 Hz`、主峰 `66 Hz`、depth `0.0967`，regional rate
`438.1/451.6/193.2 Hz`；但联合 terminal persistence 仍是 ICL `11/11`、SCL `0/4`。SCL 的 whole-window
主峰已经全部进入 `60–72 Hz`，却只通过 2–3/10 个 100-ms 小窗，且多数 target-band RMS 不超过各自 baseline。
新 recorder 同时显示 final `Z_A/Z_B/Z_surround=0.547/0.503/0.690`：即使 core rule 对称，data-driven graph
与活动轨迹也会让两个 core 的 Z 动态分开，不能再用 homogeneous core-Z 叙述。8→4→2 ms 已把 SCL 从
11-Hz 慢成分推到目标频段，但没有提高持续 duty/能量；因此没有依据继续沿 `tau_GABA` 单轴外推。

这把原来混在一起的三个 readout 分开了：**regional recruitment 是 global，population modulation 是 shallow，
30–80 Hz contact readout 是 ICL-axis restricted**。因此 8/4/2/9-ms 结果都不是原 A 的简单替换：它们证明缩短
抑制时标确实能把病理轴上的 SEEG 波纹推入目标频段；4-ms 进一步让 SCL 出现高频 burst，但尚未产生作者要求的
持续全空间 runaway oscillation。

因此当前建立的是 **原生步长、5-s 持续、basin-dependent 的窄 oscillatory ridge 候选**，不是稳健 phase：
25 格主网只有 1 格、15 格 zoom 只有 2 格通过，且两格都贴近人工 depth=0.20 与 localized/tonic 交界。粗步长
下原 `s=0.28,tau=8 ms` 单格仍是数值伪影；另有部分 1.5-s “global” 点到 2 s 后退成 localized，属于长瞬态。
现阶段可写的是：缩短 GABA decay 能把高态推入 30–60 Hz，并在 deterministic 2-mm coarse fast subsystem
产生窄的持续高调制窗；不可写成已经发现有限面积、跨 topology、完整 SNN 复现的振荡相。

这也澄清了 saddle-node 的角色。低根 fold 仍位于 `s=0.337591`，而 discovery 中最接近目标的区域位于
`s≈0.26–0.30` 的多根共存区。fold 控制低 fixed point 是否存在，不控制高态的节律深度，也不保证 low-initial
轨迹落入高态 basin。把 E/F 画成真正动力学图时，必须分别呈现 fixed-point fold、有限时 basin outcome 和
native-dt 收敛状态，不能用一张插值热图把三者合并。

### 7.4 stable / tonic-runaway 分界的定向补算（2026-09-05）

作者明确接受近饱和 `tonic plateau` 作为本轮 runaway，不再要求深调制振荡。为避免沿用过宽的
`global recruited` 门槛，本轮在读取 10-s 精扫结果前冻结新 gate：末 1 s population E mean `>=300 Hz`、
core A/core B/surround 各自 `>=250 Hz`，且末 1 s 前后半窗均值差的绝对值 `<=5 Hz`；bounded state 则要求
population mean `<=250 Hz` 且同样通过驻留检验。该门槛是模型状态定义，不是临床发作阈值。

先检查了 core-A-localized fixed-point family 上 `s` 最大的 fold。加入 rev21 动态 M
（`eta_m=0.003725797`, `tau_M=500 ms`）后，pseudo-arclength 将其精确到：

- `s_fold = 0.3554508242`；`Z_A=Z_B=0.6445492`，`Z_surround=0.7511844`；
- fold 上 mean E `47.034 Hz`，core A/core B/surround 为 `410.307/121.264/35.851 Hz`；
- tangent bracket `+0.01743 -> -0.01203`，fixed-point residual Jacobian 实零模 bracket
  `+1.13e-3 -> -7.83e-4`，两者在同一数值区间折返。

因此这是一个真实的 spatial saddle-node，不是手画虚线或求解器跳支。但是它仍**不是**原生延迟系统的
stable/runaway 分岔：保留全部 realized delay bins、`tau_GABA=9 ms` 与动态 M 后，在 fold 两侧代表点的领先模
均为约 `37.26 Hz` 的复模，growth rate `+0.0314 ms^-1`；两侧 fixed points 都 delay-unstable。原生延迟系统中
实际驻留的是围绕这些空间前沿的振荡/部分招募 attractor，而不是“稳定 fixed point”。

随后从完全相同的标准化 low/pre-fold 初值做原生 `dt=0.1 ms`、动态 M、zero-OU 的 10-s 精扫。结果出现
清楚且已驻留的跃迁：

- `s=0.428`：bounded partial recruitment，末 1 s population `221.807 Hz`，core A/core B/surround
  `431.95/429.50/211.33 Hz`，半窗漂移 `-0.016 Hz`；
- `s=0.429`：near-saturated tonic runaway，末 1 s population `316.161 Hz`，core A/core B/surround
  `432.08/430.25/310.92 Hz`，半窗漂移 `-0.014 Hz`。

所以对这一固定准备条件，operational stable/runaway boundary 已夹在
`0.428 < s_c <= 0.429`，等价于 `0.571 <= Z_core,c < 0.572` 与
`0.6997 <= Z_surround,c < 0.7004`。该边界比 fixed-point fold 晚约 `0.073 s`。同一参数又明显依赖 basin：
例如 `s=0.40` 从 low-side 初值稳定在约 `105 Hz` 的部分招募态，而 recruited 初值稳定在约 `398 Hz` 的
tonic runaway。故当前严格结论是 **operational attractor/basin boundary**；在完成连续隐藏状态上/下扫或
periodic-orbit continuation 前，不把 `s_c` 写成已证明的 local bifurcation。

原生步长 carried-state 检验进一步排除了“只是标准初值在 `s=0.429` 恰好落到另一 basin”这一解释：先在
`s=0.428` 将 bounded state 驻留 10 s，再把 rates、四类 synaptic currents、全部 delay-history bins 和 M
逐位携带到 `s=0.429`，轨迹仍从约 `222 Hz` 离开，`2.889 s` 后越过 300 Hz，末 1 s 稳定在
`316.239 Hz`、regional `432.11/430.22/311.01 Hz`。因此 `0.428–0.429` 确实夹住 bounded attractor 的
上侧 escape boundary，而不只是固定初值分类线。由于相关吸引子是 delay-induced oscillatory state，尚未做
periodic-orbit continuation/Floquet multiplier，当前仍只称 **nonlinear attractor bifurcation/escape bracket**；
不能在 saddle-node of cycles 与 boundary crisis 之间定型。

连续隐藏状态上/下扫随后作为 discovery sensitivity 完成：使用守恒全部 pathway weight 的 5× delay-bin
coarsening（`dt=0.5 ms`），相邻 `s` 间不重置 rates、四类 synaptic currents、全部 delay histories 或 M。
从 `s=0.40` 的 bounded attractor 向上每 0.005 推进时，直到 `0.465→0.470` 才满足稳定 runaway gate；
反向从 `s=0.60` 的 runaway 下扫到 `s=0.30`，全程未退出 runaway。由于这是 discovery timestep 且每点只驻留
1 s，`0.465–0.470` 不能当正式临界值；但上下扫不重合已经证明强 hysteresis，并排除“所有初值共享一个
一维 stable/runaway bifurcation”的画法。正式 Fig.5E 应呈现 basin/hysteresis，而不是把 `s=0.35545` 的
equilibrium fold 或 `s=0.428–0.429` 的固定准备条件切片单独画成唯一相变线。

对应机器结果：

- adapted spatial fold：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/runaway_boundary/localized_to_global_fold_m1.{json,npz}`；
- native-delay spectrum：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/runaway_boundary/fold_native_delay_spectrum_m1.json`；
- continuous-state hysteresis discovery：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/runaway_boundary/continuous_state_hysteresis.{json,npz}`；
- native carried-state edge tracking：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/runaway_boundary/native_edge_tracking_0p428_to_0p429.{json,npz}`；
- 10-s boundary scans：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/zm_regime_map/runaway_plateau_boundary_{s0p42_0p44,zoom_s0p426_0p429}_m1_10s.{json,npz}`；
- diagnostic figure：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/runaway_boundary/figures/dualcore-bounded-to-tonic-runaway-boundary.{png,pdf,svg}`。

诊断图与机器结果：

- `/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/oscillatory_phase_map/`
- `figures/dualcore-oscillation-phase-convergence-audit.{png,pdf,svg}`
- `figures/dualcore-native-regime-and-basin-audit.{png,pdf,svg}`
- runner：`scripts/run_topic4_dual_core_oscillatory_phase_map.py`
- classifier：`src/topic4_dual_core_oscillation_phase.py`
- full-SNN runner：`scripts/run_topic4_dual_core_oscillatory_snn_probe.py`
- contact coverage audit：`scripts/audit_topic4_dual_core_oscillatory_snn_probe.py`；机器结果
  `/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/oscillatory_snn_probe/contact_spatial_oscillation_audit_tauGABA8_vs18.json`
- native Z×M regime runner：`scripts/run_topic4_dual_core_zm_regime_map.py`；机器结果与图
  `/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/zm_regime_map/`
- tests：`tests/test_topic4_dual_core_oscillation_phase.py`

## 8. 科学边界与下一步

当前可以写：冻结 data-driven 双核底物的确定性 fast subsystem 存在多个空间 fixed-point family 与
saddle-node fold chain；其中低支 fold 是低 fixed point 的确定性存在边界，OU-on SNN 可在该 fold 前进入持续
高活动 regime。含 realized delays 的 matched coarse assay 已在 OU-on 中位截面完成，但没有被外推为整条分支的
稳定性标签。

当前不能写：

- 患者组织存在同样的 Z 场或同样数值的临床发作阈值；
- 这是热力学意义的 phase transition；
- 这是整条分支或完整 40,000-neuron SNN 的 delay-aware 稳定性定理；
- 两个 core 在 SNN 中始终以对称 Z 演化；当前 recorder 只保存 union-core Z，不能反推 `Z_A(t)` 与 `Z_B(t)`；
- 该结果已跨 coarse resolution / topology seed 复现。

图的术语也据此收紧：当前 E 是 equilibrium branch atlas，F 是有限网格的 operational-latency response
surface，二者都不叫 phase diagram。下一版 E 必须是 native-step、长时、显式区分初值 basin 的 nonlinear
dynamical-regime map；F 必须同时给出包含 no-transition 区域的 transition probability 与保留右删失 run 的
latency（如 RMST）。有限尺寸、OU-on SNN 在正文优先称 regime map，而不是热力学相图。

本轮已经补了 frozen spatial Z + dynamic M 的 native-delay regime/basin map，并在 full-SNN runner 中加入
per-core A/B/surround Z/M recorder；2-ms 新轨迹已确认两个 core 的动态 Z 明显分开，而此前完成的 8/4/9-ms
探针在 recorder 加入前启动，不能事后补造区域 Z 轨迹。下一步最小验证是：做 3 topology 的 coarse-resolution sensitivity，把新 full-SNN
transition 在 `(Z_A,Z_B,Z_surround)` regime map 上逐时刻投影，并对高态 delay mode 做可收敛的频率/空间模态
求解；在此之前不把本图直接升格为正式 Fig.5 panel。

## 9. 产出

- 模型构建：`scripts/build_topic4_dual_core_spatial_z_meanfield.py`
- spatial-Z 数学与 continuation：`src/topic4_dual_core_spatial_z.py`
- 分岔 runner：`scripts/run_topic4_dual_core_spatial_z_bifurcation.py`
- 分支续接与图谱：`scripts/extend_topic4_dual_core_spatial_z_branch.py`、`scripts/build_topic4_dual_core_spatial_z_branch_atlas.py`
- realized-delay / OU assay：`src/topic4_dual_core_spatial_z_delay.py`、`scripts/run_topic4_dual_core_spatial_z_stability_assay.py`
- 诊断图：`scripts/plot_topic4_dual_core_spatial_z_bifurcation.py`
- 机器结果：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/bifurcation/dualcore_spatial_z_bifurcation.{json,npz}`
- 诊断图：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/bifurcation/figures/dualcore_spatial_z_bifurcation_diagnostic.{png,pdf}`
- 多分支图谱：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/bifurcation/branch_atlas/dualcore_spatial_z_branch_atlas.{json,npz}`
- delay/OU 工作截面：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/bifurcation/stability_assay/delay_ou_operating_section.{json,npz}`
- exact checkpoints：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/perturbation/checkpoints/`
- Fig.5D 状态响应：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/perturbation/dualcore_rev21_state_contrast.{json,npz}`
- Fig.5C/D 候选：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/paper_ready_panels/figures/fig5-{panel-c-core-a-bifurcation,panel-d-state-response,panels-cd-dual-core-spatial-z}.{png,pdf,svg}`
- 机制补图：同目录 `fig5-supp-spatial-z-mechanism.{png,pdf,svg}`
- 图说明与 metadata：同目录 `README.md`、`fig5-dual-core-spatial-z-cd-metadata.json`
- 测试：`tests/test_topic4_patient_zm_meanfield.py` + `tests/test_topic4_dual_core_spatial_z.py` + `tests/test_topic4_rev21_fig5_random_perturbation.py`，21/21 PASS。
