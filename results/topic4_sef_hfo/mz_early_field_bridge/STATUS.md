# MZ early-field bridge — STATUS

> 当前版本：**V1 observation/readout bridge，冻结于 2026-07-19**
>
> 分支：`codex/topic4-mz-slowvars`，local-only
>
> 设计合同：`docs/superpowers/specs/2026-07-19-topic4-mz-early-field-bridge-design.md`

## 一句话结论

在一块固定的 E1146 SNN 底物上，slow-off 间期样事件给出的**双向时序轴**能够预测 `z`-only
轨迹跨过 operational-runaway 阈值前的早期 contact 能量分布；这支持的是
**“同一支架、状态依赖读出”的观测层可行性**，不是发作复现、因果机制或患者队列结论。

## 当前证据

- 分母：一块 E1146 模型底物 × seeds 1/3/4，不是 3 个患者。
- 主统计：held-out-validated slow-off 双向模板与 `t_recruit` 后 0–50 ms、`t120` 前 contact
  energy 的 mirror-invariant `rho_maxAB`。
- `rho_maxAB`：0.945 / 0.735 / 0.924，中位 0.924，3/3 为正。
- within-shaft null：p=0.0004 / 0.086 / 0.001；2/3 明确，seed3 与 null 重叠。
- source-grid 只作方向无关的轴调用补充诊断：0.651 / 0.546 / 0.585，toroidal-shift
  p=0.0087 / 0.012 / 0.017。它不与 contact 合并成“跨尺度同方向”结论。
- 哪一端先起火不是结论。A→B、B→A 都是注册模板，`maxAB` 只处理镜像不变性，不定位固定发作灶。

## 跨 seed 迁移的正确口径

跨 seed 诊断把 seed i 的 slow-off 模板用于 seed j 的早期能量场。3×3 maxAB 主要随目标能量
seed 改变；同一目标下模板 seed 的平均散度约 0.007，而目标场均值之间约 0.095。更直接地，
每个目标 seed 的“同 seed 模板 − 两个外来模板中位数”为约 +0.002 / −0.002 / −0.010，
没有描述性的 same-seed 优势。这削弱了“结果只来自同一噪声 replay”的解释。

但这项扩展仍是 **exploratory diagnostic**：

- 统计重复单位只有 3 个目标能量场；9 个格子不是 9 个独立样本。
- 这次 9/9 格子的 maxAB 都由 B→A 分支取胜，因此只能说**被调用的预测分支可跨 seed 迁移**；
  不能据此宣称 A/B 两个方向都已证明为 seed-invariant scaffold 属性。
- 原图中的 field cosine 本质仍是相关量，quartile contrast 又是在 Spearman 胜出方向上事后读取；
  两者不再写作“独立非相关验证”。正式独立替代指标需要预先固定方向或在 null 内重做选择。

## Figure 5 V1

主候选图：`results/paper-ready-figure/fig_mz_early_bridge/figures/fig_mz_early_bridge.{png,pdf}`。

- 上排是一条连续的 seed1 z-only native virtual-SEEG 轨迹，不拼接两次 replay。
- 蓝窗是按固定规则选出的一个 native returning event；粉窗是 `t_recruit` 后 0–50 ms
  pre-t120 early-energy window；红虚线是 operational-runaway `t120`。
- 下排只有两个与上方窗一一对应的 contact-readout 场：exact-event recruitment order 与
  pre-t120 energy。灰点只表示固定 E-neuron 几何，不表示局部招募。
- 案例 exact-event 相关是描述性展示；正式统计仍来自 slow-off held-out 双向模板。

跨 seed 图 `fig_mz_cross_seed_transfer.{png,pdf}` 仅作补充诊断，不作为主图证据或 n=9 推断。

## 可以写 / 不可以写

可以写：

> On a fixed patient-layout SNN scaffold, a held-out bidirectional interictal-like timing axis predicted
> the spatial distribution of pre-runaway virtual-contact energy across three noise seeds, supporting an
> observation-level same-scaffold, state-dependent-readout bridge.

不可以写：clinical seizure、clinical broadband power、complete seizure cycle、`z_i` 是唯一生物机制、
某一端是固定发作灶、间期事件因果触发失控、contact 热点等于局部神经元优先招募、结果不依赖 core。

## 工程验收

- fixed slow-off event bar 被跨状态复用；contact energy 窗口用整数步对齐，避免浮点边界误判。
- odd/even held-out 模板、maxAB 内置换重选、within-shaft / toroidal null、fail-closed eligibility
  均有合成合同测试。
- 原始仿真、配置快照、per-seed JSON/NPZ 和 provenance 已保留；大 LFP/raster 中间文件不进 git。
- 结果图都有中文 README；Figure 5 PNG/PDF 已目检。
- cross-seed 汇总现在显式记录 3 个目标重复单位、胜出方向计数和 matched same-vs-foreign 差值；
  CSV 使用 LF 行尾，并新增汇总合同测试。

## 尚未完成

1. contact 层只有 2/3 seed 明确超过 within-shaft null，强度仍不稳。
2. CRN replay 不是 checkpoint 后的真实状态分叉，不能区分全局去抑制增益与局部 `z_i` 图案。
3. early-window 神经元 raster 未持久化，contact hotspot 的局部组织参与度不能审计。
4. core-exclusion 没有删掉任何 contact，不能回答是否依赖 core loading。
5. 跨 seed 只验证了本次被调用的 B→A 分支；双向分支的独立迁移还未建立。

## V2 触发条件（等待 MZ onset-dynamics 版本）

本 V1 不被后续结果覆盖。只有并行的 MZ onset-dynamics / state-conditioned artifacts 经独立验收后，
才生成 **V2 integrated bridge**；不要提前把在跑的结果写入本结论。V2 至少要做到：

1. 用完全相同的 `t_recruit`、`t120` 和 contact 模板注册到 `D_z`、`q_eff`、`A_m` 状态；
2. 给出 state-matched checkpoint/resume 对照（native / uniform-mean / shuffled / reset z，必要时 m freeze/reset）；
3. 区分整体去抑制、空间化易感性和 nonlinear ignition threshold 的贡献；
4. 若持久化了早窗 raster，补局部组织参与度；
5. V2 单独出动力学图，保留本 Figure 5 V1 作为 observation-layer exemplar，不把相图塞回这张图。

如果这些条件未满足，下一版仍只能叫 readout update，不能升级为 mechanism/causal bridge。
