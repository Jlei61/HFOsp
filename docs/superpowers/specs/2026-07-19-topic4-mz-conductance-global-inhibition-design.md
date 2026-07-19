# Topic 4 MZ conductance + global inhibition：L=20 cheap-first design

日期：2026-07-19
状态：执行中；结果中性（positive / negative / boundary 都可接受）

## 1. 当前阻断与目标

当前 Z-only 在 L=20 SNN 上稳定产生 interictal-event train 后进入 runaway；线性 subtractive M 只能把它变成 prevention 或不恢复的 elevated plateau。新节点只回答两个顺序固定的问题：

1. **provoked lifecycle**：同一工作点是否存在刺激撤除后仍维持、但最终自终止的 bounded ictal state？
2. **spontaneous access**：若该 state 存在，当前自然 Z staircase 能否在无 kick 时进入并退出？

基础 Abbott/Liou 模型本身需要 transient focal input；因此 provoked 通过不等于 spontaneous 已通过。

## 2. 锁定 substrate

- `epilepsiae_1146 / narrow / template_source / twoend_equal`
- `L=20 mm`, `density=100/mm²`, `N=40,000`, `AR=2`
- `g=3.6`, `drive=0.6`, `dt=0.1 ms`
- primary seeds `1/3`；seed `4` 只作 stress
- 保留当前双低阈值核与连接矩阵；首屏不重调 scaffold

## 3. 新膜方程

E cells：

\[
\tau_m\dot V_i=-V_i+I_i^E+g_i^I(E_{GABA}-V_i)+g_i^M(E_K-V_i).
\]

I cells 保持原 current path。`I_I` 与 `eta_m*m` 是 mV current proxy，不是电导；用 reference-force matching 转成相对漏电导：

\[
g_i^I=\frac{a_I I_{i,eff}^I}{V_{match}-E_{GABA}},\qquad
g_i^M=\frac{a_M\eta_m m_i}{V_{match}-E_K}.
\]

current-equivalent primary 取 `V_match=18`, `E_GABA=V_L=0`, `E_K=0`：在阈值处与旧
subtractive `I_I` 做 force matching，同时在 `V_reset=11` 附近仍保留抑制力。仓库旧 M2 的
`E_GABA=V_reset=11` 是 pure-shunt sensitivity，不能默认等同于 current replacement。exact update：

\[
V_\infty=\frac{I_E+g_IE_{GABA}+g_ME_K}{1+g_I+g_M},\quad
V(t+dt)=V_\infty+[V(t)-V_\infty]\exp[-dt(1+g_I+g_M)/\tau_m].
\]

## 4. global GABA 首屏

replacement sensitivity：

\[
I_{eff,i}^I=(1-\gamma)I_i^I+\gamma\langle I^I\rangle_E.
\]

additive global-restraint primary：

\[
I_{eff,i}^I=I_i^I+\beta\langle I^I\rangle_E.
\]

两路都明确记为 received-GABA rank-1 surrogate，不是严格 presynaptic uniform kernel，也不是新的秒级 M4 pool。`I_I` 已含 GABA rise/decay kinetics。replacement 用来检验空间重分配；additive + `z_scope=local_only` 才对应“保留 local restraint，并在 conductance denominator 上增加不随 local Z 一起耗竭的 global restraint”。

## 5. 执行阶梯

1. current slow-off，seed1/T=8s：复现 returning interictal train。
2. conductance slow-off，`gamma=0, E_GABA=0`，沿 `gaba_gain={0.5,0.75,1,1.25}` 找最多两个工作点；
   `E_GABA=11` 只作 pure-shunt sensitivity。
3. seeds1/3 确认工作点；seed4 stress。
4. 在工作点上重标 Z threshold；先复现 q75/tau_z=5s 型稳定 staircase/runaway anchor。
5. provoked z-only/gamma=0 runaway reference。
6. 逐轴扫 `gamma={1/12,1/6,1/3}`；只有出现 runaway↔prevention bracket 才做至多两次 midpoint。
7. 最佳 gamma 上加 sAHP conductance；仍为 tonic plateau 才加 `phi` arm。
8. provoked lifecycle 通过后，完全同参去 kick 检验 spontaneous access。

## 6. 科学 gate

### Stable staircase

- pre-transition 至少 5 个 frozen-bar returning events；
- event-locked `median(delta_D)>0`，正增量比例 ≥0.7；
- 正向 D 增量至少 0.6 落在 event/post-event window；
- event index 与 `D_pre` Spearman ≥0.7；
- primary seeds 同方向。

### Bounded ictal oscillation

- kick 撤除后高招募 screen ≥1s、confirm ≥5s；
- 不命中 `120 Hz / 100 ms` runaway；
- ≥4 bursts、≥3 IBI、rate modulation ≥0.3，并有 0.5–20Hz 非零谱峰；
- 不能只是平坦 elevated plateau。

### Recovery

- 最后 burst 后连续 2s 回到同 seed slow-off rate/AF band；
- 不是永久沉默：late probe 应恢复 baseline-like response；
- Z 朝 pre-trigger 恢复，M/phi 从 ictal peak 回落；
- confirm T≥`max(20s,4*tau_M)`；40s 只给跨 seed 候选。

## 7. 资源与停止规则

- 单 launcher 构网一次，fork/COW 共享；禁止每个 agent 独立构 L=20 network。
- `OMP/OPENBLAS/MKL=1`；无 nested pool。
- workers 硬 cap 4；T≥20s 且保留 full `E_spk_bool` 时 workers≤2。
- 本机 100ms 实测：parent/build+run peak RSS 约 6.79GiB；T=8s 每 worker 另有约 2.38GiB 的 E raster 下界。
- conductance 全程非负有限；`tau_eff<2dt`、NaN/Inf、任何 conductance clipping 立即停，不能把截断后的 cell 当科学结果。
- 工作点 9 cells 内不复现 baseline，先修单位/映射；不靠继续加 M/phi 救工作点。
