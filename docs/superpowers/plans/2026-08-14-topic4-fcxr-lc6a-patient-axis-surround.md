# FCXR-LC6A patient-axis surround — implementation plan

日期：2026-08-14

版本：rev2

状态：**IMPLEMENTATION READY — EXECUTION REQUIRES EXPLICIT AUTHORIZATION**

对应 spec：
`docs/superpowers/specs/2026-08-14-topic4-fcxr-lc6a-patient-axis-surround-design.md`

## 1. Definition of done

本轮完成定义压缩为六项：

1. LC5v2.1 唯一右删失格获得终局裁决；
2. C0/C1/Q1/Q2/Q3 的 direction、degree、weight、delay、marginal kernel、two-hop kernel 和 hash 可审计；
3. graph family 的短 impulse characterization 完成，不以 zero crossing 决定自然轨迹权限；
4. 五条固定自然轨迹完成，统一报告 entry、D halo、global/local saturation、active area 和 late drift；
5. 对最多两个有信息量 high-state phenotype 完成 gain/state fork；
6. 若出现 carrier，完成至少一个不同 graph realization 的条件性确认，再决定是否进入 LC6B。

最终输出的是结果向量，不要求前八层全 PASS：

```text
engineering valid?
graph family instantiated?
two-hop surround changed?
baseline altered?
natural entry altered?
bounded high branch opened?
local saturation avoided?
response gain retained?
robust to one additional graph realization?
termination tested?   NO
lifecycle tested?     NO
```

## 2. T0：preflight、manifest 与并行权限

### T0.1 artifact/state preflight

只读解析并锁定：

- LC5 `tau15/gamma003` 25 s summary/traces/spikes/final_state；
- E1146 registration、positions、Vth、source/sink 和 patient axis；
- C0 `EE/IE/EI/II` graph、weights、delays；
- Z/H config、classifier、current branch source；
- 六个 blessed engine hash；
- external-input counter/hash contract。

所有路径从 manifest/上游 JSON 解析；永久产物不得写死 worktree 绝对路径。缺 artifact/hash 响亮失败。

### T0.2 sole machine manifest

新增并测试：

```text
config/topic4_fcxr_lc6a_patient_axis_surround.json
```

包含五图、q tolerance、graph seeds/convergence、weight/delay rule、trajectory horizons、global/local
classifier、saturation/readout、resource limits 与 hashes。runner 只读 manifest，不暴露旧 LC5 stage menu。

### T0.3 parallel boundary

LC5 continuation 与 T2/T3 graph coding、tests、graph-only 构建可并行。只有 LC5 得到 stable carrier/offset
时暂停 T4 之后的全部 40k functional/natural dynamics；已经完成的 graph artifacts 保留。

## 3. T1：LC5 right-censor exact continuation

把已有 candidate-extension 逻辑抽成 manifest-driven 单臂 runner：

```text
source: tau15_gamma003 at 25 s
target: total 40 s
early terminal: 1 s saturation OR offset + 2 s low observation
```

加载后核对 step/state/RNG/future-input hash。轻量 summary 每 1 s，exact checkpoint 每 5 s，并 pin
terminal state。输出独立目录，不覆盖原 screen。

结果：

- saturation/positive drift：closeout addendum，允许 T4+；
- carrier/offset：写 `LC6A_TRAJECTORIES_PAUSED_BY_LC5_POSITIVE.json`；
- instrument failure：只修同一 runner，不解释科学结果。

## 4. T2：graph module 与 tests

建议新增：

```text
src/topic4_fcxr_lc6_surround.py
tests/test_topic4_fcxr_lc6_surround.py
```

纯函数：

```text
extract_target_first_edge_tables(...)
project_displacements(...)
empirical_marginal_widths(...)
rewire_e_to_i_targetwise(...)
assign_frozen_target_weights(...)
recompute_physical_delays(...)
coarse_two_hop_kernel(...)
audit_graph_legality(...)
```

先写失败测试：

1. `IE` source=E,target=I；`EI` source=I,target=E；
2. C0 adjacency/weight/delay exact hash；
3. 每个 I target E in-degree exact；
4. 无 duplicate；
5. per-target incoming-weight multiset exact，且 weight 不按新距离变化；
6. delay 只按冻结 `tau0+d/v` 与 quantization 重算；
7. graph RNG/runtime RNG 隔离；
8. 同 seed graph bitwise reproducible；
9. asymmetric proposal 必须应用 Hastings correction；
10. q target 不可达时 fail closed；
11. off/default path 与 C0 graph parity。

不编辑 blessed engine 文件。

## 5. T3：构建固定 graph family 与 two-hop audit

### T3.1 五图

构建并冻结：

```text
C0  legacy
C1  legacy-q inhibitory microstate
Q1  q=1.00±0.05
Q2  q=1.25±0.05
Q3  q=1.50±0.05
```

不再有 q=1.375 midpoint。所有条件从同一 C0 graph、同一 graph-RNG family 出发，但 condition-specific
convergence、common maximum budget；不得强制相同 accepted-swap 数。

收敛审计至少包括：q/energy trajectory、acceptance、edge-Hamming plateau、分段链一致性、interior/edge
kernel stability。graph-only 完成前不读取任何 SNN outcome。

### T3.2 degree/weight/delay

- target in-degree 和 per-target weight multiset exact；
- source out-degree mean/CV/q95/q99、spatial gradient、interior/edge distribution 按 spec 容差审计；
- delay 按新距离重算并报告 median/q95；
- C1 dynamics 不参与 graph 是否合法的判断。

### T3.3 weighted two-hop kernel

对 interior E sources 分层抽样或用 coarse sparse multiplication 估计 `EI @ IE`。输出 center/tail width、
inhibitory-magnitude local/surround mass、ratio、latency、symmetry、boundary effect 与 shared-interneuron
motifs；带符号 operator 单独保存。

产物：

```text
graphs/C0.npz
graphs/C1.npz
graphs/Q1.npz
graphs/Q2.npz
graphs/Q3.npz
graph_audit.json
two_hop_kernel_audit.json
figures/lc6a_graph_and_twohop.png
```

只有 graph legality 失败才修 builder；不得根据后续动力学重抽“好看 graph”。

## 6. T4：short functional characterization

### T4.1 amplitude lock

在 C0 上预锁唯一 weak E-patch input：亚阈值、无 population event。patch geometry 与 amplitude 在任何
Q outcome 揭盲前写入 manifest addendum。

### T4.2 probe matrix

- C0/Q2/Q3：core-adjacent + neutral-axis 两位置；
- C1/Q1：至少 neutral-axis；
- paired sham/probe external input exact shared；
- baseline `Z=1,H=0,U=M=0,X=1`；
- 记录 0--50、50--150、150--300 ms，并按 two-hop delay q95 做补充窗。

输出 actual signed membrane contribution、E/I force magnitudes、conductance、rate、zero crossing、latency 和
forward/backward asymmetry。

这是 descriptive assay：无 zero crossing、C1 改变或响应很弱均写入结果，不阻断 T5。

不在 LC6A 做正式 operator/eigenmode reconstruction。

## 7. T5：五臂固定自然轨迹

LC5 continuation 为阴性且五图 legality 通过后，全部运行：

```text
C0 / C1 / Q1 / Q2 / Q3
```

### T5.1 common runtime

- fresh t0；Z/H dynamic；`U=M=0,X=1`；
- no kick/reset/step；
- shared counter-based/pre-generated external input；
- base 50 s；onset 后 12 s；hard cap 65 s；
- 50 s 无 onset且 IED exposure 不足则继续到 65 s；
- saturation 连续 1 s 可终止该 arm，但不得停止其他 arm。

### T5.2 output/checkpoint cadence

每 1 s 写轻量 summary；完整 exact checkpoint 每 5 s。额外 pin：

```text
onset
onset+1 s
onset+2 s
onset+4 s
onset+8 s
onset+12 s
pre-offset/offset（若出现）
```

rolling recovery checkpoint 只保留最近两个，关键 checkpoint 单独归档。每臂使用 atomic bundle 与
RUNNING/DONE/FAILED sentinel。

### T5.3 C0 control

C0 graph/input/state exact。相同 engine path 要求 bitwise trajectory parity；若 instrumentation 改变 reduction
ordering，执行 event/rate/state numerical parity 并记录原因。C0 不复现属于 instrument integrity 问题。

### T5.4 local classifier prelock

从 C0 pre-onset returning IED 用冻结 coarse grid 计算：

- 100 ms local-rate q99.5；
- IED maximum connected-area q99；
- persistence 500 ms。

在聚合 Q1--Q3 outcome 前写入 manifest addendum。之后同时使用 global LC5 classifier 和 local companion。

## 8. T6：统一 phenotype map

所有五臂完成后一次性聚合，不逐臂决定是否继续。每臂输出连续变量：

- IED count/rate/IEI/duration/participation；
- global/local onset latency 与 occupancy；
- D increment、halo width、halo lead；
- global mean 与 local q95/q99 rate；
- whole-episode `max f_ref` 与 `fraction[f_ref>5%]`；
- active area、first passage、recruitment speed；
- late rate/H/D slope；
- E/I current decomposition；
- spatial persistence、centroid motion、envelope CV。

headline labels 由纯函数产生，但互相独立的字段不被压成一个 PASS/FAIL：

```text
entry status
boundedness
local saturation
baseline tradeoff
spatial phenotype
D-halo acceleration
responsiveness        # T7 后补
```

未 onset只有达到 `ceil(1.5*N_IED_C0_to_onset)` 才标 entry blocked；否则是 delayed/unresolved exposure。

## 9. T7：两个 phenotype 的 gain/state forks

预注册选择：

1. boundedness margin 最大；
2. 最接近 saturation/block boundary，或 spatial phenotype 与第一个最不同。

最多两个，若只有一个有信息量则只选一个。每个在 onset+2 s、onset+6 s 做 paired sham/50 ms probe：

- external input exact；
- 2--3 个 exact duplicate checkpoints 先验证 replay determinism；
- duplicate 非零则修 instrument；
- 报告 susceptibility、relaxation、rate/area deviation、saturation/offset/divergence；
- responsiveness 单独分类，stationary 但 responsive 可保留为 carrier。

不使用 20-checkpoint numerical-null q99，不把单一 gain threshold嵌入 boundedness。

## 10. T8：条件性 positive confirmation

若 canonical graph/noise 出现 bounded responsive carrier：

1. q、weight/delay rule、cell/slow parameters 全冻结；
2. 用不同 graph seed 构建同 q 的 realization B；
3. noise seed 401 运行同一合同；
4. 资源允许时再做 graph A/noise 402；
5. 不根据确认结果回调参数。

graph B/401 复现后才给 `CLEAN_CARRIER_POSITIVE`。否则保留单图 provisional 或记录 graph sensitivity。

## 11. T9：聚合、图和 archive

统一生成：

1. marginal/two-hop graph family；
2. functional impulse response；
3. 五臂 onset、D halo、global/local rate、area、drift phenotype map；
4. 两个 fork（条件性）；
5. graph confirmation（条件性）；
6. `figures/README.md` 中文逐图说明；
7. STATUS、manifest、resource log 与 archive。

archive 逐层回答 engineering、graph、two-hop、baseline、entry、boundedness、local saturation、gain、
confirmation；termination/lifecycle 固定 `NOT_TESTED`。

## 12. 只有三类硬停机

### H1 artifact/state integrity

state/hash/schema、RNG/input counter 或 continuation 完整性失败。

### H2 graph legality

population direction、target degree、duplicate、weight/delay assignment、q target、graph hash 或 RNG isolation
失败。

### H3 numerical/resource integrity

NaN/Inf、不可恢复 checkpoint、OOM 后状态不完整、注册 numerical stability 失败。

C1 baseline 偏移、无 zero crossing、entry blocked、static pattern、low gain、D halo 加宽或 onset 提前均不是
停止剩余 primary arms 的 token。

## 13. 资源与 detached execution

- 每个 40k arm 内部 1 worker，BLAS/OpenMP thread=1；
- C1 首臂重测峰值 RSS，按 `1.5×RSS/arm`、MemAvailable 和 sibling 占用动态填槽，上限 4；
- `MemAvailable >=3×` 新增总预算才提交；
- swap 相对 stage baseline `+256 MiB` 停新提交，`+512 MiB` 最新 worker 保存 checkpoint 后退出；
- 所有长任务 `setsid nohup`、stage/arm flock、PID、RUNNING/DONE/FAILED sentinel；
- waiter 按 PID/sentinel，不用自匹配的 `pgrep -f`；
- 不碰 sibling worktree，不 push/merge/rebase，除非用户另行授权。

## 14. 执行量与授权边界

主科学块固定为五条 40k 自然轨迹；graph/short probe 在前，fork/confirmation 条件性。LC5 positive 可以
暂停昂贵轨迹，但不撤销 graph engineering。

本 plan 不授权：LC6B/U、M、global tail、center-preserving follow-up、H 修改、heterogeneity、全五臂
multi-seed、正式 operator/eigenmode audit 或 paper-ready 主图。当前只把 spec/plan 修到 implementation
ready；未得到用户明确授权前不运行仿真。
