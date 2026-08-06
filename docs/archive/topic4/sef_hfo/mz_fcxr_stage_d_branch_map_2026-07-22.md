# FCXR-RC1 Stage D — 冻结磨损轴的快系统分支图（bounded-negative 收口）

日期：2026-07-22
分支：`topic4-mz-fcxr-stage-d`（工作树 `.worktrees/topic4-mz-fcxr-stage-d`），RC1 base = `c0ecb1a`
设计合同：`docs/superpowers/plans/2026-07-20-topic4-mz-fcxr-stage-d.md`
结果根：`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/fast_slow_dynamics/`
状态：**Stage D 收口（bounded-negative）。D2 模态分析本轮不做（见 §5）。下一轮转向慢反馈闭环（见 §6）。**

## 0. 朴素话摘要（测了什么 / 怎么测的 / 揭示了什么）

**测了什么。** 在已验收的更真实快系统（兴奋输入是"真电导"、会朝兴奋反转电位饱和；recurrent E→E 也是电导 + 平滑封顶，
所以再强的输入也冲不到无穷）上，把抑制"刹车"按一条固定的失效轨迹冻在不同磨损档——磨损越深（坐标 D 越大），刹车越弱。
所有慢变量都冻住不动，只让快的兴奋-抑制系统自己跑。问一件事：随着磨损加深，网络会不会在某一档**突然停进一个"能持续待着
的高活动态"**（一个跟平时零星放电本质不同的新状态、一个独立的"发作样"分支），还是同一串放电只是越来越密、永远回不到一个
新台阶。只有存在这样一个稳定高台阶，才谈得上后面加"慢反馈让它进出"。

**怎么测的。** 沿磨损坐标 D 从 0 扫到 0.15（并在 0.13/0.14/0.145 加密），每一档给三个不同起点各跑 8 秒：一个自然低起点、
两个不同强度的"踢一下"高起点。用这段网络自己平时最忙的 300ms 发放率当"平时上界"（约 9.4Hz），量"发放率高于这条上界的
时间占比"。判据：如果这是一个新的高态分支，占比会在某一档**从低突然跳到高（悬崖）**；如果只是原来那串零星放电随磨损连续
加密，占比会**平滑爬升（斜坡）**。再换第二套网络连接实现（seed3、它自己的平时上界约 9.0Hz）重跑 D=0.15，看图景重不重现。

**揭示了什么。** 三条都指向"没有独立高分支"：
1. **一直是同一种放电。** 从磨损 ~0.085 到 0.15，永远是同一种"核区先点火、旁边的轴带随后跟上、更外围几乎不动"的自终止
   放电，峰值被饱和压在 ~80–150Hz、**从不失控（不 runaway）**。第一套、第二套、所有起点都是这一种。
2. **是斜坡不是悬崖。** 高于平时上界的时间占比随磨损**平滑爬升**——0.125→0.13→0.14→0.145→0.15 依次约 0.15 / 0.15 /
   0.32 / 0.49 / 0.71，没有哪一档突然跳变。所谓"高态"标签只是这条连续斜坡爬过 50% 那一段，不是跳进一个新台阶。
3. **最密处是亚稳的、还跟种子有关。** 到磨损 0.145–0.15，放电密到占大半窗口，但会**自发随机熄灭**；而且哪个起点会熄灭
   跟用哪套网络有关（第一套是"中踢"那支熄、第二套是"不推"那支熄）。所以它不是一个稳定吸引子。

结论一句话：**饱和成功挡住了无饱和姊妹线里那个"悬崖→失控"，但没有造出一个稳定的高活动台阶。** 沿冻结磨损轴，只有一族
有界、自终止的核区放电随磨损连续加密；最密处是会自己熄灭、且跟网络实现有关的亚稳密事件串，不是独立的发作样分支。

（内部归档代号：FCXR-RC1 g_sat=21.6/dt=0.05/N=40000/L=20/epilepsiae_1146 narrow；frozen-Z z_i(D)=clip(1−D·p_i,0,1) use_z=False；
workpoint band = 300ms rolling q99 = `rate_roll_hi`；`roll_occ`/`roll_end_occ`/`roll_high_ms`；`HIGH_OCC`=0.5；6-label 集
INTERICTAL_WORKPOINT/ELEVATED_EVENT_TRAIN/METASTABLE_TRANSIENT/FINITE_HIGH_FIXED/FINITE_HIGH_ORBIT/NUMERICAL_UNSAFE；
seed1 band 9.40Hz、seed3 band 9.02Hz。）

## 1. 设置（D0 re-anchor）

- **底座（RC1，已验收，本轮不动）**：external additive FF AMPA + recurrent E→E 电导朝 E_E=58 + recurrent-only 平滑 tanh
  饱和 `g_rec_eff = g_sat·tanh(g_rec_raw/g_sat)`，`g_sat=21.6`，`dt=0.05`，`N=40000`（NE=32000），`L=20`，
  epilepsiae_1146 narrow montage。
- **冻结失效场**：`z_i(D)=clip(1 − D·p_i, 0, 1)`，`p_i` = onset-depletion 快照的 mean-1 归一化空间图案；`use_z=False`（Z 冻结、
  不动态演化），M/phi/x 全关。`assert_field_substrate_aligned` 守住场与 substrate 对齐。
- **平时上界（empirical band）**：8s slow-off baseline 的 300ms rolling-mean 发放率 q99。seed1 = 9.40Hz，seed3 = 9.02Hz。
- **判据分类器（workpoint-relative）**：只有把 300ms rolling 率**在 ≥1s 的窗口内、整窗 ≥50% 时间压在平时上界之上、且尾窗
  仍 ≥50% 在上**（占空判据、允许短暂跌落 gap-tolerant），才进 FINITE_HIGH；有 ≥1s 连续越界但不满足持续占空 → METASTABLE；
  只是偶发越界 → ELEVATED；否则 INTERICTAL。**负控通过**：D=0 / 0.05 / 0.075 三档三起点全部 INTERICTAL_WORKPOINT。

## 2. D1 分支图（seed1 全扫 + refine + seed3 复现）

**整窗高于平时上界的时间占比 `roll_occ`（T2=8s 优先，否则 T1=4s）**：

| D | 不推 low | 中踢 kick3 | 强踢 kick12 | 判读 |
|---|---|---|---|---|
| 0.0–0.075 | 0.01–0.08 | 0.00–0.06 | 0.01–0.09 | 间期工作点（负控通过） |
| 0.085 | 0.19 | 0.09 | 0.09 | 偏高事件串起点 |
| 0.09–0.125 | 0.14–0.17 | 0.07–0.11 | 0.10–0.19 | 偏高事件串（稀疏） |
| 0.13 | 0.15 | 0.11 | 0.20 | 偏高事件串 |
| 0.14 | 0.38 | 0.28 | 0.29 | 加密中 |
| 0.145 | 0.41 | 0.45 | 0.60 | 爬过阈值处（混合 elevated/metastable/finite） |
| 0.15 | **0.85** | 0.64 | **0.81** | 最密（2/3 finite、1/3 metastable） |

占比中位数沿 D = 0.15 / 0.15 / 0.32 / 0.49 / 0.71（D=0.125→0.15）——**连续斜坡，无悬崖**。

**D=0.15 seed1 T2（8s）逐支**：不推 FINITE（occ 0.85 / 末窗 0.87 / 最长连续 1358ms / 末率 11.5Hz）；中踢 METASTABLE
（occ 0.64 / 末窗 0.01 / 最长连续 1818ms / 末率 **0.0Hz**，长瞬变后熄灭）；强踢 FINITE（occ 0.81 / 末窗 0.99 / 末率 11.8Hz）。

**D=0.15 seed3 T2（8s，band 9.02Hz）逐支**：不推 METASTABLE（occ 0.82 / 末窗 0.39 / 末率 5.7Hz）；中踢 FINITE
（occ 0.86 / 末窗 1.00 / 末率 11.5Hz）；强踢 FINITE（occ 0.77 / 末窗 0.57 / 末率 10.8Hz）。→ **两套网络都 2/3 finite，但熄灭的
那支不同（seed1 中踢、seed3 不推）**：高态存在可重现，稳定性边缘、哪支熄灭随种子。

**有界性**：所有 D 的末段率被压在 ~10–12Hz（无 runaway），佐证饱和封顶生效。

## 3. D=0.145 标签敏感性 caveat（重要）

D=0.145 的精确标签（elevated vs metastable vs finite）**对 full-resolution 在线指标与降采样重分类之间的差异敏感**：在线判据
用的时间分辨率与离线从降采样 `rate_E` trace 重分类不完全一致，落在阈值附近的这一档会因此在标签之间摆动。**因此 `HIGH_OCC=0.5`
只是一个把"高态"标签打上去的操作阈值（operational threshold），不代表一个真正的动力学断点（dynamical breakpoint）。** 归档里
任何"D=0.145/0.15 是 finite-high"的措辞都必须绑定这条 caveat——它是连续占比斜坡爬过操作线的那一段，不是分岔点。

## 4. 判决（bounded-negative，最终）

> **no robust independent finite-high branch; bounded elevated event trains and seed/IC-dependent metastable dense-event
> regimes near maximal frozen depletion.**

沿冻结磨损轴、在 RC1 饱和快系统里：只有一族有界、自终止的核区事件，其密度随磨损**连续**上升；到最大磨损（D≈0.145–0.15）
密到占大半窗口、被贴上 FINITE_HIGH_ORBIT 标签，但（a）它是连续加密爬过操作阈值、不是分岔到一个独立吸引子，（b）它是亚稳的
——自发随机熄灭、哪支熄灭跟种子有关。饱和挡住了失控，但没有造出稳定高台阶。这与无饱和姊妹线（`codex/topic4-mz-slow-fast-transition`：
陡转变但**排除有限幅逃逸**、是陡崖非吸引子）一致——饱和把"悬崖→runaway"换成了"连续加密→有界亚稳密事件串"，两条线都
指向"没有稳定有限高分支"。

**允许写**：上面的 safe claim；"连续加密"；"亚稳、种子/起点依赖"；"饱和挡住 runaway"。
**禁止写**："发现一个发作样高分支"；"存在稳定高吸引子"；"D=0.15 是分岔点"；把 0.5 阈值当动力学断点；把 2/3-finite 写成
"branch confirmed"。

## 5. D2 模态分析本轮不做（用户 2026-07-22 拍板）

原计划 D2（frozen-Jacobian sech²/eigenmode 模态分析）用来形式化确认"高态与偏高事件串是同一个空间模态、没有独立的高分支
本征模"。本轮**不做**，理由：(a) 逐点轨迹已直接目视到"全程同一种核区→轴带招募模态"（`seed1/seed3_d015_traces.png` 右列），
D2 只会确认而非改变结论；(b) 干净 GO 的前提（三起点全 finite）从未满足（实测 2/3 + 连续斜坡 + 亚稳），"独立分支"这个 D2
预设的对象在数据里就不存在。D2 留作可选补充，不作为收口前置。

## 6. 最终模型与 Figure 5 合同（用户 2026-07-22 锁定）

### 6.1 最终动力学对象

最终目标是在**同一个空间 scaffold** 上得到一个可分析的完整过程：

1. 系统位于间期背景邻域，稀疏、不规则 IED 自发出现并自终止；
2. 慢变量把快系统带过 onset corridor，进入有界、可持续一段时间的发作样活动区；
3. 延迟的负反馈使活动终止；
4. 系统返回原来的间期状态空间邻域，并再次按基线统计规律产生稀疏、不规则 IED。

**不要求间期是固定点上的静默，也不要求它是固定周期的极限环。** “间期没有固定节律”完全不妨碍它对应稳定背景态、稳定吸引域
或稳定概率分布。恢复的承重定义不是波形逐点重合、周期相位复位或回到同一个微观状态，而是 post-ictal 窗重新进入 baseline
neighborhood：事件率/IEI、事件时长、参与度、峰值、空间模板与回落占空比都重新落入预注册的间期统计带，并且没有继续漂移或
再次 runaway。

因此，下一版不必强求“间期极限环 ↔ 发作极限环”的双极限环结构；**excitable interictal invariant set / stationary distribution +
slowly entered bounded ictal metastable set + delayed escape and return** 已经足以满足科学目标，但进入、发作驻留和返回都必须由动态
慢变量真实产生，不能用拼接窗口、外部持续驱动或手工 reset 伪造。

### 6.2 最终主图的四个承重 panel

参考布局：
`.worktrees/topic4-early-readout/results/paper-ready-figure/fig5_snn_state_readout/figures/fig5_candidate_E1146_snn_state_readout_with_modes.png`。
该图是布局/readout 原型，不是当前生命周期验收图；上方目前仍是 terminal runaway，没有 recovery。

1. **Virtual-SEEG lifecycle**：同一次连续仿真中清楚显示稀疏不规则间期事件 → 有界发作 → 终止 → 恢复后的稀疏不规则间期事件。
2. **Slow-variable phase portrait**：画出慢变量如何离开间期邻域、穿过 onset corridor、在发作区驻留、再越过 termination corridor
   返回间期邻域。轨迹不必闭合成严格周期轨道，但必须显示方向、时间和返回。
3. **Data bridge / energy field**：用预先锁定的同一 readout 和窗口合同，展示间期 event energy field 与 early-ictal energy field
   在空间 scaffold 上的一致，同时保留状态相关的幅值/招募范围差异。
4. **Spatial eigenmode / stimulation**：在 baseline interictal、early ictal 和 recovered interictal 三个窗口分别计算同一有效算子下的
   leading eigenmode、eigenvalue/gain/IPR 与局部刺激响应，检验“同一病理通路、不同状态易感性”，并验证恢复后 mode/readout 回到
   基线带。仅凭 core→axis 的参与度轨迹不能替代 eigenmode 分析。

### 6.3 Stage D 对这个目标已经提供什么、还缺什么

- **已提供**：数值安全且保住间期统计工作点的 RC1 快系统；有界、自终止、核区→轴带结构稳定的事件族；一个可供慢反馈捕捉/释放的
  亚稳密事件区；冻结失抑制轴上的分支/占空图。
- **尚未提供**：动态慢变量驱动的 onset；独立且可控的发作驻留；自主 termination 和 post-ictal return；真实慢变量相图轨迹；三个
  生命周期窗口的 eigenmode/stimulation 对比；基于真实 early-ictal 窗口的 energy-field bridge。

所以当前模型**不能直接产出完整目标图**，但已经把最危险的 runaway 快系统改造成可用的有界 fast substrate。下一轮的成功标准不再是
“冻结 D 轴上必须找到稳定高台阶”，而是动态慢反馈能否围绕现有亚稳密事件区构成一次进入—驻留—退出—统计返回的闭环。

## 7. 下一轮方向（用户指定，尚未开始）

**转向"利用亚稳密事件区构造非线性慢反馈闭环"，而不是继续沿冻结磨损轴扫描。** 直觉：这一族亚稳密事件串（有界、自终止、
核区结构稳定、能撑数秒再熄灭）本身可能是一个"可被慢反馈捕捉/释放"的工作物质——与其在冻结轴上找一个不存在的稳定高台阶，
不如让一个非线性慢反馈去围绕这个亚稳区构造进入/维持/退出的闭环。具体 spec 待起草。**本轮不进动态 Z/M/X。**

## 8. Provenance / 资源

- **代码**：`src/topic4_mz_fcxr_dynamics.py`（workpoint 分类器 + frozen_z_field + resolve/aggregate）；
  `scripts/run_topic4_mz_fcxr_stage_d.py`（baseline/smoke/pilot/grid/cells，seed-aware baseline）；
  `scripts/assemble_branch_map_topic4_mz_fcxr.py`（seed1-only 组图 + seed3 overlay + 最终 verdict）；
  `scripts/plot_topic4_mz_fcxr_stage_d.py`（分支图）；`scripts/plot_topic4_mz_fcxr_traces.py`（逐点轨迹，按 seed 命名输出）。
- **测试**：`tests/test_topic4_mz_fcxr_dynamics.py` 36 passed（含 D=0 间期负控、占空判据、聚合）。
- **结果**：`results/.../fast_slow_dynamics/branch_map.json`（40 cells / 7 T2 / seed3_d015 overlay / 最终 verdict）；
  `baseline_ref.json`（seed1 9.40Hz）、`baseline_ref_seed3.json`（9.02Hz）；`autodriver.log`（自动驱动全程）。
- **图**：`figures/frozen_branch_map.{png,pdf}`、`figures/seed1_d015_traces.png`、`figures/seed3_d015_traces.png`
  （+ `figures/README.md`）。作废：`figures/_SMOKE_*_DO_NOT_USE.png`。
- **引擎纪律**：6 个 guarded engine 文件未动（bless 未变）；改动只在非 guarded `mz_slow_vars.py`（z_frozen_E 注入）+ 新
  模块/脚本。全程 `--confirm-run` 门 + flock launcher + ≤2 workers OMP=1 + setsid detached，RAM 全程安全（234G avail）。
- **分支**：`topic4-mz-fcxr-stage-d`，RC1 base `c0ecb1a` 之上 20+ commits，未 push。
