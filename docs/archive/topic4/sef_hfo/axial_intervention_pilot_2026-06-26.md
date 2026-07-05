# Stage 3 事件触发轴向干预 — 工具就绪 + baseline 不合格(结构性 STOP)

- 日期：2026-06-26
- 状态：实现完成并测试通过；pilot 在 baseline eligibility 门**停住**(预声明的 3 个工作点全部不合格)
- 分支：`topic4-axial-intervention-probe`
- spec / plan：`docs/superpowers/specs/2026-06-25-stage3-deadzone-barrier-probe-design.md` / `docs/superpowers/plans/2026-06-25-stage3-deadzone-barrier-probe.md`
- 上游：Stage 3 `twoend_equal` cm-SNN(一直卡在 one-core-dominance + 伪相撞)

---

## 摘要(朴素话 / §8)

**测了什么** — 一张会"放电"的薄片组织,两端各有一个易点火的热点。我们想问:当一次大事件**已经从一端起火、正在沿轴往对面扩散**时,在中间通道上做一次"触发式压制"能不能挡住它继续往远端扩散。要回答这个问题,前提是先得有"从一端起火 → 过一会儿才传到对面"这种**有时间差的行波**——否则根本没有"抢在它传到对面之前下手"的时机。

**怎么测的** — 先把仿真工具按测试驱动一块块搭好(几何/源标注/动态阈值钳制/重放调度/六臂 runner/汇总脚本,共 23 个单测全过),其中最关键的一关是:把"加干预"的仿真做成和原引擎**逐位一致**(不加干预时一模一样,加了之后只在动手那一刻之后才不同)——这一关过了。然后跑"无干预 baseline",用一个**事先冻死的合格门**判断这个工作点有没有"可挡的行波":要求 ≥20 个返回事件、两端各 ≥3、≥5 个真正越过中线、并且其中 ≥5 个是"先起火、后传到对面(有时间窗)"。在 3 个**事先约定好**的热设置上各跑一遍(T=3000)。

**揭示了什么** — 3 个设置**全部不合格**,而且失败方式高度一致:① 返回事件太少(9~11 个,不够 20);② 一端独大(比如一端 0 次、另一端 6 次单源事件);③ 近一半是"两端同时点火"的相撞;④ 最关键——**"先起火后传到对面"的时间窗 = 0 个**。逐事件看:凡是能越过中线的事件,对面第一个放电的时刻和起火端起火时刻**几乎同一瞬间(时间差≈0)**,而且一下子就铺满整片(reach≈20mm),整个事件只持续 7~22ms。而一个真正以轴突传导速度(0.3mm/ms)走 20mm 的行波得花 ~67ms——比事件总时长还长好几倍。所以**这个热工作点下的大事件不是"从一端扫到另一端的行波",而是整片几乎同时点亮的同步爆发**;小事件则是出不了门的局部点火。**既然没有"传播时间差",就没有"抢在传到对面之前下手"的时机,这个轴向干预问题在当前底物上不可测**。冻结的合格门正确地把这一点挡了下来。按事先锁好的协议,**到此停手,不再调参,交给用户复核**。

(内部归档代号:twoend_equal、`core_source_raw`、`oracle_far_ratio`/`oracle_reach_mm`/`far_onset_time`、`n_trigger_opportunity`、`simulate_dynamic_vth` parity、baseline_eligibility gate)

---

## 1. 实现产出(已完成,23 测试全绿,已提交)

| 模块 | 内容 | 测试 |
|---|---|---|
| `src/sef_hfo_axial_intervention.py` | 几何(`band_mask`/`split_near_target_far`)、源标注(`core_source_raw`,去 readout 可读性门)、参与率(分母排除被钳制 cell)、目标掩码 + 动态钳制(`intervention_vth_at_time`/`make_on_axis_target`/`make_off_axis_target`)、合格门 + 重放调度(`baseline_eligibility`/`select_first_eligible_event`/`build_replay_schedule`/`build_late_schedule`)、**动态阈值仿真适配器 `simulate_dynamic_vth`** | 16 |
| `scripts/run_stage3_axial_intervention_probe.py` | 六臂 runner(baseline/static_deadzone/dynamic_on_axis/dynamic_off_axis/late_on_axis/wall_only),`--schedule-json`/`--baseline-json`,importlib 复用 canonical 构件,逐事件 oracle+instrument 指标 | 4 |
| `scripts/summarize_stage3_axial_intervention_pilot.py` | 按 (arm,seed) 汇总,JSON/CSV,保留 fail-guard 字段 + 排除目标电极的 instrument 指标 | 3 |

**关键不变量验证(§6 合同)**:`simulate_dynamic_vth` 无 schedule 时与 `simulate_kick(KICK_BOOST=0,t_kick=1e9,V_th_per_neuron=base_vth)` 的 `E_spk_bool`/`rate_E` **逐位相同**;有 schedule 时干预开始前逐位相同。原因:干预只改变 spike 阈值比较,不增删任何 RNG 抽样。**工具本身正确可用**——本次 STOP 是底物问题,不是工具问题。

## 2. baseline eligibility(预声明 3 工作点,seed 1,T=3000,各 ~33min)

| 工作点 | eligible | reason | n_returned | neg/pos | collision | cross-midline | trigger-opp |
|---|---|---|---|---|---|---|---|
| m17.0 / sep0.6 / std1.5 | ❌ | too_few_events | 11 | 0 / 6 | 5 | 3 | **0** |
| m16.5 / sep0.6 / std1.5 | ❌ | too_few_events | 9 | 1 / 4 | 4 | 5 | **0** |
| m16.5 / sep0.5 / std1.5 | ❌ | too_few_events | 2 | 0 / 1 | 1 | 1 | **0** |

门第一关栽在 `too_few_events`(n<20);但**真正的结构性障碍是 `trigger-opportunity=0`**——即使跑更长的 T 凑够 20 个事件,也不会凭空长出"有时间差的行波"。

## 3. 逐事件诊断(为什么 opp=0)

单源(neg/pos)事件逐条看(节选 m17.0/sep0.6):

| src | t_on | dur(ms) | src_on | far_on | gap | far_ratio | reach(mm) | n_part |
|---|---|---|---|---|---|---|---|---|
| pos | 750 | 15 | 750.0 | 750.0 | **0.0** | 0.043 | 20.03 | 7 |
| pos | 986 | 18 | 986.0 | 986.0 | **0.0** | 0.162 | 20.03 | 7 |
| pos | 1466 | 9 | 1466.0 | 1474.8 | 8.8 | 0.000 | 10.47 | 3 |
| pos | 2356 | 18 | 2356.0 | 2356.0 | **0.0** | 0.218 | 20.03 | 8 |
| pos | 2812 | 18 | 2812.0 | 2812.0 | **0.0** | 0.203 | 20.03 | 8 |

事件分三类,**没有第四类(有延迟的行波)**:
- **同步整片爆发**:`gap=0`、`reach≈20mm`(满场)、`far_ratio` 0.04–0.22、`n_part` 7–8、时长仅 7–22ms。对面第一放电与起火端起火同一瞬间。
- **局部出不了门**:`far_ratio=0`、`reach` 8–10mm、`n_part` 2–3。
- **相撞**:两核 30ms 内同点(占 ~45–50%)。
m16.5/sep0.6 的 5 个单源事件**全部**是同步整片型(gap=0,reach≈20)。

**行波判据(决定性,且不依赖 far_onset 的灵敏度)**:reach 20mm / 事件时长 ≤22ms ⇒ 等效速度 ≥0.9mm/ms,是轴突传导速度 v_axon=0.3mm/ms 的 ~3 倍。真正受传导延迟限制的行波走 20mm 需 ~67ms ≫ 事件时长。所以这是**快速多突触递归招募的近同步点亮**,不是以轴突速度扫过的前沿。无论 far_onset 怎么定义,时间窗都装不进一个 ≤22ms 的同步事件里。

## 4. 这意味着什么 / 不意味着什么

- **意味着**:在预声明的热工作点上,`twoend_equal` 底物产生的是同步整片爆发(或局部点火 / 相撞),不是"从源端扫向汇端、带招募延迟"的焦点行波。事件触发轴向干预的前提(有传播延迟可抢)在此**不成立**,问题不可测。这与 Stage 3 早期 pilot 的 one-core-dominance + 伪相撞一致。
- **不意味着**:不代表轴向干预策略本身错;只代表**当前底物/工作点产生不出可挡的行波**。也不代表工具有问题(parity 等全过)。
- **caveat(已诚实记录)**:门第一关是 `too_few_events`;`opp=0` 的 far_onset 用的是"对面第一个放电"(对单个噪声 spike 很敏感)。但 §3 的行波判据(reach/时长)不依赖该灵敏度,结构性结论稳。

## 5. 交给用户复核的选项(本轮未执行,需用户决定)

1. **换出真正会行波的底物/工作点**:更冷 + 更强各向异性、或更大 sep、或调 E→E 让前沿以轴突速度可见地扫过(而非整片同步)——这属于"调参找合格 regime",spec 明令需用户复核后才做。
2. **重定义问题**:若底物天然是同步爆发而非行波,"轴向行波干预"可能不是这个模型该问的问题;可考虑改问"同步爆发能否被中线钳制打散/降幅"(但那不是行波阻断,需重写 spec)。
3. **先验证行波缺失**:加一个 per-neuron onset-vs-轴向位置 的诊断图(确认是平的=同步,不是斜的=行波)再定方向——一次额外仿真(~30min/run)。

## 6. 规程合规

- 走完预声明 fallback 阶梯(m17.0/sep0.6 → m16.5/sep0.6 → m16.5/sep0.5),全部不合格 → 按 spec §6 **STOP,不再调参**。
- 未跑四/六臂矩阵(baseline 不合格,不进入干预对照);未做正式长跑;未出"图当证据"。
- 结果 JSON 在 `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/axial_intervention_probe/`(gitignored);本 doc 内联保留了判定所需的全部数值。
