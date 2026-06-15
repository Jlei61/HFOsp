# Axis A — 病灶内部机制 → 传播指纹：执行归档

**Date:** 2026-06-15 · **Status:** 本轮(Level 2 = 阈值离散臂 + E/I 臂)**已完成,screen 级收口、待验收**(用户 review 2026-06-15:A3 不升 verdict) · **Plan:** `docs/superpowers/plans/2026-06-14-sef-hfo-axisA-lesion-propagation-fingerprint.md` · **Spec(engine):** `docs/superpowers/specs/2026-06-13-sef-hfo-snn-axisA-ei-local-lesion-design.md`

用户 2026-06-15 锁定:scope = 阈值离散臂 + E/I 机制臂(改引擎);A1 = fixed-mean 主 + matched-rate 敏感性;引擎 re-bless 自主进行;阶段提交 + 阻塞降级继续。

---

## §0 朴素 abstract(测了什么 / 怎么测的 / 揭示了什么)

**测了什么** —— 在同一张网络上,用两种不同的"局部病灶"方式让一小块组织变得容易自发放电:(1) 把那一小块的**放电门槛离散度**调宽/调窄(阈值异质性);(2) 把那一小块的**抑制刹车调弱**(w_EI↓)或**自激回路加粗**(w_EE↑)。问:不同的病灶方式,会不会让间期 HFO 在网络上传播时留下**不同的"指纹"**(往哪传 axis_dir、波垂直方向多宽 pathway_width、入口漂不漂 onset_jitter)。

**怎么测的** —— 每个病灶配置自发放出一串间期事件,透过同一套虚拟 SEEG 电极读出,用冻结的指纹仪器(`src/sef_hfo_fingerprint.py`)抽 primary 三件套,跨 seed 聚合后做非参检验 + 效应量。所有比较都在**事件率匹配的工作点**上,防"更容易点火"冒充"机制指纹不同"。

**揭示了什么** ——
- **阈值离散臂(A1/A4):描述性 NULL。** 在干净工作点(mean=17.0)上,把核内放电门槛的离散度从窄(0.5)调到宽(1.5),**只改变事件点火的频率**(clean 事件 窄26 / 中40 / 宽47),**不改变传播指纹**——三档的 pathway_width 全是 2.6mm、axis_err 全是 0°、最早入口全锁在 A0。也就是说"离散度大小"在这个电极分辨率下不留下可区分的传播指纹,它只决定"多久点一次火"。
- **E/I 机制臂(A3):描述性 NULL(screen 级)。** 把核内抑制刹车调弱(w_EI↓)或自激加粗(w_EE↑),核确实自发点火且 nucleation 率能匹配 V_th↓,但放出的事件**自终止却空间局部**(参与触点 <7、读不出方向);**加大磁量不让事件长大,只把 bare-sheet 背景抬高 ~130×(破坏"只从核点火"不变量)**。即**没有磁量窗口能既保持 bare sheet 安静、又复现 V_th↓ 的 clean 方向模板**。受 A1 约束的机制解读:传播靠的是**降阈值**(把核外细胞也降到易招募),局部 E/I 只动核内、核外满阈值 → 波停在核内。

(一句话:"病灶内部机制不同 → 传播指纹不同"这个 A 轴主假设,**两条都是 screen-NULL**——阈值离散度只改点火频率不改指纹;局部 E/I 病灶在测试磁量内不复现 V_th↓ 的可读方向模板。read-out 的方向模板签名**在当前 read-out 层、测试磁量、flat-threshold 设置下只被 V_th↓ 复现**——这**不是**"一般局部 E/I 不能复现"的 final verdict(见 §6.5 范围声明)。**更准的重述(用户 2026-06-15):局部成核能力 ≠ 外向传播接力能力**——V_th↓ 让核更容易形成一个能向外接力的 pulse;局部 E/I 能点火但活动困在核内/边界、传不成方向模板;这和 Stage 3 双灶"source 成核了但 clean template 不稳"是同一类现象。净效果是**收窄而非支持**:V_th↓ 在该读出层面不被这两类局部机制(测试设置下)等价替代。按 spec,V_th↓ 保留唯象标签,E/I 轴写 future direction。)

---

## §1 A0 指纹仪器冻结(user sign-off 2026-06-14)

冻结 event-level schema + extractor。primary = {axis_dir, pathway_width, onset_jitter};secondary = {latency_jitter, recruit_extent};deferred = {speed, event_size};amplitude_proxy = diagnostic。`pathway_width` = 相对 **LOCKED 几何轴 theta**(非 event-fitted)的垂直 robust span(p95−p5),<3 触点 NaN。`n_min_events=6` 为 A1/A3/A4 最低门。两道边界 loud-fail:name-alignment + `n_part==非空rank数`。baseline 从现有 oneend neg/pos 抽出(无新仿真):axis_err 中位 0°、sign +1/−1、pathway_width 2.6mm。详见 `results/.../fingerprint/baseline_oneend{,_summary}.json` + commit b955298。

## §2 A1-0a 点火可行性(`a1_0a_feasibility/`)

单灶 oneend_neg,`mean{17,16.5,16} × std{0.5,1,1.5}`,seed 1,T=2000,只报可点火性。

| mean \ std | 0.5 窄 | 1.0 中 | 1.5 宽 |
|---|---|---|---|
| **17.0** | clean **5** ✓ | clean **8** ✓ | clean **9** ✓ |
| 16.5 | clean 8 ✓ | clean 9 ✓ | 14 ev / **0 clean** ✗ |
| 16.0 | clean 9 ✓ | 20 ev / **0 clean** ✗ | 8 ev / **0 clean** ✗ |

**关键(推翻 plan 旧假设)**:窄档(std=0.5)在 mean=17.0 **并不** gate-fail——三档在 17.0 都干净点火。降 mean 反而让中/宽档**过度点火成不可读事件**(事件多但 0 clean directional)。

**对 A1 formal 的影响**:fixed-mean = **17.0** 就是三档共同的干净点火工作点,**不需要降 mean** → plan 担心的 mean/工作点混淆在 17.0 基本消解。这是个比预期更干净的结果。

## §3 A1 formal — 阈值离散 → 指纹(fixed-mean 17.0 + matched-rate 敏感性)= 描述性 NULL

`a1_formal/`,9 cells(std{0.5,1,1.5} × seed{1,2,3})@ mean=17.0 T=3500。结果(`a1_formal_results.json` + `figures/a1_heterogeneity_fingerprint.png`):

| 档 | clean 事件(3 seed 合) | pathway_width | axis_err 中位 | onset top1 |
|---|---|---|---|---|
| 窄 std=0.5 | **26** | 2.6mm | 0° | 1.00 |
| 中 std=1.0 | **40** | 2.6mm | 0° | 0.975 |
| 宽 std=1.5 | **47** | 2.6mm | 0° | 1.00 |

- **primary 三件套在三档完全相同**(`pathway_width_degenerate=true`、`axis_err_degenerate=true`、入口几乎全 A0)→ Kruskal 退化(无方差)。matched-rate 敏感性(count-match 到 26/档)同样无差(值恒等)。
- **唯一的档间差异是点火频率**:clean 事件随离散度上升(窄→宽 26→40→47)。
- **结论(描述性 NULL)**:阈值离散度在本电极分辨率下**不产生可区分的传播指纹**,只调制点火频率。这证伪了"离散度 → 指纹"这条;与 plan 预案一致("若没有指纹效应,E/I 那条更值得投")。注:pathway_width 的"2.6mm 恒定"部分是 montage 离散化(虚拟电极间距把 ⊥ 展宽量化到同一格),所以是"在该读出分辨率下无差",非"物理上严格无差"。

## §4 A4 paired single-focus = A1 宽-vs-中 pairwise(同样 NULL)

A4 plan = "分别单灶、同 seed/同 geometry/同工作点,一端 std=1.5 一端 std=1.0"——正是 A1 数据里 **std1.5 vs std1.0** 的两两对比(同一套 fixed-mean/geometry/seeds),无需额外仿真。结论 = **宽与中两个不同离散度的灶,传播指纹无可区分差异**(pathway_width/axis_err/入口全同),只有点火频率不同(宽 47 vs 中 40)。即"两个不同内部属性(离散度)的灶给同一指纹"。**true two-focus(同网双灶)本轮不做**(scope 锁定 paired single-focus)。

## §5 A2 — 局部 E/I 病灶引擎(guarded edit + re-bless logged diff)

引擎在 `results/topic4_sef_hfo/lif_snn/engine/`(**gitignored**),由 `engine_versions.json` sha 守护。本节是这次改动的**durable 记录**(git 不收 gitignored 文件)。

**改了什么**:`build_connectivity_rot` 加一个**可选 per-neuron 突触权重缩放场**,在 partner 采样**之后**施加(rng 抽取顺序不变 → 默认路径 BIT-IDENTICAL,已对 pre-edit snapshot 验证:a_tot=22272000.0 / g_tot=14844211.2 完全相等):
- `local_scale_EI`:E **靶**在核内 → 乘 w_EI(该 E 细胞的 GABA 输入),即 perisomatic 抑制塌陷。**target-indexed scalar,绝不碰 w_II**。
- `w_EE_gain_core` + `core_mask_E`:E→E 边的**源与靶都在核内** → 乘 w_EE,即复发兴奋簇。**edge-indexed both-in-core**(只看源会热扩散出核、破坏 bare-sheet-quiet);源用 **E-local** 下标。
- `tau_I`(A4 行)= 全局内禀量,**DEFERRED**,不做空间场。

**re-bless logged diff**(2026-06-15):`engine_versions.json` 6 文件中**仅** `connectivity_rot.py` sha 变:`9694171ae636 → fe8527c6e440`;`kick_probe.py / params.py / model.py / connectivity.py / lfp.py` 不变。

**TDD**(`tests/test_sef_hfo_axisA_ei_engine.py`,5 pass,直接 import 不过守护 runner):默认确定性 + explicit-noop 一致;w_EI↓ 只降 in-core E 的 GABA mass(结构不变);w_II 永不被 EI 场缩放;w_EE↑ 只升 both-in-core AMPA mass;**out-core 靶即使有 in-core 源也不缩放**(both-in-core 判别子)。

**runner 接线**:`oneend_inhib`(w_EI↓)/ `oneend_recur`(w_EE↑)/ `oneend_combined` 三个单灶模式,FLAT V_th=18.0(可激性来自权重病灶、非降阈值),权重场进 `build_connectivity_rot`(**不进 `simulate_kick`**——权重建连时烘焙;spec §5 原措辞已改)。新增 `--ei-scale`(默认 0.5)/ `--ee-gain`(默认 1.5)+ config provenance。V_th 路径不变。commit 9eaaabd。

## §6 A3 — E/I 机制 → 指纹(事件率匹配)

**pilot(`a3_pilot/`,T=2000,ei=0.5 / ee=1.5)**:两种 E/I 病灶都点火(inhib 8 事件 / recur 7 事件),**全部 self-terminate(returned=True)**,bare sheet 仍安静(true_floor 0.00016)——但事件**空间局部**:参与触点数多为 1–6(<PART_MIN=7),axis 读不出(sign=None),inhib 仅 1/8、recur 0/7 达到可读方向模板。即**这个磁量下 E/I 病灶给的是自终止的小局部兴奋,不是 V_th↓ 那样招募成可读方向模板的事件**。

**A3-0a 磁量扫描(`a3_0a_scan/a3_0a_scan.json`,T=2000,seed 1)**:

| 病灶 | 磁量 | n_events | clean_dir | n_part_max | n_part≥7 | true_floor(bare sheet) |
|---|---|---|---|---|---|---|
| inhib | ei=0.5 | 8 | 1 | 7 | 1 | 0.00016 ✓安静 |
| inhib | ei=0.35 | 11 | 0 | 8 | 1 | **0.0207** ✗破 |
| inhib | ei=0.2 | 11 | 0 | 8 | 1 | **0.0219** ✗破 |
| recur | ee=1.5 | 7 | 0 | 6 | 0 | 0.00016 ✓安静 |
| recur | ee=2.0 | 10 | 0 | 8 | 1 | **0.0226** ✗破 |
| recur | ee=2.5 | 12 | 0 | 8 | 1 | **0.0223** ✗破 |

**A3 = 描述性 NULL(同一张网、受控对比)。** 三点:
1. **没有磁量能既保持 bare sheet 安静、又产生 clean 方向模板。** 温和磁量(ei=0.5 / ee=1.5)bare sheet 安静但事件空间局部(n_part 卡在 7–8、clean_dir 0–1);加强磁量(ei≤0.35 / ee≥2.0)把 bare-sheet 背景抬高 ~130×(true_floor 0.00016→0.022,**破坏"事件只从核点火"不变量**,spec §3 gate),却**仍不让事件长大**(n_part_max 还是 8)。
2. **事件率其实匹配上了,read-out 仍不同。** ei=0.5 的 nucleation 数(8)≈ V_th↓ 同工作点(a1_0a 宽档 9),但 clean-directional 数 1 vs V_th↓ 的 9——**在 nucleation 率匹配的前提下,E/I 事件局部不可读、V_th↓ 事件传播可读**。
3. **机制解读(受 A1 约束、coherent)——成核 ≠ 传播接力:** A1 已证**连窄档 V_th↓(几乎无离散度)都干净传播**,所以传播不是离散度 seed 驱动的,而是**降阈值**驱动的。更准的框架(用户 2026-06-15):V_th↓ 只降核内 E 阈值,但这让核更容易形成一个**足够同步/足够强/足够成形的 outgoing pulse**,经原本的各向异性 E→E 连接**招募核外正常阈值组织**;局部 E/I 也让核点火,但主要**增强/解抑核内活动**,**没把局部点火转成可传播的 traveling pulse** → 波困在核内/边界。**read-out 的"传播方向模板"签名在当前 read-out 层、测试磁量、flat-threshold 设置下只被 V_th↓ 复现**(**非**"一般局部 E/I 不复现"的 final verdict——见 §6.5)。

**口径(spec §3,严格)**:"**在读出层面、测试磁量内,局部 E/I 病灶不复现 V_th↓ 的 clean 方向模板**(nucleation 率可匹配但 read-out 不同)";按 spec 合同,**E/I 轴写 future direction,V_th↓ 保留唯象标签**;**禁止**"E/I 机制被证实 / 被证伪"。

**多 seed 确认(`a3_seed_confirm/a3_seed_confirm.json`,matched-nucleation 磁量 ei=0.5 / ee=1.5,seeds 1–3)**:NULL 跨种子稳健。

| 病灶 | clean_dir / seed | n_part_max / seed | bare-sheet 安静(全 seed) | local_robust |
|---|---|---|---|---|
| inhib ei=0.5 | [1, 0, 0] | [7, 3, 5] | True | **True** |
| recur ee=1.5 | [0, 0, 1] | [6, 4, 8] | True | **True** |

3 个网络种子下,两种局部 E/I 病灶都:每次 ≤1 个 clean 方向模板(多为 0)、事件局部(n_part_max 3–8)、bare sheet 全程安静。即"局部 E/I 点火但读不出方向模板"不是单 seed 偶然,**跨种子稳健**。图:`a3_0a_scan/figures/a3_ei_screen.png`(左=加强只抬背景不长事件;右=模板数跨 seed 贴 0、够不到 V_th↓ 参考)。

**screen 级 caveat(仍不是终极 verdict)**:T=2000、coarse 磁量格、单一 core(r=1.5/位置/单灶)、flat threshold、未跑 rate-band 内的 A3 formal(因无既安静又出模板的窗口)。未测的 future directions:E/I 核**加配 V_th 离散度 seed**、**更大 core**、**调 drive/core_r**。本轮按"阻塞降级继续"以**跨 3 seed 稳健的 screen-NULL** 收口。

## §6.5 范围声明:本轮 NOT 跑"所有异质性"(防 final-verdict 误读)

A3 是 **screen-NULL**,不是"局部 E/I 机制一般不能复现 V_th↓"的 final verdict。本轮**只在**单一设置下测了 `w_EI↓` / `w_EE↑`(T=2000、coarse 磁量、单 core r=1.5/单灶、**flat threshold**)。**讨论过但本轮没作为正式主结果执行**的:

- **`core_mean` 本身的变化**:只作点火可行性 / 工作点(A1-0a),**不作机制轴**。
- **`oneend_combined`**(E/I 两旋钮合并):runner 支持,但本轮主 screen **没把它做成正式结论**。
- **`tau_I` / 抑制时间常数**:当前引擎里更像 global intrinsic,**不适合首批局部空间场,后置**。
- **其他细胞参数异质性**(`tau_m` / `tau_ref` / 输入噪声 `sigma` / `drive`·`mu`):代码层有概念工具,**本轮没进 SNN Axis A**。
- **future-direction 类**:`core` size、`drive`、core-shell 梯度、boundary coupling、source-to-outside `w_EE`。
- **true two-focus 异质性对比**:plan 讨论过,但本轮 A4 实际**只是 paired single-focus,不是真正同网双灶**。

**结论纪律**:报告 / 主文档只写"**screen-NULL / Level 2 收口**",**不写 final verdict**;措辞用"在当前 read-out 层、测试磁量、flat-threshold 设置下只被 V_th↓ 复现",不写"specific to 降阈值"那种一般化口气。

## §6.6 下一轮计划:"成核 vs 传播接力" 最小 2×2 screen(不扫大参数海)

问题已从"哪种异质性改指纹"转成 **"什么条件让局部成核变成可传播模板"**。最小 2×2(两轴,不扫所有参数):

| 轴 | 水平 1 | 水平 2 | 问 |
|---|---|---|---|
| core 内成核机制 | `V_th↓` | `E/I↓/↑` | 谁能点火 |
| core 向外接力条件 | 原始边界 | 加强边界 / 扩大 core / 加 V_th seed | 谁能传出去 |

4 组(1 seed、T=2000、只看 gate,不做 formal 统计):① `V_th↓` r=1.5(positive ref)② `E/I` r=1.5(NULL ref)③ `E/I` r=2.5/3.0(core 太小?)④ `E/I` + mild V_th seed(core 内 17.5,问 E/I 是否需 threshold seed 才能输出 pulse;runner 已加 `--ei-vth-seed`)。**Go gate**:clean directional ≥6 ∧ axis_err<25° ∧ bare-sheet true_floor 不爆(<0.001)∧ n_part 稳定覆盖传播轴(非刚过 7)。**只对 survivor 做 3 seed confirm**;无 survivor **不**继续扫 `tau_m/tau_ref/sigma/tau_I`。优先级:E/I+mild V_th seed → larger core → 最后才 boundary/core-shell。执行见 §6.7。



split-half **AND** odd-even 双折门(各 ≥0.6,比模板级 forward_reverse_reproduced 的 OR 规则更严)。axis_dir 17 PASS(含 6 weak_axis + 4 degenerate 硬否);onset_jitter 34 PASS(用 cluster_rank 折间一致性作 proxy,per-fold earliest-prob 重算 deferred);**pathway_width 40 DEFERRED**(path_axis 只存全量 perp_spread、无 per-fold;23 个可日后重算)。4 个 borderline(一折≥0.6 一折<0.6,被 AND 门拦下,OR 规则会放过),最强 epilepsiae_1096(0.25 / 1.00)。纯描述、按数据集分层(mm 不跨库 pool)、无机制 label。

## §8 内部归档代号

axis A = E/I 局部病灶 + 阈值异质性 → propagation fingerprint;A0 fingerprint freeze(SCHEMA_VERSION=A0-frozen-2026-06-14);A1-0a ignition feasibility;A1 fixed-mean 17.0 + count-matched sensitivity;A4 = A1 wide-vs-mid pairwise(paired single-focus);A2 engine = local_scale_EI(w_EI↓ target-in-core)/ w_EE_gain_core+core_mask_E(w_EE↑ edge both-in-core);engine re-bless connectivity_rot.py 9694171→fe8527c;A3 rate-matched 0.8–1.25×;A5-real reliability split-half AND odd-even ≥0.6。
