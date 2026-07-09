# M4 动态 q_I：两轴向灶自发耗竭抑制 → runaway；全局除法池 S_G 的刹车/围堵

E1146 真实布局（twoend_equal，L=20），两个小轴向灶（源/汇核）**自发**放电（KICK_BOOST=0），
"抑制油箱" `q_I` 随事件动态耗竭（`k_q>0`，非冻结），对比开/关全局除法池 `S_G`。代表图和复核数据在
`results/topic4_m4_dynamic*` 系列目录。**单位**：`res["rate_E"]` 已是每神经元 Hz
（`kick_probe.py:363` 已换算），图里直接用，runaway 判定线 = 120 Hz。

### fig_m4_dynamic_qi.png

代表性 4 联（`alpha_G=6` 一档，k_q=0.35/0.18 × 开/关池）。A 衬底两灶+E→E 轴；B `q_I(t)` 一次事件
从满抽到底（1.0→0.05）；C 群体放电率：不开池 latch 到 ~509 Hz、开池 ~300 Hz（**砍强度**），两者都
远超 120 Hz runaway 线且铺满全场；D `S_G(t)` 涨到 0.83。**这一档：刹得住强度、关不住范围**。
**关注点**：C 面板红/蓝的稳态高度差 = 池的刹车幅度；但两条都在 runaway 线以上——alpha_G=6 不 bound。

### fig_m4_phase_diagram.png

`(k_q × alpha_G)` = 耗竭速度 × 池强度的 pass1 相图。验收口径以长窗复核为准：`alpha_G≤12` 主要是延迟 runaway，
`alpha_G=16` 打开一个窄的 bounded window，`alpha_G≥20` 又出现过冲/延迟失控。40 s 多 seed 后，最稳健锚点是
`(k_q=0.10, alpha_G=16)`（3/4 seed 无 runaway）；`(0.25,16)` 只能算 15 s 确认，40 s/多 seed 不稳健。
**关注点**：绿色区域不能读成“完整发作态”；它只说明除法池把 runaway 压成非失控持续吸引子，且窗口很窄、marginal。

### fig_m4_alphaG_slice.png

k_q=0.10 沿 `alpha_G` 的切片，按每格最长确认窗给 verdict。非单调是核心信号：池太弱时 `q_I` 抽到底并失控；
`alpha_G≈16` 附近出现 bounded 缺口；池更强时延迟反馈/过冲又把系统推回晚发失控。**关注点**：不要写成
“越强全局抑制越安全”，真正结果是窄 Goldilocks window。

### fig_m4_oscillation_diag.png

围绕 `alpha_G=16/20/24` 的振荡诊断图，显示 bounded window 上边界附近存在延迟反馈振荡放大的迹象。该图只能支持
“Hopf-like / delayed-feedback oscillation” 的经验表述，不能单独证明 Hopf bifurcation。
**关注点**：它解释为什么 `alpha_G=20` 比 `alpha_G=16` 更容易晚失控，但不是严格分岔证明。

### fig_m4_continuity_eigenmode.png

多 seed fine-scan 的连续性和经验 leading-mode 诊断：左图看 runaway time 随 `alpha_G` 的边界，中图看经验增长率
`sigma(alpha_G)`，右图看复模频率。`sigma` 穿零并不干净，因此这个图是“支持/反证 Hopf-like wording”的诊断，
不是全模型 Jacobian 结果。
**关注点**：可写“经验上接近 delayed-feedback oscillatory boundary”，不要写“已证明 Hopf”。

### m4_dynamic_qi_runaway.gif

空间活动 no_pool vs pool（k_q=0.35，0–800ms）：静息 → ~350ms 点火 → 铺满全场；池版扩散略慢、强度低，
但**同样铺满**（alpha_G=6）。
**关注点**：两栏都从两灶点起、最终整片亮——直观印证 alpha_G=6 只降强度不关范围（与相图右下角的围堵格对比看）。

### m4_seeg_readout.gif

在按 E1146 病人真实电极几何搭的仿真皮层片上，两个小轴向灶自发放电把"抑制油箱"（`q_I`）慢慢抽干，我们在病人**真实触点**（`SCL`/`ICL` 两根杆共 15 个触点）位置放"虚拟深部电极"，录下医生会在这些电极上看到的信号，对比全局共享的"除法式"抑制池（`S_G`）开/关。**不开池**（`alpha_G=0`）时油箱在约 **386 ms** 就被抽到地板、整片网络失控冲到约 **477 Hz**——虚拟电极上**所有**触点齐刷刷跳上一个饱和高平台；**开池到 bounded 档**（`alpha_G=16`）时放电被摁成一个"有界、铺得很宽（约六成网络、一条满宽横条）、持续但不失控"的状态：率封顶约 **81 Hz**（低于 120 Hz 失控判定线）、全网平均 `q_I` 停在地板 `0.05` **之上**（0.14）、池 `S_G` 稳定在约 0.43——压在这条活动带上的触点（`ICL`）读到早期几次事件后的低幅持续活动，离带子远的触点（`SCL`）基本安静。**要点**：这个池能把"抑制资源耗尽→失控"摁成有界持续态、不再全场失控，**但它只是把住、没把它收回去**（带内 `q_I` 钉在地板、放电自锁、不会自己终止回到间期），所以是"有界持续吸引子候选"、**不是**会走完的完整发作周期；失控 / 强直饱和本身**不是**发作样（ictal-like）事件。单种子（seed=1）单轨迹可视化诊断；为省算力，失控臂只真跑到 1000 ms 再原样保持其饱和平台（该态是近乎恒定的饱和不动点，保持忠实），有界臂全程真跑。

**关注点**：对比两行虚拟 SEEG——池开(上)时压在活动带上的 `ICL` 触点低幅持续、远端 `SCL` 安静；池关(下)时所有触点在 386 ms 失控时刻齐刷刷跳到饱和高平台。这正是临床医生会在病人自己电极上看到的差别（脚本 `scripts/paper_figures/plot_fig_m4_seeg_readout_gif.py`；重渲染 `--render-only` 读同目录 `m4_seeg_readout_cache.npz`）。

---
**红线 / 当前口径（2026-07-09 pass1 验收）**：M4 pass1 可以验收为**机制筛选通过**，不能验收为完整 seizure cycle。

**已锁结论**：全局除法池 `S_G` 能把 q_I-耗竭 runaway 压成一个**窄窗口的 bounded sustained attractor**。40 s 多 seed
后主锚点是 `(k_q=0.10, alpha_G=16)`：3/4 seed 无 runaway；`(0.25,16)` 15 s 可过，但 40 s/多 seed 不稳健。
这个 attractor **空间宽**（满宽横条、位置随 seed 漂=对称破缺、非 localized ictal core）、**marginal**、**不可撤回**，
因此不是完整 seizure-like cycle。⚠️措辞：sheet-mean q_I above floor 不是全局 `q_I` preserved，局部仍可触底。

**它回答了一个关键问题**：global divisive inhibition 能不能造出 runaway 与 transient 之间的第三态？——**能**。

**它没回答 / 明确阴性**：
- 是否 localized ictal core？——**否**，当前是 broad stripe。
- 能否回间期？——**否**。`qI_refill` 不能把系统带回稳定间期，系统重新抽干并回到有界吸引子；`inhibitory_pulse`
  释放后反跳成 one-shot runaway-like burst（峰值 317Hz，runaway_ms=8534）。
- 是否完整终止？——**否**，池只 bound 不 terminate，需要下一代 adaptation / `g_K` / 慢恢复负反馈。

**机制归因已做**：dynamic divisive `S_G` 有界；matched-subtractive 仍失控；clamped-SG 压死活动。结论不是“任意强
全局抑制都能 bound”，而是“活动依赖、动态、除法式 recurrent-gain 抑制池”能打开窄的 bounded window。

数据：`results/topic4_m4_dynamic_{sweep,confirm,delay,longconfirm,multiseed,extend,confirm2,reversibility,mechanism}/`；
验收文档：`docs/archive/topic4/sef_hfo/m4_pass1_divisive_shared_pool_acceptance_2026-07-09.md`。
