# Topic 5 V2 Phase-1 图 — interictal HFO geometry ↔ early-ictal multi-band energy field alignment

判读 tier = **exploratory candidate early-ictal spatial recruitment scaffold（cohort 层；非 formal/机制）**。数据 = full n_perm=1000（narrow n=20 / broad n=17）。

**✅ 验收核心结论（2026-07-06）**：**间期传播模式（HFO 顺序几何 G_HFO）与发作早期宽带能量空间相关 = 队列一致的承重结论**；**频带偏向性是个别患者现象（per-patient band bias），非队列级机制主张**。

**图集约定（2026-07-06 重制）**：F1 / F2 / W1 都 **narrow / broad 各一版独立文件**（不并排）；ticks + 内部字放大；band 命名 **R = 80–150 Hz（ripple）· FR = 150–250 Hz（fast ripple）**。W1 是新增的 **1/f 主图**（这部分核心 = 看 1/f）。F3 仍单张合并。脚本 `plot_topic5_v2_phase1_figures.py`（F1/F2/F3）+ `plot_topic5_v2_W1_aperiodic.py`（W1）。每图各答一个独立科学问题（CLAUDE.md §7）。

### phase1_F1_observed_maxAB_heatmap_{narrow,broad}.png
subject×band 的发作 onset 后 0–20s 能量空间场 vs 间期 HFO 几何 maxAB |corr| 热图（红蓝 diverging，以 0.5 为心，蓝<0.5<红；**行按每被试显著 primary 频段数降序**〔显著多的在上〕；显著 cell〔该 subject 自身空间 null p<0.05〕标白星黑边；末行=cohort 中位带数值；黑虚线分隔 primary(7) | composite(4)，band 名 δ/θ/α/β/γ/R/FR | low/LVFA/Rf/Rs）。narrow / broad **各一版独立文件**。narrow 版整体偏红、broad 版偏蓝 → narrow>broad；无单频带列独占（band-generic）。描述性幅度（平滑场 |corr| 相对 0 天然偏高），非显著性判决。
**关注点**：行序＝显著频段数（顶端多星、底端稀疏，呼应 F3 的 per-subject 异质）；narrow>broad + band-generic；别把颜色当显著（显著看 F2）。

### phase1_F2_null_per_band_{narrow,broad}.png
每 primary 频带的 per-subject Δ（对齐 − 该 subject 自身弱空间 null 中位）**violin 分布 + 背景散点（每点=1 subject）**；柔和 muted-red(过)/gray(n.s.)；**黑横条=cohort Δ（被检验的统计量）**；crimson `*`=该带过 max-over-bands **family-wise (FWER) 校正**、`n.s.`=不过。narrow / broad **各一版独立文件**；star 保持在框内（比例 headroom）、去右上 spine、legend 移框外右上（否则遮住最右 R 星）。FWER 控制"7 带里出现任意假阳性"的总概率。两池都 6/7 primary 过、唯 **FR（150–250）n.s.** → **NOT ripple/FR-specific**。
**关注点**：正面只在 weak/subject-wide null 下（反保守、likely inflated），formal within-shaft Gate A 未评估（2/20）；FR 最弱；violin 很宽、下缘常压到 0 以下（per-subject 异质，呼应 F3）。

### phase1_v2_W1_aperiodic_{narrow,broad}.png
**这部分的 1/f 主图**（回答"扣掉 1/f 背景后还剩什么"——本部分核心）。narrow / broad 各一版。
**A「1/f 怎么测的」**：一条代表性触点的基线功率谱（log-log 黑线）+ **实测拟合的 1/f 直线**（红虚线，log-log OLS over [1,200] Hz，标 slope/r²）+ 灰阴影＝超出 1/f 地板的余量（band excess）+ 竖灰带＝挖掉的工频 bin；**顶部 band 轴**（δ/θ/α/β/γ/R/FR 落在各自频率区间，让人知道每个鼓包属于哪个频带）。选点＝clean-hugging fit（r²≥0.96、低 overshoot）里 gamma-excess 最大的触点。
**B「扣 1/f 前后存活塌缩」= residual 存活小方阵**：行＝两层 residual 控制（− broadband／− 1/f），列＝7 primary 频段；格填色＝该残差场仍过 max-over-bands FWER（弱空间 null）、灰＝不过，格内写 p 值；γ 的 −1/f 格＝金色（唯一两池都稳）。**故意不画 raw 层和 Δ 高度**——raw 是 F2 的信息（标题"raw baseline 6/7 → see F2"），Δ 是 F2 的 y 轴，都不重复。从上往下读＝塌缩：narrow 4 格 → 只剩 γ；broad 4 格 → β/γ/FR 3 格，但只 γ 金色 robust，β 近奈奎斯特脆弱、FR＝family-ceiling 假象（标题标注）。
**关注点**：A 看"1/f 是 log-log 一条直线扣掉、鼓包落在哪个 band"；B 看"扣 1/f 后填色格缩到只 γ"＝频段特异大半是 1/f 背景、**NOT ripple-specific**；γ 残差是描述性、**非 LVFA-specific 机制**。承重口径＝间期传播 ↔ 发作早期宽带能量相关，频带偏向是个别患者现象。

### phase1_F3_per_subject_stability.png
每 subject 显著频带数 n_sig（Δ>0 且该 subject 自身 null p<0.05，of 7 primary），按 n_sig 排序、条色=空间 null 强度档（深蓝=within-shaft strong / 中=distance-bin / 浅=subject-wide weak）。暴露 cohort 6/7 是**聚合**：narrow 中位仅 2/7、≥5/7 仅 3/20；多带阳 subject 大多不是 within-shaft-strong（唯 1146 深蓝且高）。
**关注点**：cohort 6/7 ≠ per-subject 稳健——这张是整份验收的承重 caveat。

### phase1_v2_W2_subject_phenotype.png
W2「谁？」收尾图，两面板各答一个独立问题（纯后处理 phase1_v2_subject_phenotype.csv，无新 null / 仿真 / KMeans）。**A（按 subject tier 着色的频带梯度散点）**：把每个被试放在「HFA(80–250 Hz) 减 low(1–13 Hz) 早发作对齐差」这条横轴上（<0 偏低频、>0 偏 HFA、≈0 band-generic），narrow / broad 两行；颜色=三档 tier（strong / directional / weak_absent），正负由点相对 0 线的位置表示、不再用红蓝重复编码。多数被试挤在 0 附近、跨 tier 混杂——梯度是 band-generic 且异质，没有某一频段独占。**B（相关性筛查）**：多频带阳性（7 个 primary 里显著的频带数）对每个候选特征的 |Spearman r| 横条，narrow（紫）/ broad（绿），红虚线=锁定判据 |r|=0.4、星=p<0.05；上半=真正独立的被试特征（发作次数 / 触点数 / 原始对齐幅度 / 跨发作一致性 / 频带梯度），几乎全在 0.4 以下（唯 cross-seizure consistency 在 narrow 过、broad 不过→不稳健），下半=同一对齐差向量的再描述（正频带铺开度、各频段组对齐中位），天然共变必然过闸、不算「独立预测」。合起来=没有单一干净 phenotype 能预测多频带阳性（这是预期结果，n 小、效应异质）。band_profile_group 是描述性分桶（±0.05 / 0.6 是分桶刻度、不是检验阈值），不是 KMeans / 统计 subtype 主张。（内部代号：n_sig_7bands / HF_minus_low / band_genericity_index / profile_entropy / tier）
**关注点**：A 看「梯度贴 0、跨 tier 混杂 = band-generic 且异质」；B 看「独立特征几乎全在 0.4 以下（cross-seizure 只 narrow 过），过闸的都是同一对齐差向量的再描述」→ no single clean phenotype；分桶是描述性、非 subtype。

### phase1_v2_W3_trajectory.png
W3「什么时候？」主图，两面板各答一个独立问题（纯后处理 phase1_v2_alignment_trajectory.csv + phase1_v2_trajectory_contrasts.csv，无新 null / 仿真 / 重跑对齐）。**测的是**：间期这个人的 HFO 几何和「发作前后每一小段时间里各频带能量在空间上怎么分布」这两张图有多像（一个 0–1 的相似分数，取 7 个主频带的中位数当作「不挑频带的总相似度」），把这个相似度沿「发作前 100 秒到发作后 20 秒」画成一条时间曲线，用来分辨这套空间底盘是**发作前就一直摆在那儿**（像解剖结构、静态易感场）还是**发作起始那一刻才被点亮**（起始招募）。**怎么测的**：相似度逐层取中位数（每个时间窗 → 每次发作 → 每个被试 → 全队列，统计单元＝被试，不是窗口）；主检验＝对每个被试算「靠近发作的一档 减 更早一档」的配对差，再问「如果这个上升只是随机的、把每个被试的正负号随便翻，差的中位数应该落在 0 附近」，实测偏离 0 多少（符号翻转置换，n_perm=20000 或被试数≤14 时精确枚举，双侧）——不是打乱窗口标签（窗口之间自相关会把 p 做小）。**A（队列轨迹，narrow/broad 两行）**：x＝相对 EEG 起始的时间（5 个 bin 中心 −80/−45/−20/0/+15 秒），y＝相似度，蓝实线＝以 EEG 起始为锚（主），橙虚线＝以临床起始为锚（敏感性对照），误差棒＝被试间 IQR，竖点线＝EEG 起始。**B（三个配对对比 + 显著性）**：near−far / post−far / post−near 三个配对差的队列中位数横条，池×锚四条一组，星＝该对比在被试层符号翻转 p<0.05。**揭示了什么**：发作前四档都**高且大致平**（near 减 far ≈ 0，没有逐步爬升 → 排除「发作前逐渐逼近阈值」那种读法），到发作后 early_post 才小幅抬一点；这个小上升在 broad 池、以 EEG 起始为锚时被试层站得住（符号翻转 p≈0.004–0.005），narrow 池借线（符号翻转 p≈0.06、Wilcoxon<0.05）；换成临床起始锚，上升被抹平甚至压低、全不显著——所以「看得清这个起始上升」要靠 EEG 锚，临床起始滞后会把它糊掉。合起来＝**一块发作前就静态高的空间底盘 + 发作起始一个小幅、偏弱、EEG 锚才看得清的上升**；判读停在 exploratory candidate scaffold（描述性、探索性档）——这是一块描述性的「候选早发作空间招募底盘」，它到底算不算病理临界模态属于本阶段范围之外、留待 Phase-2 评估。不挑单一频带（band-generic、NOT ripple-specific）。（内部代号：band_generic_scaffold_score = median align_abs_maxab over 7 primary bands / epoch_region 五档 / sign-flip subject-level / periictal(EEG) vs periictal_clin）
**关注点**：A 看「前发作平、early_post 小抬、EEG 实线比 clinical 虚线干净」；B 看「只有 broad-EEG 两个 post 对比带星、narrow 借线、clinical 全不带星」→ 静态高底盘 + 发作起始弱上升、以 EEG 锚为准；far pre-ictal 一档覆盖偏薄（大 EEG–临床间隔的发作，其 EEG −100~−60s 落在 −130s 缓存之外），别把最左那点当强证据。另外相似分数是「对当前窗里在场的主频带取中位数」，约 7% 的窗（各档都有、early_post 略多到 ~11%）只有 5/7 主频带——掉的永远是那两条 80–250 Hz HFA 频带（一起被跳过）；这是刻意的（强行要求全 7 条会进一步稀释本就偏薄的远端窗），要求全 7 条的敏感性分析里头条幅度变化 ≤~20%（如 broad post−far 中位 +0.0837→+0.0668）、显著性判定全不变（broad 两个 post 对比仍 p≈0.004、narrow 仍借线 ~0.06）→ 结论稳健；median_n_bands 已进 trajectory csv 备查。
