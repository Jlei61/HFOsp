# Topic 5 V2 Phase-1 图 — interictal HFO geometry ↔ early-ictal multi-band energy field alignment

判读 tier = **exploratory candidate early-ictal spatial recruitment scaffold（cohort 层；非 formal/机制）**。数据 = full n_perm=1000（narrow n=20 / broad n=17）。样式遵 `docs/figure_style_guide.md` §0（紧坐标 / 单共享 colorbar / render→目视）；按 user 要求 F1 用红蓝 diverging（非 viridis）、F2 用 muted-red 标显著。脚本 `scripts/plot_topic5_v2_phase1_figures.py`。三图各答一个独立科学问题（§7）。

### phase1_F1_observed_maxAB_heatmap.png
subject×band 的发作 onset 后 0–20s 能量空间场 vs 间期 HFO 几何 maxAB |corr| 热图（**红蓝 diverging，以 0.5 为心（蓝<0.5<红，突出弱对齐的蓝格）；显著 cell（该 subject 自身空间 null p<0.05）标白星黑边**，两池同心可比；末行=cohort 中位带数值；**黑虚线分隔 primary(7) | composite(4)**）。narrow(20) 整体偏红、broad(17) 偏蓝 → narrow>broad 一眼；无单频带列独占（band-generic）。是描述性幅度（平滑场 |corr| 相对 0 天然偏高），不是显著性判决。
**关注点**：narrow>broad + band-generic + 上红下蓝的 per-subject 异质（呼应 F3）；别把颜色当显著（显著看 F2）。

### phase1_F2_null_per_band.png
每 primary 频带的 per-subject Δ（对齐 − 该 subject 自身弱空间 null 中位）**violin 分布 + 背景散点（每点=1 subject）**；柔和 muted-red(过)/gray(n.s.)；**黑横条=cohort Δ（被检验的统计量）**；crimson `*`=该带过 max-over-bands **family-wise (FWER) 校正**、`n.s.`=不过。FWER=家族错误率，控制"7 带里出现任意假阳性"的总概率（非单带侥幸）。两池都 6/7 primary 过、唯 ripple_high n.s. → **NOT ripple-specific**。
**关注点**：正面只在 weak/subject-wide null 下（反保守、likely inflated），formal within-shaft Gate A 未评估（2/20）；ripple 最弱；violin 很宽、下缘常压到 0 以下（per-subject 异质，呼应 F3）。

### phase1_F3_per_subject_stability.png
每 subject 显著频带数 n_sig（Δ>0 且该 subject 自身 null p<0.05，of 7 primary），按 n_sig 排序、条色=空间 null 强度档（深蓝=within-shaft strong / 中=distance-bin / 浅=subject-wide weak）。暴露 cohort 6/7 是**聚合**：narrow 中位仅 2/7、≥5/7 仅 3/20；多带阳 subject 大多不是 within-shaft-strong（唯 1146 深蓝且高）。
**关注点**：cohort 6/7 ≠ per-subject 稳健——这张是整份验收的承重 caveat。

### phase1_v2_W2_subject_phenotype.png
W2「谁？」收尾图，两面板各答一个独立问题（纯后处理 phase1_v2_subject_phenotype.csv，无新 null / 仿真 / KMeans）。**A（按 subject tier 着色的频带梯度散点）**：把每个被试放在「HFA(80–250 Hz) 减 low(1–13 Hz) 早发作对齐差」这条横轴上（<0 偏低频、>0 偏 HFA、≈0 band-generic），narrow / broad 两行；颜色=三档 tier（strong / directional / weak_absent），正负由点相对 0 线的位置表示、不再用红蓝重复编码。多数被试挤在 0 附近、跨 tier 混杂——梯度是 band-generic 且异质，没有某一频段独占。**B（相关性筛查）**：多频带阳性（7 个 primary 里显著的频带数）对每个候选特征的 |Spearman r| 横条，narrow（紫）/ broad（绿），红虚线=锁定判据 |r|=0.4、星=p<0.05；上半=真正独立的被试特征（发作次数 / 触点数 / 原始对齐幅度 / 跨发作一致性 / 频带梯度），几乎全在 0.4 以下（唯 cross-seizure consistency 在 narrow 过、broad 不过→不稳健），下半=同一对齐差向量的再描述（正频带铺开度、各频段组对齐中位），天然共变必然过闸、不算「独立预测」。合起来=没有单一干净 phenotype 能预测多频带阳性（这是预期结果，n 小、效应异质）。band_profile_group 是描述性分桶（±0.05 / 0.6 是分桶刻度、不是检验阈值），不是 KMeans / 统计 subtype 主张。（内部代号：n_sig_7bands / HF_minus_low / band_genericity_index / profile_entropy / tier）
**关注点**：A 看「梯度贴 0、跨 tier 混杂 = band-generic 且异质」；B 看「独立特征几乎全在 0.4 以下（cross-seizure 只 narrow 过），过闸的都是同一对齐差向量的再描述」→ no single clean phenotype；分桶是描述性、非 subtype。
