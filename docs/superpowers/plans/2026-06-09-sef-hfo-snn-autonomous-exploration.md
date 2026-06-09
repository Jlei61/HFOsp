# SEF-HFO SNN 异质性 — 自主探索会话计划（2026-06-09，user 离开期间）

> 承接 mean-scan（archive §9，找到真实点火边界 ~16.3–16.9mV + 宽尾把边界推 +0.57mV）。
> user 指示：按 spec 继续推进异质性、多做参数 + 机制探索仿真。本文件 = 本次自主会话的
> 锁定议程 + advisor 三条加固 + **停止条件**（防止无监督下 sprawling）。
> 所有结果保持 **screen / 探索级**，standing caveats（单工作点、小样本独立网络、非 cohort 裁决）；
> 不把 "screen" 继续往上爬级。**结论被推翻 → 停下大声标记，不自行抹平。**

## 锁定议程（只做这些，不中途发明 D/E/F）

### A — 边界 2nd pass（advisor 明确 deferral，nails headline）
- 细网格 means = [18.0, 17.0, 16.75, 16.5, 16.25, 16.0]（18=sanity 锚 + 17 bracket + 0.25 分辨率穿过边界）× std {wide 1.5, narrow 0.5}。
- **FRESH 连接种子 7–14**（8 张新网络，advisor 催化剂）× 2 fn = **16 独立实现/格**。**禁止**复用 coarse 的 conn 1–6（确定性 → 共享 means 上 bit-identical、零新信息）。
- 输出独立文件 `mean_scan_fine_metrics.json` + 图 `mean_scan_fine.png`（**不覆盖** coarse）。
- **预注册读法 = 复制（replication）**：① sanity mean=18 ≈ 0；② 边界是否仍落在 16.3–16.9（独立网络上）；③ 宽曲线是否仍在窄曲线左侧（wide-left-of-narrow ordering）；④ 收紧的边界位移 / 16.5 档纵向对比。fresh 种子 → coarse+fine 在共享 means {18,17,16.5,16} 可**合并**（不重复计数）。
- **overturn 信号**：若边界大幅偏离 16.3–16.9，或 wide/narrow 顺序翻转 → 停、标记、不抹平。

### B — kick 幅度稳健性（roadmap P1 explicit）
- 固定 mid 核 + end kick。KICK_BOOST ∈ {1×, 2×, 3×}·ν_θ。
- **只扫 matched + baseline**（advisor：igniting 条件自点火在 [0,150]、kick 在 150ms，**幅度不变 by construction** → 不浪费种子）× 多种子。**真问题 = matched 在 3× 更强刺激下是否还是 null**（matched 保持 evoked_clean、d_core_paf 有效）。
- 一个 igniting 格子**只跑一次**作 "pre-kick 不变" sanity 面板，不做种子扫。
- 输出 `kick_amp_metrics.json` + 图。
- **预注册读法**：① matched d_core_paf 在 1×/2×/3× 是否都 ≈ 0（null 稳健）；② baseline 自身不因更强 kick 自点火；③ igniting sanity = pre-kick 轨迹三幅度 bit-identical。
- **overturn 信号**：若 matched 在 3× 出现真诱发效应（之前 2× 没有）→ 停、标记。

### C — 核尺寸轴（guarded stretch，A+B 顺利且有时间才做）
- patch_radius ∈ {0.3, 0.45, 0.6}（**cap ≤ 0.6**，advisor：大核 = 更多低门槛尾 → spontaneous nucleation 随 r² 升，**baseline 宽核自身可能自点火**、毁掉干净参考——就是当初 false "matched reduces synchrony" 的坑）。
- **守护**：r=0.6 baseline 宽核先做 **kick-OFF 自发爆发检查**（必须安静 ~2Hz）才信任任何 evoked 度量。
- 框架成 "**自点火的临界核尺寸**"（finite-size 问题），不是单纯"大核更易点"。
- 若 toss-up / 时间紧 → **descope**（A+B 是高置信产出）。

## 停止条件（advisor hygiene）

- 做完 **A + B（+ guarded C）**，每个**独立输出文件**（launch 前验证 path 是新的——已吃过 L=1 clobber grid_metrics.json 的亏）、eyeball、archive + memory、commit。
- 然后写**一份合并 digest**给 user 回来看。**不**中途加新实验。
- 全程 screen-grade + standing caveats；overturn → 标记不抹平。

## 不变量（承 roadmap §不变量）

gap-limit 不放宽（std 锁 1.5/0.5，mean 下移记录）；matched-null 写"没复现不证否"；B 侧 tail-driven 独立机制不并链；每个结果 JSON 带 full provenance（git hash + engine checksum + config + seeds + metric version）。

---

## 结果 DIGEST（2026-06-09 自主会话收尾，user 回来看这段）

三个实验全部完成、各自归档（archive §10/§11/§12）+ 出图 + 提交（commits cb07c73 / 873f364 / 0634cb2）。**全部 screen-grade、没爬级、没推翻 standing 结论。** 白话三段：

**测了什么 / 揭示了什么（三件事，一条主线）**

主线：放电网络里一小块"病灶核"会不会在戳之前**自己先烧起来**（自点火），由什么决定。前面已知"压低核平均门槛 → 自点火，存在一个真实点火边界 ~16.3–16.9mV，宽参差的低门槛尾巴帮点火"。这一轮把这条主线在三个方向加固：

1. **A — 边界复制（加密 + 全新网络）**：在 8 张**全新独立网络**上重做细网格，边界（宽 16.88 vs 之前 16.86）、宽尾把边界往高平均推、窄参差亚阈升温曲线——**全部复制，无推翻**。最强判别现在是 16.75 档**宽 16/16 vs 窄 2/16**。→ 边界 + 尾巴效应**在两组独立网络上各自成立**，置信度升一档。

2. **B — 更强刺激下还是 null 吗**：固定核，把戳的强度调成 1×/2×/3×。**结论 = matched（纯参差）的诱发同步在锁定强度（2×）和更强（3×）下都贴零 → null 对更强刺激稳健**（这是 roadmap 明确要答的问题，答 YES）。**意外发现（已大声标记、没抹平）**：在**弱戳（1×）**下，收窄参差反而让核招募的细胞**明显更少**（−0.41，10/12 网络）——因为弱戳时只有低门槛细胞够得着，宽核有尾巴、窄核没有。**这跟点火的尾巴是同一个机制**，方向跟 Rich 文献相反（收窄→更少同步）。注意：这**不是**我们守的"更强刺激冒效应"那种推翻——locked-kick 的 null 稳稳的，只是补了一个"弱刺激 regime 尾巴也管招募"的细化。

3. **C — 病灶核越大越危险吗（新机制轴）**：把核半径从 0.3 加到 0.6mm（395 → 1628 细胞）。**结论 = 核越大、越浅（越健康）的平均就能自点火**（宽边界 <16 → 16.5 → 17.25 随尺寸单调上移）；而且**有一个临界核尺寸**——r=0.3（395 细胞）无论多易兴奋都不自点火（小核撑不起，有限尺寸效应）。宽参差在每个尺寸都比窄容易点（尾巴专属）。**advisor 担心的"大核 baseline 自己烧起来毁掉参考"在 L=3 没发生**（守护干净 0/0/0；那个坑只在 L=1 小网络出现，守护正确分辨了）。

**机制收口（一句话）**：自点火需要"够多低门槛细胞**集体**点着"。**尾巴宽度（A/B）和核面积（C）都在决定低门槛细胞的绝对数量**，存在一个临界量——数量够了就集体点火、不够就点不着。这是放电版独有、率均值场看不见的有限尺寸现象。

**诚实边界（全程不变）**：全部 **screen-plus / 单工作点 / 6–8 网络 / 非正式裁决**；gap-limit 没放宽（mean 锁 16–18、std 锁 1.5/0.5、r cap 0.6）；matched-null 写"在这个工作点没复现 Rich、不证否"；B 侧 tail-driven 仍独立不并链。**没有把"screen"爬级成"formal verdict"或"parameter law v2"。**

**建议下一步（给 user 选，未做）**：① 正式裁决需"连接/阈值场/噪声"三类种子分离 + 更多网络；② forward-LFP/HFO 读出层（才敢对应真实 SEEG）；③ B 的弱刺激尾巴招募 regime 可单独深挖（刺激强度 × 参差的二维）；④ C 的 r=0.3 边界没在 spec 范围内钉死（要钉需 mean<16 = 动 gap-limit，框架级决定）；⑤ 率/F-I 配平控制去 Jensen 残留。
