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
