# PR-6-A：Interictal Template Semantic Alignment via Seizure Onset Propagation

先说一下文档定位：下面这份是对标 `docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md` 的合同级计划，落盘后应放在 `docs/archive/topic1/pr6a_template_ictal_alignment_plan_2026-04-xx.md`。Topic 1 主文档 §7 增加 PR-6-A 条目；§7.9 的 KONWAC placeholder 已剥离为「未来模型层（不绑 PR 编号）」。

> 本计划占用 PR-6A 编号；PR-6 编号空间从此对应 PR-6A/B/C/D/E 数据发现序列；KONWAC v2 已剥离到主文档 §7.9 未来模型层，不绑 PR 编号。

---

## 1. 科学问题与判断边界

### 1.1 核心问题

在 PR-2 / PR-2.5 已锁定的 `30/30` stable interictal templates 上，是否存在至少一个 template，其 channel rank geometry 在统计上复现 ictal onset 的 channel-wise earliest activation rank？

对 9 个 `inter-cluster r < -0.5` 的 forward/reverse subject，双 template 是否呈现 Smith 2022 双向行波的"一个同向、另一个反向"signature？

### 1.2 这个 PR **不**回答

- 模板是否随发作邻近发生几何变形 → PR-4C 已封板为 null
- 模板招募率是否在 post-ictal 升高 → PR-5-B 已 Bonferroni-pass
- SOZ 通道的慢调制优势 → Topic 3 PR-1/PR-2 的独立线

### 1.3 已发表先验（决定了假设的 prior strength）

Smith et al. 2022 eLife: "the majority of IEDs are traveling waves, traversing the same path as ictal discharges during seizures, and with a fixed direction relative to seizure propagation... were bidirectional, with one predominant and a second, less frequent antipodal direction" — MEA μm-scale 的双向行波直接预测我们 k=2 + 9/30 forward/reverse 的几何。

Korzeniewska et al. 2014: "In patients with focal ictal onsets, the patterns of propagation recorded during pre-ictal... and interictal... intervals were very similar to those recorded during seizures" — SdDTF 在 HFO band 展示 interictal-ictal propagation 高度重合。这是 H1 最强的方法论先验。

Liu et al. 2019: "Regions exhibited highly stable preferences to appear upstream, intermediate, or downstream in spike propagation sequences... DP-Stability was 0.88 ± 0.07" — 他们用 Degree Preference (Spearman) 衡量 interictal spike propagation 跨 30-min segment 的稳定性，方法学与我们 r_sz 跨 seizure 的稳定性检验同构。

Bailey et al. 2021: "Spike onset and seizure onset seemed to be distinct networks in most cases" — **负面文献**，显式挑战 H1。把 H1 cohort null 的可能性留在桌上，不预设方向。

Aung et al. 2026 Epilepsia: "Spectral similarity between Sp-HFOs and ictal HFA increases as seizure onset approaches, indicating dynamic preictal evolution" — 最新证据：HFO 在 interictal → ictal 是**渐进式**转变，而不是两态切换。支持"interictal template 里的 HFO 应与 ictal HFA 共用 generator"。

---

## 2. 数据合同

### 2.1 Cohort 定义（严格分层）

**主 cohort（Epilepsiae only，main config）**：

- 硬前置：属于 PR-5-A retained main cohort（n=23）
- 必须有 ≥3 个 seizure onset annotation 可用（Epilepsiae 的 `seizure_onset` 列）
- 必须 k=2（`stable_k = 2`）— 确保 T0/T1 二元对比语义清晰
- 预期 cohort 大小：n ≈ 15–18（PR-5-A main 23 中扣掉 k=4/k=6 的 2 个 + 少 seizure 的 3–5 个）

**辅 cohort（sensitivity）**：

- Yuquan PR-5-A retained (n≈15)：所有 subject 都走同样 pipeline 但不纳入 cohort-level inferential。作为**replication hint**，不参与 α 分配。
- 理由：Yuquan 录制更短、seizure 数普遍 <5、onset annotation 质量我们之前已知较差。强行 pool 会污染 main 的判读。

**Case-series 单独报告**：

- k=4 subject（`818`）
- k=6 subject（`zhangjinhan`）
- 9 个 forward/reverse subject 的子集（和主 cohort 重叠），单独跑 H1'

### 2.2 通道对齐合同

关键约束：interictal template 只在 `n_participating ≥ min_part`（当前 `min_part = 5`）的通道子集上有 rank；ictal onset rank 原则上在所有 recording channels 上可测。

- 定义 `C_interictal`：subject 的 PR-2 template 定义所用通道集合
- 定义 `C_ictal_reachable`：ictal onset pipeline 成功提取 onset time 的通道集合（见 §3）
- **对齐集合 `C_common = C_interictal ∩ C_ictal_reachable`**
- 硬门槛：`|C_common| ≥ 5`，否则该 subject 退出 main cohort（case-series 另报）
- 所有 template centroid rank 和 r_sz 都**在 C_common 上重新 re-rank**（保持 rank 连续性），不是在原始集合上截断

---

## 3. Ictal Onset Rank 提取 Pipeline（核心方法学）

这是你关心的核心问题："如何定义+提取 channel-wise earliest activation，不被传播带来的大范围同步高信号干扰"。答案是 **4 层防御**，每层针对一种污染源。

### 3.1 Layer 1 — Feature choice: Energy Ratio (ER)

**为什么不用 amplitude crossing**：Seizure 的大 amplitude 是 propagation 后的事，amplitude crossing 会被晚起但大振幅的通道率先触发。

**为什么用 ER**："The Page-Hinkley algorithm provides a detection time Nd for each brain structure if involved in the generation of a rapid discharge" — ER 对频谱结构变化敏感（fast oscillation replacing slower background），不是对 amplitude 本身。"EI was developed by Bartolomei et al. to quantify the appearance of high frequency oscillations by averaging the energy ratio between high and low frequency bands over the lag in seizure onset time relative to a reference point"。

**具体定义**（相对 Bartolomei 2008 调参以适应我们的 HFO-focused 数据）：

- 滤波：bandpass 4–250 Hz（Epilepsiae 主流 fs 1024 Hz 可支持）
- `E_fast[n] = ∫_{60}^{100} |S(f,n)|² df`（gamma band burst）
- `E_slow[n] = ∫_{4}^{20} |S(f,n)|² df`（theta/alpha/low-beta 背景）
- `ER[n] = log(E_fast[n] / E_slow[n])`（log 稳定化，避免 slow band 低谷时 ratio 爆炸）
- 滑窗 1 s，步长 100 ms

**为什么是 60–100 Hz 而不是 Bartolomei 原版 12–127 Hz**：我们整个 pipeline 是 HFO-centered（80–250 Hz ripple/fast ripple 是 detection 焦点），LVFA 期望在 gamma 带最早出现。12–127 Hz 过宽会把 alpha/low-beta 的 ictal rhythmic 活动一起吸收，这些在 widespread propagation 期才明显，污染 onset rank。注意 gamma 带 60–100 不包括我们的 HFO 带 80–250 的主体，故意避开以防 HFO detector 的影响被双重计入。

### 3.2 Layer 2 — Per-channel baseline z-score（抗 identity-bias）

**污染源**：有些通道 HFO 基线就高（Topic 1 §3.2 的 identity-bias 86% 在 seizure 里同样存在）。ER 绝对值大的通道会先触发绝对阈值，把它误判为 onset。

**解决**：每个 channel 独立 z-score normalize：

- `baseline window = [-300s, -60s]` 相对 seizure onset
- 排除窗内所有 "known IED peaks"（用 Epilepsiae 已有 spike annotation 或从 HFO detection 反推）
- 若某通道 baseline < 60s valid → 该 seizure 该通道退出（不参与 rank）
- `z_ER[n] = (ER[n] − μ_bl) / σ_bl`
- 对所有通道 baseline-normalize 后才进入下一步

这层是原始 EI 没做的，对我们尤其重要——因为我们已知 identity-bias 大。

### 3.3 Layer 3 — Page-Hinkley CUSUM 找 change-point

**为什么不用阈值穿越**：阈值穿越对噪声敏感、对渐进 onset 不稳健。"The key idea of the Page-Hinkley algorithm is to perform a test on the mean of ER statistic by building a quantity (cumulative sum U)... Decision that a significant change has occurred is taken at alarm time"。CUSUM 累积统计量，对早期缓慢增长敏感，并且一旦 detect 就给出**第一次偏离 baseline 的时点**（而不是 peak 的时点），这正好对应 onset。

**合同**：

- 对每个通道的 `z_ER[n]` 跑 Page-Hinkley CUSUM：
  - `U[n] = max(0, U[n-1] + z_ER[n] − bias)`
  - `bias = 0.5`（Bartolomei 2008 默认）
  - `M[n] = max_{k≤n} U[k]`；alarm 时刻 `n_d` = 第一个 `n` 满足 `M[n] − U[n] ≥ λ`
  - `λ` 阈值通过 per-subject permutation 定：在 baseline window 上滚动跑 CUSUM，取 `False positive rate < 1/hour` 的 λ
- detection window = `[onset_clinical − 5s, onset_clinical + 30s]`
  - 前 5s 容忍 clinical annotation 的不精确
  - 后 30s 覆盖大多数 onset pattern 的 recruitment 窗；>30s 通常是 propagation 阶段，不是 onset

### 3.4 Layer 4 — Tie-handling 与 "unresolved onset" 标记

**污染源**：部分 subject 的 seizure 会在 <500 ms 内招募 >60% 通道（rapid-spread pattern）。如果强行 ranking 会得到一堆 tied ranks，噪声占主导。

**合同**：

- 对 `n_d` 做 ranking（最小 `n_d` = rank 0）
- **Fractional rank for ties**：`Δt < 50 ms` 的通道赋予平均 rank
- 若某次 seizure 中 `>60%` 通道在 `[clinical_onset, clinical_onset + 1s]` 内被 detect → 该 seizure 标记 **`onset_tied`**：
  - 保留在 sensitivity analysis
  - 不参与主 r_sz 中位数计算
- 若某次 seizure 中 `<30%` 通道有 valid `n_d`（多数 channel "unreached"）→ 该 seizure 标记 **`onset_unreached`**，完全排除
- Subject 至少 3 个 non-tied non-unreached seizures 才 qualifies for main cohort

### 3.5 Per-subject r_sz 构造与 stability gate

- 对 qualifying subject，每次 seizure 得到一个 per-channel rank vector `r_{sz,s}`（s 为 seizure index）
- 主 r_sz = **median rank per channel across seizures**
- **Stability gate**：计算所有 seizure pair (s, s') 的 Spearman ρ on r：
  - `s_sz = median{ρ(r_{sz,s}, r_{sz,s'})}` over all pairs
  - `s_sz < 0.3` → subject 退出 main cohort（"onset rank 本身都不稳，template 对比无意义"）
  - `0.3 ≤ s_sz < 0.5` → 参与 cohort 但在敏感性分析中降权
  - `s_sz ≥ 0.5` → 全权重

这个门槛直接对标 Liu 2019 的 "DP-Stability was 0.88 ± 0.07"——他们的 stability 在 interictal spike propagation 上是 0.88；我们允许 ictal 场景降到 0.3 / 0.5 是因为 ictal seizure 数通常远少于 interictal segment，统计上噪声更大。

---

## 4. 假设与统计合同

### 4.1 主假设 H1（Smith 2022 一般预测）

- **定义**：对每个 qualifying subject（k=2, s_sz ≥ 0.5），计算：
  - `ρ_T0_sz = Spearman(T0_centroid_rank, r_sz)` on C_common
  - `ρ_T1_sz = Spearman(T1_centroid_rank, r_sz)` on C_common
  - `ρ_max = sign-preserving argmax by |ρ|`；取该值为 `ρ_aligned`
- **Cohort-level test**：
  - 主检验：one-sample Wilcoxon on `ρ_aligned` against 0（H1 预测 ρ_aligned 中位数 > 0）
  - 次检验：sign test（n subjects with ρ_aligned > 0.2 / total qualifying）
  - 报告 `|ρ_aligned|` 中位数和 IQR
- **α 分配**：
  - Main cohort Bonferroni：α = 0.05 / 2 tests = 0.025
  - "PASS" 判据：Wilcoxon p < 0.025 且 sign-test 超过 70% 正向

### 4.2 备择假设 H1'（Smith 2022 双向预测，9-subject subset）

在 `inter-cluster r < -0.5` 的 9 个 subject 子集上（与 main cohort 重叠的那些）：

- 两 template 应与 r_sz 呈符号相反的相关
- 定义 **bidirectional score**：`B = −(ρ_T0_sz × ρ_T1_sz)`（若符号相反且绝对值都大，B 大）
- H1' 预测 B > 0 在大多数 9-subset subject 上成立
- Sign test：`# subjects with ρ_T0_sz × ρ_T1_sz < 0` / 9
- 报告 individual `(ρ_T0, ρ_T1)` 散点图

这是 PR-6-A 里**最直接对应 Smith 2022 的检验**。9 个 subject 功效极有限（sign test min p ≈ 0.004），所以这一层从一开始就声明为 **exploratory**，不分 α。

### 4.3 Sanity check（pipeline 可信度 gate）

**必须先通过这一关才能解读 H1/H1'**。用 Epilepsiae `focus_rel` 三值标注（`i` = focal / `l` = lesion / `e` = extra-focal）：

- Per-subject：`i` 通道的 r_sz rank 分布 vs `e` 通道的 r_sz rank 分布 → Mann-Whitney U
- Cohort-level：`i` 通道 median r_sz vs `e` 通道 median r_sz 的 paired Wilcoxon
- **PASS 判据**：cohort Wilcoxon p < 0.05 且方向是 `i < e`（focal channel rank 更小 = 更早激活）
- **如果 sanity check FAIL**：pipeline 有 bug 或 SOZ label 有问题，**H1/H1' 结果不发布**，回到 §3 调参（λ, bandpass, baseline window 三选一）

### 4.4 失败合同

| 触发条件 | 响应 |
|---|---|
| <50% PR-5-A retained main subject 能通过 §3.5 stability gate | cohort underpowered → 降级为 case-series (n≥5)，H1 作为 descriptive only |
| Sanity check (§4.3) cohort null | pipeline 问题或 label 问题；**暂缓 H1/H1' 发布** |
| H1 Wilcoxon p > 0.025 且 `|ρ_aligned|` 中位数 < 0.25 | Smith 2022 在 HFO population event 层面不复现；诚实公开 null |
| H1 PASS 但 H1' 9-subset null | H1' 视作 underpowered exploration，不作为主结论一部分 |
| `onset_tied` seizure 比例 > 40%（跨 cohort） | Epilepsiae 多数 seizure 是 rapid-spread 模式；rank extraction 时间分辨率限制；在 `onset_tied` vs `onset_resolved` 两层上各跑一次，分别报告 |
| H1 和 sanity check 同时 PASS 但 H1' 显示 `ρ_T0` 和 `ρ_T1` 同号且都 > 0 | 两 template 都同向对应 ictal onset，inter-cluster 负相关是在非-onset-related 维度上发生；Smith bidirectional hypothesis 不成立，但"模板锚 ictal"成立 → 这是一种有趣的 intermediate outcome |

---

## 5. 代码入口（新建文件）

```
src/ictal_onset_extraction.py        # §3 pipeline
    ├─ compute_er(signal, fs, fast_band, slow_band, win)
    ├─ baseline_zscore_er(er, baseline_window)
    ├─ page_hinkley_cusum(z_er, bias, threshold)
    ├─ extract_onset_rank_per_seizure(subject, seizure_idx) → (n_d dict, tied_flag, unreached_flag)
    └─ build_rsz_per_subject(subject) → (r_sz_median, s_sz_stability, qualifying_flag)

src/template_ictal_alignment.py      # §4 统计
    ├─ compute_rho_template_vs_rsz(template_centroid, r_sz, c_common)
    ├─ h1_cohort_test(rho_aligned_list) → (wilcoxon_p, sign_p, median_abs)
    ├─ h1prime_bidirectional_test(subset_9) → (B_list, sign_p)
    └─ sanity_check_focus_rel(r_sz, focus_rel_labels) → (mw_p_per_subject, cohort_wilcoxon_p)

scripts/run_pr6a_onset_rank.py       # 一次性批跑全 cohort
scripts/run_pr6a_alignment_stats.py  # 输出 archive 数值
scripts/plot_pr6a_alignment.py       # cohort 散点 + 9-subset individual + sanity panel
```

---

## 6. TDD 测试合同（锁 9 项，主文档不允许在未跑过前松动）

```
tests/test_pr6a_ictal_onset.py

T1. test_er_channel_independence:
    构造 2-channel synthetic signal，channel 0 在 onset+1s 注入 80Hz burst，
    channel 1 在 onset+3s 注入同等 burst。验证 n_d[ch0] < n_d[ch1].

T2. test_baseline_zscore_removes_identity_bias:
    2 channels，ch0 baseline ER 始终比 ch1 高 5 dB，但 seizure 时两者都升高
    同等幅度。未做 z-score 时 ch0 n_d < ch1 n_d（错误）；做 z-score 后应
    两者 n_d 接近.

T3. test_page_hinkley_no_false_alarm_on_pure_baseline:
    仅 baseline segment，CUSUM λ 按 §3.3 permutation null 设定后，
    false positive rate < 1/hour.

T4. test_onset_tied_flagging:
    synthetic seizure, 80% channels 在 onset+0.2s 同时 recruit.
    pipeline 必须标记 onset_tied=True.

T5. test_c_common_rebalancing:
    C_interictal 和 C_ictal_reachable 部分重合 (|common| = 7);
    template centroid 被正确 re-rank 到 7-channel space;
    r_sz 同样 re-rank 到 7-channel space.

T6. test_stability_gate:
    构造 subject with 5 seizures: 3 一致的 onset rank + 2 随机 rank.
    s_sz 应落在 [0.3, 0.5] 区间，subject 被降权而非剔除.

T7. test_h1prime_bidirectional:
    synthetic subject with ρ_T0_sz = +0.7, ρ_T1_sz = −0.6
    → B = 0.42 > 0, 计入 "bidirectional supported" 样本.

T8. test_sanity_check_gate_blocks_h1_release:
    模拟 sanity check fail 情景（focal vs extra-focal rank 无差异），
    统计函数应返回 h1_released=False 标记.

T9. test_k_neq_2_routed_to_caseseries:
    k=4 subject 走 case-series path，不进入 main H1 test.
```

---

## 7. 与 PR-5 / PR-4C / Topic 3 的边界

- **PR-5-A retained cohort 是硬前置**：不重新做 novel-template gate；所有通道、template、rate 定义继承 PR-5 合同。
- **不动 PR-4C 的封板结论**：PR-4C 检验的是"模板内部几何随发作邻近变形"（cohort null），PR-6-A 检验的是"模板几何与 ictal onset geometry 的跨时态一致性"。两者是不同问题，PR-4C null 不预测 PR-6-A 结果。
- **不进入 Topic 3 PR-6-C core source/sink 工作**：PR-6-C 是从 identity-bias 出发挑选核心通道；PR-6-A 是验证已定义 template 是否锚 ictal。两者并行、结论独立可叠加（"锚 ictal 的 template 的 core source 是否也是 SOZ" 是 PR-6-A × PR-6-C 的交叉 bonus，不纳入任一方的主结论）。

---

## 8. 预期时间线（按 2026-04-21 起算）

| Week | 交付 |
|---|---|
| 1 | `src/ictal_onset_extraction.py` 完成 T1-T3；在 1-2 个 pilot subject 上手动验证 onset rank 提取合理 |
| 2 | §3.3 λ 阈值校定 + §3.4 tie 门槛 tune；cohort 级 sanity check pass 为硬 milestone |
| 3 | `src/template_ictal_alignment.py` + cohort run + archive 中间报告 `pr6a_onset_extraction_validation_2026-04-xx.md` |
| 4 | H1/H1' 全 cohort 跑通 + sensitivity analyses + 验收文档 |

验收条件：sanity check PASS + 所有 9 个 TDD test 通过 + archive plan 同步更新 + 主文档 §2/§4/§7 一句话结论更新。

---

## 9. 几个你可能追问的技术点，我先写答案

**Q1: 为什么不直接用 clinician 标的 SOZ 通道作为 "ictal onset channel set"，跳过 pipeline？**

因为 SOZ 是**二值标签**，不给 rank。我们要的是 per-channel timing rank，Smith 2022 / Korzeniewska 2014 的预测也是关于 ordering，不是关于 set membership。SOZ 用在 §4.3 sanity check 足够了。

**Q2: 为什么不用连续 `π` (PR-6-B) embedding 上的 template alignment，而要用离散 centroid rank？**

两个原因：(a) π embedding 还没做 TDD；PR-6-A 必须能在 PR-6-B 结果出来前独立运行；(b) centroid rank 是我们已经信任的量（PR-2.5 strong/moderate 全 cohort 覆盖），改成 π 会引入新的不确定性来源，妨碍归因。PR-6-B 完成后可以重跑 PR-6-A 做 robustness check。

**Q3: 如果某个 subject 的 template 是 k=2 但 baseline rate 把它驱成 `dom_frac > 0.85`（一个模板极其稀少），第二个 template 的 centroid rank 是否可信？**

不可信到需要门槛：`min events per cluster ≥ 200` 才能定义稳定 centroid。PR-2 本身有这个门槛但 PR-6-A 要显式重检。若某 subject T1 events < 200 → H1' 退出（9-subset 会损失 subject），但 H1 仍可用较大 cluster 的 centroid 做 ρ_aligned。

**Q4: seizure onset annotation 本身有 ±几秒的不准确，这会多大程度影响结果？**

§3.3 的 detection window `[onset−5s, onset+30s]` 就是为此而设。ER CUSUM 的 alarm time `n_d` 是相对 detection window 起点的，不依赖 annotation 精准到秒。Annotation 误差只会影响 baseline window 是否被 ictal 活动污染——我们 baseline 窗在 `[−300s, −60s]`，60s buffer 已覆盖绝大多数 annotation 误差。

**Q5: Epilepsiae 部分 subject 是 ECoG，seizure 有 DC shifts；gamma ER 会不会漏掉这些？**

"SEEG signature of LVFA was found to be largely observed (~70%) and the other onset patterns like burst-suppression and delta brush patterns were very rare"，即便非 LVFA onset 也有 20–30%。针对这类情况，sensitivity 分析里我们额外跑一个 **broader-band ER** (`E_fast = 12-127 Hz` 的 Bartolomei 原版)，专门救那些 gamma band 失败的 seizure。若 broader-band 拯救了 >10% subject 的 stability，我们在正式结果里并列报告两版结果。

---

需要我接着把这份 plan 转成正式的 archive markdown 文件（含完整 §1–§9 + numbered section anchors 与 Topic 1 主文档的 back-links），还是你想先对几个设计点拍板再定稿？重点你可以现在决定的几件事：

1. ER 的 fast band 固定 60–100 Hz，还是并列 broader-band 12–127 Hz 两版？
2. §3.5 的 stability gate 阈值 0.3 / 0.5 是否合理？（我从 Liu 2019 的 0.88 往下降了大步，因为 ictal seizure 数远少于他们的 interictal segment 数）
3. §4.3 sanity check 用 `focus_rel` 三值还是二值（`i` vs non-`i`）？三值更精细但 cohort 里 `l` 通道数往往少。
4. H1' 的 9-subset 是否声明完全 exploratory、不分 α？