# Topic 5 Stage 2b — Early-Ictal Dynamic Pattern Echo, Sentinel Gate (NOT PASSED)

> **状态**：sentinel 人工门结果，**exploratory**。`B=500`（sentinel preview，**不是**最终 cohort-级 inference）。**未进 cohort**，**不支持**"发作早期动态稳定复演特定间期传播通道"的主张。
> **日期**：2026-06-12。分支 `topic5-ictal-recruitment-stage2`。
> **谱系**：Bridge Q1/Q1′ → Stage 1 echo gate（ER 代理粗锚）→ Stage 2 first-onset recruitment（量错，失败）→ **Stage 2b dynamic-pattern echo（本文）**。
> **代码**：`src/topic5_dynamic_echo.py`（纯数学，18 tests）+ `scripts/run_topic5_dynamic_echo.py`（sentinel）+ `scripts/plot_topic5_dynamic_echo.py`。图 + README：`results/topic5_dynamic_echo/sentinel/figures/`。spec `docs/superpowers/specs/2026-06-11-topic5-stage2b-dynamic-pattern-echo-design.md` (v2.1)、plan `docs/superpowers/plans/2026-06-11-topic5-stage2b-dynamic-pattern-echo.md`。

---

## 摘要（第一性原理）

**测了什么** — 一次癫痫发作的最初十秒里，间期高频传播"模板"靠前（=源头）的那些触点，是不是更早 / 更强 / 更快地被点亮。换句话说：发作早期被招募的动态次序，是否回响间期已经看到的那条传播通道。三个哨兵发作：Epilepsiae 1146 的 sz2、sz5，Yuquan litengsheng 的 sz0（间期模板各覆盖 15 个触点）。

**怎么测的** — 每个时刻给一个"模板对齐分数"（正 = 源头触点此刻更强），连成一条随时间走的对齐曲线；取曲线峰值，再跟"把触点身份随机打乱、同样取峰"比（500 次），看真实峰能不能赢过随机峰——取峰本身是在时间上挑选，所以零分布也每次重算整条曲线再取峰，把这个挑选吸收掉。三种打乱：①完全随机；②只在同一根电极杆内打乱（控解剖杆）；③只在"总体活跃度相近"的触点间打乱（控"源头本来就更活跃"）。另外单独看一个**不挑时间点**的诚实量：早窗 0–5s 的平均对齐。

**揭示了什么** — **门没过，但不是纯噪声，而是"粗锚"的样子。** 挑时间峰确实赢过完全随机打乱（三发作都显著），所以早发作动态里**确实有**和模板相关的结构。但它不是一个干净的早期正回响：峰落的时间不一致（两个发作落在偏晚的 8–9s，只有一个落在早窗 1.2s）；不挑时间点的诚实量逐发作**变号**（−0.34 / +0.31 / +0.08），没有稳定正向；Yuquan 在"同杆内打乱"下就塌掉（说明那点对齐是整根电极杆/解剖层面的粗锚，不是杆内精细路径）。结论：早发作动态里有模板相关结构，但主要是**共享的粗解剖/杆级优先级**、且逐发作不稳，**不是稳定的具体通道路径复演**——和 Stage 1 的 ER 代理"粗锚"结论一致。

---

## 预注册门 + 裁决

**门（spec §4/§8，plan Phase-2 GATE）**：进 cohort 仅当 epi(CAR) 和 yuquan(bipolar) **都**满足——

| 条件 | 要求 | 实测 | 判 |
|---|---|---|---|
| (a) 早期同向正峰 | echo(t) 峰在 `[0,~6]s`、同向为正 | 峰时 8.6 / 1.2 / 9.2s（2/3 偏晚）；诚实早窗均值变号（−0.34/+0.31/+0.08） | **FAIL** |
| (b) 过随机 max-null | echo_peak 显著 vs channel max-over-time null | p = 0.002 / 0.010 / 0.008 | PASS |
| (c) 过杆/锚之一 | within-shaft **或** anchor-matched 不全消 | epi 两者都过；yuquan 仅 anchor 过（shaft p=0.321 不过） | PASS（技术上） |

**裁决 = 「稳定锚为主 / 没看清」，不是「站住·动态 echo 含路径」。(a) 这条载重条件没满足 → 按合同不进 cohort、不写路径复演主张。**

---

## 哨兵逐发作数值（B=500，primary 窗 `[0,+10]s` 临床锚）

broadband activation echo（主特征），`echo_peak = max_t align_score(t)`；confirmatory = broadband `echo_mean[0,5]`（无挑选）：

| seizure | swap | echo_peak | t_peak | p_channel | p_shaft | p_anchor | **echo_mean[0,5]** | wide[-5,15] | cluster1 | ER(held) |
|---|---|---|---|---|---|---|---|---|---|---|
| epi 1146 sz2 | strict | +0.911 | **8.6s** | 0.002 | 0.004 | 0.002 | **−0.344** | −0.345 | +0.024 | +0.180 |
| epi 1146 sz5 | strict | +0.804 | **1.2s** | 0.010 | 0.028 | 0.006 | **+0.312** | +0.327 | −0.136 | −0.103 |
| yuquan sz0 | none | +0.814 | **9.2s** | 0.008 | **0.321** | 0.018 | **+0.079** | +0.081 | −0.040 | −0.129 |

要点：
- **channel max-null 三发作全显著** → 非空结构，但只说明"有模板相关性"，不说明是早期路径。
- **confirmatory 早窗方向变号** → 不挑时间点时没有稳定正向；wide 窗几乎同值 → **不是 onset 标注偏差**造成的。
- **yuquan within-shaft 塌掉** → 该侧是杆/解剖粗锚，非杆内精细路径。
- **anchor-matched 三发作仍显著** → 也不只是"源头本来更活跃"（控总体强度后顺序结构仍在），但出现在偏晚、方向逐发作飘。
- **sz5 是三发作里唯一"理想早期正回响"形态**（峰 1.2s、早窗 +0.31、三 null 全过）——**仅作"存在这种形态但不稳定复现"的例子，不升级为机制/cohort 主张**。

---

## 锁定结论（archive 口径）

```text
Stage 2b sentinel did not pass the dynamic-path echo gate (B=500, n=3 sentinel seizures).
All three sentinel seizures exceeded the channel-shuffle max-over-time null, showing
non-null template-related structure in early ictal dynamics.
However, the confirmatory early-window direction was not stable, peak timing was often late,
and the Yuquan case failed the within-shaft null.
Therefore the current evidence supports a shared coarse anatomical/channel-priority anchor,
not stable replay of a specific interictal propagation path during early seizure dynamics.
```

---

## 方法合同（已实现，nulls / 符号 / 窗）

- **仪器**：每 seizure 从 Stage-2 cache 读 raw feature traces（无 EDF 重读）→ robust-z（复用 `stage2._z_from_traces`，detrend=`rolling_median`）→ 各特征插值到同一 0.1s 时间网格（per-feature `t_center` 差半窗）→ activation/slope echo 曲线、上升斜率潜伏期、早爬升强度。
- **符号锁（§2.1）**：`align_score>0` 永远 = 模板靠前触点更早/更强/更快（intensity=−Spearman、latency=+Spearman）；sign-flip 回归测试拦截。
- **max-over-time null（§2.2）**：每 draw 重算整条 echo(t) 取 max_t；channel（全打乱）/ within_shaft（`parse_shaft` 块内）/ anchor_matched（per-channel 平均活跃度四分位块内）。块-null 缺 blocks 硬 raise（不静默退化成 channel null）。
- **窗合同（pre-registered，不 post-hoc 加宽）**：primary `[0,+10]s` 临床锚承载判据；sensitivity `[-5,+15]s` 临床/EEG 锚 + confirmatory `echo_mean[0,5]` 只报 robustness，不救 primary。本轮 wide 窗与 primary 同号同值 → 排除 onset 偏差。
- **per-dataset montage（P0）**：yuquan ictal=bipolar→alias-left（语义 `bipolar_aliased_left`）；epi=car。`assert_channel_identity` 用 montage 语义比对（非 raw reference 字符串），cache 实证匹配。
- **ER held-out**：报告但不投票（construct-validity 旁证）。

---

## 决策与下一步（user-locked 2026-06-12）

1. **接受 sentinel gate 失败**（本文）：不进 cohort，不支持动态路径复演主张，sz5 不作主结论。
2. **可选的描述性小扩展（非路径复演验证）**：若要队列数字，仅 build-cache 到 n≈8–10，报告"理想早期正回响出现频率"与"粗锚型显著出现频率"，定位为 **case-series 描述**，不改变本裁决，不叫 cohort inference。
3. **不继续迭代仪器**：失败是结构性的（峰时散 + 早窗方向变号 + yuquan within-shaft 塌掉 + wide 窗排除 onset 偏差），换 detrend/特征会变成追结果。

**caveat**：n=3 哨兵只够下门、不够队列结论；`B=500` 是 sentinel preview 非最终 cohort 精确 p；sz5 是例子非结论。
