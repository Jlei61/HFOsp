# Topic5 A 线加固结果 — 间期轴是持续网络 scaffold，不是发作早期特异招募（2026-06-15）

> **白话摘要（先读这一段）。** 我们看一件事:每个病人**平时**(没发作时)那些高频小事件,在脑内各触点的
> 发放先后顺序连成一根"传播主轴";发作时各触点"烧得多旺"也有个空间高低。问题是这两张图对不对得上。
> 这一轮把这个结论狠狠加固了五道,结论收窄成一句更稳、更诚实的话:
>
> **平时那根传播主轴是病人身上高度可复现的稳定结构(把事件砍一半搭轴、去预测另一半的发放顺序,中位相关
> 0.76、18 个病人全部置信区间为正);它读出一根与发作激活相关的"粗网络骨架"。但这根粗骨架是持续存在的
> ——在发作前 90–120 秒同样强(甚至更强),所以它不是"发作早期才特异冒出来"的招募现象,而是一直都在的
> 网络底座。** 最细那层(快活动 60–100Hz 同时控住电极杆和活跃度)有信号但在统计边界上,只能当敏感性提示,
> 不当主结论。
>
> 一句可安全写进论文的话:**间期传播轴是患者内稳定结构,并能读出与发作激活相关的粗网络骨架。**
> 暂不能写:该对齐是发作早期特异出现的现象。

- 上游:`axis_alignment_AB_result_2026-06-14.md`(A+B 首轮);加固计划
  `docs/superpowers/plans/2026-06-15-topic5-axis-alignment-AB-hardening.md`
- 队列:Epilepsiae 主队列 18 被试 / 350 合格发作(v2 缓存,见 §5 attrition);Yuquan 仅描述(1 被试)
- 主统计量:`|corr_pair_mirror_invariant(间期轴场, 发作激活场)|`(镜像不变、符号自由),**保持锁定
  P-current**(见 §6 abs-max 诊断)

---

## 1. 五项加固 — 逐项结果

### 3.1 腿(i) 轴自洽 / 反套套性 —— ✅ 干净通过
held-out ρ:用一半事件搭轴、预测另一半的发放顺序。复用 `split_half_axis_validation`(masked 输入,随机
split + bootstrap),从现有 `results/topic4_sef_hfo/skeleton_geometry/per_subject/` 提取 18 人。
- **中位 held-out ρ = 0.761**(门 ≥0.6);**18/18 病人 ρ 的 CI 下界 > 0**(门 ≥12/18)。
- 结论:间期轴不是套套逻辑,是高度可复现的患者特异结构。
- 工件:`results/topic5_ictal_recruitment/axis_alignment/legi_axis_selfconsistency.json`

### 3.1 腿(ii) 对齐稳健(砍两半重搭轴再对齐)—— ✅ 粗层稳
半轴产出器 `run_contact_plane_readout.py --event-split {first_second,odd_even}` → 72 个 t_a 半轴
(4 组合 ×18,**0 退化**)→ 各自重跑对齐(`run_topic5_axis_splithalf.sh`,B=1000)。
- **broadband × channel(粗骨架,主):四个半(两砍法 ×两半)每一个都赢过全随机洗牌**——赢过随机的病人
  数 binomial 全 <0.05(0.00017 / 2e-05 / 0.00155 / 0.01087),Wilcoxon 全 <0.05(0.0014 / 0.037 /
  0.007 / 0.024),"真值−随机"差 0.06–0.11、CI 基本都 >0。跨被试两半对齐强度一致:Spearman ρ
  first_second=0.54、odd_even=0.94(门 ≥0.5)。
- **HFA × joint(最严层):** 两半对齐强度高度一致(ρ 0.70 / **0.998**)、Wilcoxon 四个半全显著
  (0.0006–0.011),但赢过随机的病人数低(13 个有效里 1–3)、效应量 CI 在奇偶砍法触 0、且随砍法变。
  → 真、可复现,但最细层是边界(与主队列同调,见 §2)。
- 工件:`.../axis_alignment/splithalf/splithalf_summary.json`

### 3.2 窗口敏感性 + 远端负对照(ictal-specificity)—— ⚠️ 负对照未掉下去 → scaffold
v2 轨迹缓存切 6 窗(0-5 / 5-10 / 0-10 / 0-20 / 近端[-10,0] / **远端[-120,-90] 负对照**),
broadband+HFA × B=1000(`run_topic5_window_sweep.sh` + `aggregate_topic5_windows.py`)。
- **逐被试配对 ictal-specificity 检验(发作 0-10 窗 vs 远端 pre 窗的原始 |corr| 差):**
  - broadband:post−distal 中位 = **−0.026**(负!),Wilcoxon(post>distal) p=0.367 → **不特异**。
  - HFA:post−distal 中位 = +0.056,p=0.162 → **不特异**(微弱非显著趋势)。
- **窗口表(broadband × channel):远端 [-120,-90] 最强(8/18,eff 0.111)> 发作 0-10(6/18,eff 0.091)。**
- HFA × channel 整体最强(各窗 eff 0.12–0.19、CI>0),但远端同样强(9/18,eff 0.117 CI>0)→ 也不特异。
- 结论:**粗对齐是持续 scaffold,不是发作早期特异。** verdict 字段
  `persistent_scaffold_NOT_ictal_specific`。
- 工件:`.../axis_alignment/window/window_summary.{json,md}`

### 3.3 频率特异性 —— 分层锁定(措辞)
- **broadband = robust activation burden(主指标)**:稳过粗层 channel(主队列 FDR q=0.020)。
- **HFA 60–100Hz = mechanistic sensitivity**:整体信号最强(channel 层 eff 0.12–0.19),最严 joint 层
  靠秩检验显著但效应量 CI 边界(Bartolomei 2008 EI 的 fast-activity 生理依据)。**不抢主结论。**
- ramp / EI = secondary,不进窗口/砍两半扫描。

### 3.4 方向符号(1D 沿轴)—— 补充,不作判据
runner 旁挂 `sign(corr(typical_rank, 激活))`(与 y 镜像无关的一维沿轴量;主统计 |alignment| 不变)。
每病人同号比例 / signed median / 符号熵 已记录。**明确不作成败判据**(主统计方向已合并;eLife 2022 +
上一轮 echo 都提示轴本来双向)。

### 3.5 病人级统计 —— effect-size + CI + adequate 分母
`aggregate_topic5_axis_alignment.py` 加 patient-level effect-size(median real−null)+ bootstrap CI +
rank-biserial。**主统计单位 = 18 病人;350 次发作 = 病人内精度,不作独立 N。**
- **❗adequate 分母修正(审阅 P1-1)**:joint 层只有 13 个有效病人(其余 5 个该层随机对照退化、几乎没真
  打乱)。effect-size CI 现按 adequate-only(`effective_shuffle_n>=4`)算,并输出 `effect_n_all/
  effect_n_adequate` + `wilcoxon_p_adequate`。

---

## 2. 主队列定稿口径（FINAL，adequate-fixed）

| 指标×层 | n_pass | Wilcox(adeq) | FDR q | effect(adeq) | 95% CI | 判读 |
|---|---|---|---|---|---|---|
| broadband × channel(主) | 5–6/18 | 0.008 | 0.020 | 0.087 | **[0.006, 0.129]** | 粗骨架**稳**(CI>0) |
| broadband × within_shaft | 4–5/18 | 0.049 | 0.066 | — | — | 边界 |
| broadband × joint(最严) | 0/18 | 0.75 | 0.75 | — | — | 不过 |
| HFA × joint(最严) | 3/18(adeq13) | 0.013 | 0.029 | 0.050 | **[−0.003, 0.099]** | 秩检验显著、**效应量 CI 触 0** = 敏感性,非主 |

定稿表:`results/topic5_ictal_recruitment/axis_alignment/axis_alignment_FINAL.{json,md}`。

---

## 3. 允许 / 禁止措辞

**允许写:**
- 间期传播轴是患者内高度可复现的稳定结构(held-out ρ 0.76,18/18 CI>0)。
- 它读出一根与发作激活相关的**粗网络骨架**(broadband 粗层 FDR q=0.020,砍两半两砍法都稳)。
- 这个粗骨架**持续存在**(发作前 90–120 秒同样强),是网络底座、非发作早期特异招募。
- 更细 / 更快(HFA)的对齐有信号但在统计边界,作敏感性。

**禁止写:**
- ❌ 该对齐是发作早期特异出现的现象 / ictal-early-specific recruitment。
- ❌ 发作沿平时顺序逐触点重放(主统计是符号自由的轴/梯度共线,非 replay)。
- ❌ HFA 干净碾过最严 null(效应量 CI 触 0,只能作敏感性)。

---

## 4. 文献锚

- Smith/Schevon/Rolston eLife 2022:人类 IED 可作 traveling waves,与 ictal 共享路径、方向关系不必同向
  → 支持"间期传播结构有意义",不要求逐触点重放。
- Matarrese Brain 2023:interictal spike propagation 揭示 effective connectivity、预测手术结局 → 支持
  "持续网络结构"框架。
- Bartolomei Brain 2008(EI):fast activity × 相对 onset delay → HFA 在更严层更强有生理合理性。

---

## 5. Provenance / 输入来源（防以后追不回）

- 主队列对齐:`axis_alignment_{broadband,hfa,ramp,ei}_B{1000,2000}.json`;cache=`t0_feature_cache`,
  axis=`propagation_geometry/observation_readout/real_subjects`(masked 模板,P-current 主统计)。
- 砍两半:axis=`real_subjects_{mode}_half{N}`(符号链接自 `real_subjects_splithalf`);cache 同主队列(0-10s)。
- 窗口:cache=`t0_feature_cache_v2_windows`(`--store-bb-zt --pre-feature-window 130`),切片→
  `window_caches/<window>`;axis 同主队列。
- **v2 缓存 attrition**:`v2_cache_attrition.csv` —— 预期 351 → 缓存 350,**丢 1**
  (epilepsiae_1077 sz5,load skip);远端 [-120,-90] 窗 353 全覆盖。病人单位仍 18。
- **P2 待补(归档前)**:主 alignment JSON 未记录 cache_dir/axis_dir;本 doc §5 暂代 provenance,后续给
  runner 加 provenance 字段。

---

## 6. abs-max 诊断(为什么主统计保持 P-current）

审阅曾建议把主统计从 `abs(max_signed)` 收紧到 `max(|c_id|,|c_mir|)`。只读诊断(354 发作)显示两者
36–52% 发散,大差来自横向(y)nuisance 结构 → abs-max 会把横向当成对齐、**系统性高估**。用户拍板:**主统计
保持 P-current(FINAL 不动),符号读出改 1D 沿轴。** A1 `corr_pair_mirror_invariant_signed` 降为诊断工具。

---

## 7. 交接 / 下一步

- ✅ 五项加固执行完;主结论收窄为**稳定 scaffold readout**(数据确证非 ictal-specific)。
- ⏳ 三张论文级图(patient-level / null-hierarchy / window-sensitivity,后者承载 scaffold 结论)+
  figures/README + 主文档 §3.0 保守口径 + memory 更新。
- 这阶段结束 = topic5 主线可冻结为"间期轴 = 患者特异稳定网络 scaffold 读出"。

---

## 8. 自查与 caveat（2026-06-15 对抗式复核）

- **3.4 符号补充数值**:18 人病人内同号比例中位 **0.75**(方向中等稳定),队列 signed median 8 正 / 10 负
  = **无系统方向**(与轴本来双向一致)。仅描述,不作判据。
- **远端负对照的 pre-ictal 有效性(caveat + 辩护)**:远端窗 [-120,-90] 相对临床起始;缓存元数据未存逐发作
  EEG-vs-临床起始偏移,故无法逐发作排除"EEG 起始早于临床 90s 以上时远端窗被早发作污染"。**但更强的辩护成立:
  近端 [-10,0] 窗(更靠近发作)broadband eff=0.057 < 远端 0.111**——若对齐是发作驱动,越靠近发作该越强,实际相反
  → 既反 ictal-specificity、又反"远端被 EEG 起始污染"(污染会让近端更强)。scaffold 结论对该 caveat 稳健。
- **broadband 砍两半的边界点(诚实)**:跨被试两半对齐强度 first_second ρ=**0.54**(刚过 0.5 门)、odd_even
  ρ=0.94;odd_even h2 效应量 CI 触 0。结论是"粗层稳但 first_second 跨半一致性是边界"——按 split-half OR
  odd-even 取 OR,粗层稳成立。
- **无重大过度声称**:主结论(稳定 scaffold readout)、HFA 敏感性档、轴自洽,均与数据和上游 hfa_joint 复验
  一致;未发现需回退的 claim。
