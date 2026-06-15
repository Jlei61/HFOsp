# Topic 5 后续计划：从"路线重放"转向"网络骨架 / 状态门控"

日期：2026-06-13　状态：plan（待用户审核）　性质：exploratory，多线并行

---

## 一、先把现在的结论钉死

**测了什么** —— 病人不发作的平时，脑子里一直在放很短的高频小放电；我们之前发现这些小放电不是同时冒出来，而是像接力一样沿一条相对固定的路线一站一站传过去（有"起点站""终点站"）。我们想知道：真正的大发作来的时候，头几秒里各个触点被点着的先后，是不是还沿着平时那条老路线走一遍。

**怎么测的** —— 用平时那条路线给每个触点一个"排队号"，然后看发作头几秒里号小的触点是不是真的更早 / 更强 / 更快被点亮；判断办法是跟"把触点身份随机打乱"比，真信号必须明显赢过随机。我们前后换了三种测法（粗代理量 → 真"谁第一"仪器 → 整条点亮形状）。

**揭示了什么** —— 三种方法殊途同归：平时那条**精细**的传播路线，在发作最初几秒**并没有被稳定、特异地重新走一遍**。能看到的只是一个更粗的倾向——病灶附近、同一块解剖区域的东西在发作早期普遍更早、更活跃。一句话定位：不是"完全没关系"，也不是"那条路被精确重放"，而是"**有一个粗的解剖锚，但没有精细路径的复演**"。

**文献背书这个转向是对的，而不是项目失败**：

- Smith 等 (eLife 2022) 在**微电极致密阵列**上发现间期放电是双向行波、和发作放电走**同一条路**、相对发作扩散方向有固定关系——但这是局部皮层 mm 尺度，**不能直接平移到 SEEG 深部触点跨脑区的 HFO 排名重放**。我们能在更粗的 SEEG 尺度上看到任何轴对应，本身就是把那个发现往外推一格的新结果。
- 发作起始形态本来就很杂（Brain 2014 报告 7 类），有些起始形态还会出现在扩散区——所以"压一个统一的 first-onset 检测器"先天承重不足。这正好解释了我们 Stage 2"谁第一"仪器为什么测不稳：不是参数没调好，是这个量本身在发作早期不适合承重。
- 发作传播高度异质、且随昼夜和更慢尺度漂移（Proix Nat Commun 2018；Schroeder PNAS 2020 / Brain Comm 2023）：同一个病人发作起点可以相对稳定，但传播图谱会变。所以"每次发作都沿同一条间期模板前向重放"这个强假设，先验上就站不住。

结论：继续调 0–5 s 前向回响、继续优化 first-onset 检测器，是在一个结构性阴性上追阳性，**停**。

---

## 二、间期和发作"哲学上的共性"是什么（含对核心假设的一个诚实修正）

### 2.1 一个必须先说清的诚实修正

用户的核心直觉是"间期事件的多次 replay **促进 / 塑造 / 推动**了状态向发作发展"。这里要分两层：

- **结构层（强、有数据、可检验）**：间期 HFO 和发作走的是**同一个致痫网络**，因为两者都受同一套病理连接（effective connectivity）约束。这一层文献高度一致，也是我们粗锚阳性所支持的。
- **因果驱动层（弱、有争议、不能当默认）**："每一次间期事件是否在因果上把系统往发作推"——文献本身是**分裂的**。de Curtis / Avoli 等的实验工作明确指出：间期放电既可能**促发**（pro-ictogenic）也可能**抑制**（anti-ictogenic）发作；海马 CA3 约 1 Hz 的间期活动在体外是**抗发作**的。另一条线（"发作前是网络韧性 resilience 逐渐丧失"）把间期放电看成**扰动**——它把系统从间期动力学上推开一点，但真正决定走不走向发作的是缓慢的韧性下降，不是单次放电本身。

所以：**"间期 replay 促进发作"不能写成默认前提**。可检验、且经得起高水平审稿的共性，应该定在**结构层**——而把"因果驱动"降一档，只作为一条**低先验、诚实标注**的探索线（下面的 D 线）。

### 2.2 共性的一句话定义（建议作为下一阶段主线措辞）

> **间期 HFO 刻板时序 = 致痫网络骨架的一次次读出；发作 = 同一副骨架在特定状态、特定入口条件下的快速招募。二者的共性不是"每个触点按同一顺序重放"，而是"入口、传播轴、早期参与区、状态门控"四者一致。**

文献锚点：
- Smith 2022 "echoing"——间期和发作共享路径与方向（轴层）。
- Brain 2023（Matarrese / Tamilia）——间期 spike 传播图就是一张 effective-connectivity 图，分 onset / early-spread / late-spread 三区，切掉 onset 区能预测术后结局（分区 / 入口层 + 临床价值）。
- source-sink connectivity / neural fragility——间期数据的价值在**定位未被处理的病理网络节点**、预测复发，而不是预测发作那一瞬（临床层）。
- Schroeder——发作通路受慢状态门控（状态层）。

这套框架顺手把整个故事的论点摆正了：**间期数据的价值不在"预告发作哪一秒来"，而在"标记并定位那副病理骨架"**——这恰恰是我们那个阴性结果真正支持的论点。

---

## 三、P0 停止项（不再投入）

1. early 0–5 s 前向回响调参（结构性阴性：峰偏晚 + 诚实数变号 + 一个病人针内塌掉）；
2. first-onset recruitment 检测器（"谁第一"在发作早期测不稳，是量错不是参数错）；
3. 单一模板预测所有发作；
4. pool 所有发作的 replay 检验。

---

## 四、一个共享"仪器"先搭好（A / B / C 的共同前置）

阴性结果里有一件东西是**好的、可复用的**：Stage 2 那套特征曲线本身（line-length / 各频段功率 / spectral-edge 随时间走的曲线）测得没问题——测不稳的只是"从曲线上抠出一个唯一的 onset 时刻"。所以：

**前置任务 T0：把"发作早期激活图"缓存做成队列级（先过 eligibility audit，不预设全 inventory 可用）。** 复用 `src/topic5_ictal_recruitment.py` 的 trace 计算，但**丢掉 first-onset 检测器**，改成对每个触点、每次发作存一张 0–10 s 的激活描述：

- 宽带 AUC、HFA(60–100 Hz) AUC（同时为 B 线的 EI-like 准备）、line-length AUC；
- 爬升斜率 ramp slope；
- 5-bin 能量轨迹（给主成分 PC1 用）。

**T0 = axis-linked ictal eligibility audit 已跑完（2026-06-13，`scripts/run_topic5_t0_eligibility.py`，自适应 pre=max(120, min(|eeg_rel|,300)+120)）**。**口径要点（避免分母误导）**：这**不是全发作库 audit**——它只清点"已有可用间期传播轴记录"的被试的发作。三层分母都报（写进 `_summary.json`）：
- **全发作库 592**（Epilepsiae 540 + Yuquan 52）；
- **axis-linked attempted 408**（被 A 线"有无间期轴"先过滤过的）；
- **analysis_eligible 354（占 attempted 87%）**，覆盖 **19 个被试**（cohort 25，其中 6 个 Yuquan 因无发作清单贡献 0 行）。

按数据集：Epilepsiae **351 合格 / 18-of-18 被试**（充足）；Yuquan **3 合格 / 仅 1 个被试（zhangkexuan）**。不合格原因全是正当的：前一次发作 <300s（23）+ 窗口越过记录段开头/结尾（pre 21 + post）+ 坏时间戳/块边缘读取错误（7，已逐条记到 `load_error_rows.csv`，确认是数据边缘非代码 bug）+ 区间不完整（3）。**T0 检查的是什么、不是什么（不夸大）**：查了 ictal 特征可读 + 干净基线（脑电起始-aware，≥30s）+ 前一次发作间隔 ≥300s + 完整 EEG 区间；**没有**强制 day/night-crossing 或窗口中段 recording-gap（`day_night` 仅作记录列）。结果文件 `t0_eligibility_audit.csv` + `_summary.json` + `load_error_rows.csv`。
> **关键覆盖发现 — A 线实际是 Epilepsiae-only**：Yuquan 的"有传播轴记录"被试 {chengshuai, liyouran, songzishuo, zhangbichen, zhangjiaqi, zhangkexuan, zhaochenxi} 和"有发作 ictal 数据"被试 {gaolan, huanghanwen, litengsheng, pengzihang, sunyuanxin, xuxinyi, zhangjinhan, zhangkexuan, zhaojinrui} **几乎不相交（只有 zhangkexuan 同时具备）**。两个可补的缺口：(a) 8 个有 ictal 的被试缺坐标文件 `chnXyzDict.npy` → 没有轴记录；(b) 6 个有轴记录的被试不在发作清单里（发作未抽取 / 或本就没有）。**A/B/C 现在就能在 Epilepsiae（18 被试 / 354 次发作）上跑，不被 Yuquan 卡；Yuquan 进 A 线是单独的覆盖修复任务，非阻塞。临床收口 E 用消融覆盖表（独立，不受影响）。**
> **证据等级锁（A 线 cohort）**：**Epilepsiae = primary cohort（18 被试）；Yuquan = descriptive sensitivity only**（当前仅 1 个被试 / 3 次发作，只能做病例描述 / 流程 sanity，**不并入跨数据集"稳定"主张**）。等 Yuquan 覆盖救援把可分析被试拉上来，再考虑是否升级。

现状（旧）：只有 2 个哨兵被试缓存了（`results/topic5_ictal_recruitment/sentinel_cache/`）。**inventory 数（Epilepsiae 542 次、Yuquan 52 次发作）不等于可用数**——T0 要读原始发作信号、做 montage、baseline、通道对齐、质控，实际进缓存的会少于 inventory。所以 **T0 第一步不是直接缓存，而是先做一份 eligibility audit**，对每次发作逐条核：

- 原始 ictal 信号可读；
- onset / offset 标注完整；
- baseline 窗够长（够做 robust-z）；
- montage 能对齐到该被试的 template 通道；
- 60–100 Hz 没超 Nyquist（采样率够）；
- 不跨 seizure / postictal / gap / day-night 边界。

audit **报三个数：attempted / cacheable / analysis_eligible**，不写"全队列可用"。缓存只对 `analysis_eligible` 的发作做。**A / B / C 三条线都吃这张缓存**——这是唯一的串行依赖，先做。

---

## 五、五条可并行探索线

每条线都标了：主问题（白话）／ 用什么现成数据 ／ 怎么测 + 对照 ／ 预注册的门 ／ 诚实先验。**证据等级：A = 唯一 primary；B / C / D = secondary / exploratory；E = 临床收口（被 outcome 标签 gating）。**

### A 线（主检验，预注册）：跨杆、符号自由的"传播轴"对齐

> ✅ **已执行（2026-06-14）。结果 + handoff：`docs/archive/topic5/axis_alignment_AB_result_2026-06-14.md`。**
> 18 Epilepsiae 队列：粗"共享网络主轴"稳（broadband 稳赢全通道 null，FDR q=0.020 / LOSO p=0.015）；
> 比杆/活跃度更细的对齐只在快活动（hfa）上稳（过最严 joint，q=0.029），主指标 broadband 止于粗层。
> 符号自由共线，非逐点重放。下文为原始设计，数值口径以归档为准。

**主问题** —— 把平时那条路线的"排队号"看成一根**方向轴**（不是严格顺序）。问：这根轴的方向，和发作头十秒"哪里更早更强被点亮"的空间梯度，**对不对得上**——而且要在**跨电极杆**（不同针之间）这个层面对，因为同一根针上的触点天然共线，"针内顺序"和"电极几何"在数学上分不开（这正是 yuquan 一做针内打乱就塌掉的原因）。

**大量复用 topic3↔4 已建好的几何读出**（这是 A 线最大的省力点，不重造轮子）—— 间期那一侧的"轴"和"二维触点平面"已经在 `results/spatial_modulation/propagation_geometry/` 做完并验收过：
- 间期传播轴本身已算好（`src/propagation_skeleton_geometry.py::compute_axis_frame` 给源→汇方向；`split_half_axis_validation` 已用一半事件搭轴、另一半验证，留出半数 Spearman ρ 中位 0.752、强轴 16/26）——A 线的"间期轴"直接拿这个，不用重算。
- 把发作激活图压到同一张标准化二维触点平面、再算"两个场像不像"的整套管线也现成：`src/propagation_contact_plane_readout.py` 的 `build_readout_record`（标准化 record）、`smooth_field`（连续场）、**`corr_pair_mirror_invariant`（镜像不变 = 已经是符号自由的场相似度，正好是 A 线主统计量）**、`compare_model_to_cohort` + `placement_in_distribution`（把"一张外来场对真实场的相似度"落在"真实彼此相似度"分布里报百分位——这套 real-vs-model 的描述性 placement 方法，A 线原样换成 ictal-激活场 vs 间期-轴场即可）。
- 可视化照搬：`observation_readout/figures/static_maps`（52 张真实二维触点图）的画法直接复用，A 线只是把"模型场"那一层换成"发作激活场"叠上去。

**用什么数据** —— masked 间期模板 / 已建好的间期轴（`results/spatial_modulation/propagation_geometry/`，path-axis 26 被试 + `results/interictal_propagation_masked/per_subject/` 40 被试）+ T0 激活缓存 + 坐标 / 杆标签（`src/seeg_coord_loader.py`、`src/propagation_skeleton_geometry.py::parse_shaft`，已就绪）。

**怎么测 + 对照** —— 每个病人算一根轴坐标 `z_i`（归一化排队号），和发作激活量 `A_i`（0–10 s 宽带 / HFA AUC、ramp、或 5-bin PC1）做**镜像不变 / 轴向（axial）的对齐度**（方向取模 π，允许反向，符号自由；直接用 `corr_pair_mirror_invariant`）。三层对照：① 全通道打乱（粗解剖锚）；② 同一根针内部打乱（保守 null，控解剖共线）；③ 只在"本来一样活跃 / 离病灶一样远"的触点间打乱（控源头活跃度）。

**预注册的门**（避免"调对齐量"变成新的"调检测器参数"）——
- **唯一主 endpoint（primary）**：每个病人的**对齐度绝对值 `|axis_alignment|`（镜像不变，方向已合并）是否赢过预注册 null**。注意 null 必须用**同一个镜像不变 / 轴向统计**去算（不是普通相关的 null），这样"取绝对值会抬高假阳"的问题被对照本身抵消掉。
- 门：跨杆轴**只赢全通道 null** = 粗解剖轴（和现在结论一致，仍可作为"宏观网络轴"发表）；**还赢针内 + 活跃度 null** = 精细传播轴。
- **方向符号稳定性是 secondary（异质性 / 敏感性读出），不是 primary**：每个病人这根轴的符号在他多次发作间稳不稳，只用来描述"这个病人是单向还是双向轴"，**不当主判据**。理由很硬：(a) primary 用的就是镜像不变 `|·|`，方向已经合并，再拿符号当 primary 是对不同对象下结论，自相矛盾；(b) 我们自己的几何结果里双模板轴 7/10 接近**同轴反向**（median cos = −0.977），Smith 2022 和上一轮 echo(t) 诚实数变号也都指向**轴本来就是双向**——强行要求符号稳定，会把真实的双向轴误判成"失败"。

**诚实先验** —— SEEG 跨杆采样稀疏，信号可能本来就弱；这是 SEEG 深部触点相对 Smith 致密微阵列的**结构性局限**，不是失败。能看到任何跨杆轴对应都是把 Smith 往粗尺度外推的新结果。

**A-line implementation contract（执行接口——复用"函数 + IO schema"，不是"复用思想"再重写）** —— 这是为了防止"名义复用、实际重拼"重新引入通道顺序 / 坐标空间 / mirror-null / support-mask 的 bug。固定如下：

```
输入：
  间期场（现成）：results/spatial_modulation/propagation_geometry/observation_readout/
                  real_subjects/<ds>_<subj>_t_<a|b>.json
                  —— usable record 判据用 SCHEMA 条件，不靠 status=='ok'（52 个 usable
                   record 多数根本没有 status 字段）：channels[] 非空 AND n_channels ≥ 6
                   AND flags.low_contact_count == false；status 仅用于排除
                   'descriptive_only' / 'error:...'，不要求 status=='ok'。
                  （当前 52 个 usable record / 26 subject；8 descriptive_only、9 yuquan
                   因 coord 文件缺失被排除 → A 线跨杆侧目前以 Epilepsiae 为主，
                   yuquan 仅 chengshuai 等少数有坐标，是真实覆盖约束，不掩盖）
                  每 record 已带 channels[]（name / x_norm / y_norm / typical_rank /
                  support / is_soz）+ norm_scale_mm + scalars。
  发作激活（T0 产出）：每触点每发作的 broadband_auc_0_10s（主激活量，见下）。

步骤（全部调现成函数，禁止重写）：
  1. 读间期 record（上面的 JSON）。它就是"间期轴场"的来源——轴/平面已建好，
     不重算 compute_axis_frame。
  2. 构造"发作场 record"：复制间期 record 的 channels（保留同一套 x_norm/y_norm/
     support = 同一根轴、同一平面），把每个 channel 的 typical_rank 值
     替换成该触点的 broadband_auc_0_10s。
     —— 关键：T0 激活按 channel NAME（first_contact alias）join 到 record channels，
        解析不到的 record channel 直接丢弃，**绝不按下标对齐**（这是通道顺序 bug 的根源）。
        eligibility 口径见下（≥80% record channel 能解析 + 解析后仍 ≥ MIN_CH=6），
        否则该 seizure 不进 A（由 T0 eligibility 标记）。
  3. 两个 record 各过 R_smooth_rank(rec, X, Y, sigma_xy, s_thresh)
     （= smooth_field，scalar='rank'）得到 (T, S)。X,Y = make_plane_grid()。
  4. 主统计量 = | corr_pair_mirror_invariant(T_间期, S_间期, T_发作, S_发作)['corr'] |
     （镜像不变去掉横向 nuisance；取绝对值 = 沿轴 forward/reverse 符号自由）。
  5. null：把 T0 激活值在通道间打乱后，重走 2–4（同一个 corr_pair_mirror_invariant
     + 同一个 |·|），B≥1000。三层 null 各自的打乱域见上（全通道 / 针内 / 活跃度匹配）。
  6. cohort 汇总沿用 placement / subject_first_fold 的折叠纪律（swap 被试 t_a/t_b
     同 subject 不重复计数）。swap 被试 A-primary 只用 primary 模板（t_a）；
     max-over-templates 留给 C 线，不在 A-primary 里。
```

**A 线 primary 数字门（T0 后、跑 A 前冻结，不留事后自由度）** ——
- **主激活量：只有一个 primary = `broadband_auc_0_10s`**（最稳健、不预设频段、且与 B 的 HFA-based EI 解耦，保持 A/B 互相独立）。`HFA_auc_0_10s` 与 `ramp_slope_0_10s` 是**预注册 secondary**（fast-activity 版本是 on-thesis 的确认层，但不当 primary）。*若改用 HFA 作 primary，只需把这两行对调——但本计划默认 broadband。*
- **per-seizure 统计**：上面第 4 步的 `|corr_pair_mirror_invariant|`。
- **per-subject 聚合**：median across 该 subject 的 seizures。
- **null**：同一镜像不变统计，B ≥ 1000；三层（全通道 / 针内 / 活跃度匹配）。
- **per-subject 通过门**：real `|corr|` > 该 subject 全通道 null 的 95th percentile（先赢全通道 = 粗解剖轴；再看针内 + 活跃度 null 是否仍赢 = 精细轴）。
- **cohort 通过门**：通过的 subject 数对 5% 期望做 binomial，**且** (real − null_median) 跨 subject 做 Wilcoxon——**两者都提前写死**，结果出来不挑。
- **A-line montage join 口径（eligibility 分母）**：分母 = **readout record 的 channels[]**（A 实际 join 的就是它，不是 subject 全 template channel set）。要求 **≥80% 的 record channel 能在该发作的 ictal montage 里解析到**，且解析后仍 **≥ MIN_CH = 6** 个匹配通道。（B / C 若另用 full template，各自单独定义分母，不套用这条。）baseline 与 T0 一致 = [−90s, −60s]（≥30s 干净，guard=[−60s,0]）。

### B 线（secondary / exploratory，不作 primary 队列主张）：EI-like"早期参与梯度"对齐

> ✅ **已执行（2026-06-14），并入 A 线同一套轴对齐仪器作为 `ei` 激活量。** 结果：赢全通道 / 同杆 / 活跃度
> 三层、不赢最严 joint（FDR q：channel 0.009 / within_shaft 0.037 / anchor 0.029 / joint 0.055）。
> 仍是 secondary——不得当 primary cohort claim。详见 `axis_alignment_AB_result_2026-06-14.md` §5。

**证据等级**（先钉死，避免事后挑主终点）—— **B 是 secondary / exploratory 的机制读出，不作 primary cohort claim**。primary 只有 A 一条。若 A 阴性、B 阳性，**不能**写成 primary claim，只能写"次级 / 提示性"。（除非将来明确改成 A+B co-primary 并提前做 alpha 分配——本计划暂不这么定。）

**主问题** —— 不去抠"谁第一"那个不稳的毫秒延迟，改成问：间期模板的**早号触点 / 入口小群**，是不是落在发作的"早期参与度高地"上。"早期参与度"用经典 SEEG 致痫性指标 EI 的思路：**快活动强 + 出现早**——序数量、对"头几秒分不出先后"鲁棒，正好绕开 Stage 2 的不稳。

**从 per-seizure 起步，先调研再继续** —— B 上一轮探索过、但做的是 per-seizure，本轮**仍从 per-seizure 起步**（不先做 per-subject 聚合）。落地前先做一轮既有工作 / 文献调研（尤其 **Yuquan 侧的提取口径**：Yuquan 走 bipolar、Epilepsiae 走 CAR，montage 不同，EI-like 的能量比定义要先核对一致再跑），再继续。

**EI-like 公式写死（方向不能反）** —— 延迟是**惩罚项**（早→高分），不是奖励项；绝不能写成 `快活动比 × 原始延迟`（那会让晚出现的触点反而高分，结论直接反号）。锁定 Bartolomei 2008 的形式 `EI = 能量比 / 延迟`：

```
EI_like_i = ER_i / (Δ_i + τ)
  ER_i  = 触点 i 在 [δ_i, δ_i + H] 窗内 (60–100 Hz 功率 / 低频参考带功率) 的均值
          低频参考带 = 4–40 Hz（与 60–100 不重叠）；功率用 T0 缓存的 band-power trace
  δ_i   = 触点 i 的快活动起始时刻 = 在 60–100 Hz 能量比 trace 上做 CUSUM / Page-Hinkley 变点检测
          （CUSUM 已在 src/topic5_ictal_recruitment.py）
  Δ_i   = δ_i − δ_0，δ_0 = 全部"被招募"触点里最早的 δ（参考时刻 N0）
  τ, H  = 审计期锁死的常数（首版 τ = 1 s、H = 1 s；写进 spec，本轮不调）
  baseline = 发作前 [−90s, −60s]（落在 [−60s, 0] guard 之前，≥30s 干净基线；
             不足 30s → 该 seizure 不进 B；与 T0 eligibility 的 baseline 定义一致）
             做 per-channel robust-z，再形成能量比。sensitivity = [−120s, −60s]，不救 primary。
             —— 绝不再用 [−60s, −5s]（那段落在 guard 里，把 preictal/peri-onset 变化混进 baseline，
                会把 EI 信号自己归一化掉）
  无 onset 规则：CUSUM 在 [0, T] 内检不到变点的触点记 δ_i = NaN、Δ_i = 不定，
                 EI_like_i = 0（不进"早参与"集合），不静默补值
```

**怎么测** —— 算每触点每发作的 EI_like，做"模板早号是否富集在 EI_like 高分位"，或"轴坐标 `z_i` 是否解释 EI_like 梯度"（Spearman / 序相关）。对照同 A 线三层。

**为什么能和 A 并行** —— A 用原始激活 AUC / ramp，B 用 EI_like 组合量；**同一套模板输入、不同发作目标量**，互为交叉验证（但 B 是 secondary）。

**诚实先验** —— EI_like 是序数、领域标准、对早期不可测鲁棒——技术上最稳，但证据等级是 secondary。

### C 线（病人内 modifier，exploratory，不做队列主张）：子型条件下的入口 / 方向对齐

**主问题** —— 同一个病人**不同发作子型**，是不是对应间期模板的**不同入口小群 / 不同方向**（正向模板 vs 反向模板 vs 不同入口群）。这直接接住"每个人内部发作种类不同"的直觉。

**用什么数据** —— z-ER 发作子型标签（`results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject/*__zer_binned.json`，25 被试，含 `subtype_label` / `n_subtypes`，gamma_ER + broad_ER 两个频段）。

**关键纪律**（这是和报告2 的主要分歧，按更严格的一份办）——
- 子型标签本身**不稳**（PR-1 自评：exploratory、未过 sensitivity、~70% 命中），很多病人发作次数少（Yuquan 中位 6、最少 2）。**把子型当主轴去切，per-子型-per-病人的 power 几乎为零**。
- 所以子型只能是**病人内的 modifier**：用一个分层 / 混合模型，**同时**估"有没有共享轴"和"不同子型是不是不同轴"，而不是拆成两个互相稀释 power 的分析。
- `subtype_size < 3` 只做 case-series，不做 cohort claim。**整条 C 线标 exploratory。**

### D 线（状态层，低先验，单独并行）：发作前 motif 漂移

**主问题** —— 发作前，间期 HFO 事件的"模板组成 / 入口分布 / 方向 / 覆盖范围"会不会**朝即将到来的那次发作的几何漂移**。这是唯一直接碰"动态驱动"的线——但按 2.1 的诚实修正，**先验低**（如果共性本质是结构性的静态骨架，漂移就是个更弱的赌注）。

**怎么测** —— 分窗：baseline `[-12h,-6h]`、preictal-30 `[-30,-10]min`、preictal-10 `[-10,-1]min`，**排除 `[-60s,0]`**（避免发作标注和早期变化污染）。只保留 5 个指标：模板出现率、事件离主模板的距离、入口熵、正反向比例、空间覆盖范围。病人内 paired / permutation。

**措辞红线** —— 这是 **state drift（状态读出）**，**不是 seizure prediction**。**即使阴性也有价值**：说明刻板间期事件不是短时程 preictal biomarker（和"发作通路受慢状态门控"一致）。

### E 线（临床收口，关键路径——数据采集现在就启动，分析等标签）：**Yuquan-only** 触点级结局层

**主问题** —— 间期 HFO 序列的 **hub / 起点 / 轴**，是不是一个**没被切 / 没被消融、且和复发相关**的节点。这是整个故事**唯一有真正临床新意**的断言，也对应 source-sink / fragility 那条最强的间期线（间期数据 = 定位未处理病理网络）。

**cohort 边界先划清（避免把"有结局"误当"有触点级治疗覆盖"）** —— 这个问题要同时有两样东西：**触点级 resection/ablation 覆盖** + **结局标签**。只有结局、没有触点级覆盖，做不了"hub 是否未切除"。所以拆成两条，不混进同一套 schema：

- **E1（主分析，本轮就做）= Yuquan-only clinical capstone**：Yuquan 已有触点级覆盖（`results/template_ablation_coverage/yuquan_coverage_prep.csv`，20 被试，`n_ablated` / `source_ablated_frac` / `template_not_ablated`），只缺结局标签——补上结局就能跑"hub 未切除 ∧ 复发相关"。结局 `source` 固定 = `yuquan_doc`。
- **E2（先 feasibility，不进主分析）= Epilepsiae feasibility audit**：单独先查 Epilepsiae 到底**有没有触点级 resection/ablation 覆盖**。先验上很可能没有——既往机制-paper 工作已判定 **Epilepsiae 临床层是 region-level（非触点级）**，那样就只有 region 粒度、做不了触点级 hub 检验。**没有触点级覆盖 → Epilepsiae 不进 E 主分析**，最多做 region 级的弱版本，且单独立项、不与 Yuquan capstone 混口径。

**结局标签来源（仅 E1 / Yuquan）** —— Yuquan 病例 doc：**病人几年后再回来 = 复发**，从随访记录推。（Epilepsiae 的 SQL 即便有结局字段，也要先过 E2 触点级覆盖这一关才谈得上进分析。）

**先冻结 outcome 表结构，再开查（不要边查边改字段）** —— 采集前先把 schema 定死，之后只填值、不动字段：

```
outcome schema（per subject，冻结）：
  subject              被试 ID
  surgery_type         切除 / 消融 / SEEG-only（无手术）
  engel_class          Engel I–IV（拿不到就留空，不改字段名）
  ilae_class           ILAE 1–6（与 Engel 二选一或都填）
  followup_months      随访月数（从手术到最后一次随访 / 复发）
  recurrence           bool（是否复发）
  recurrence_date      复发日期（无则空）
  source               'yuquan_doc'（E1 capstone 固定 Yuquan；Epilepsiae 先过 E2 feasibility 再说）
  notes                自由文本（口径例外、推断依据）
```

**为什么现在就启动** —— 它是最高价值断言的**限速步**，且是人查病历、不和 A/B/C 抢计算资源。**采集（填上面这张冻结表）现在并行启动**，分析（"模板 hub 是否落在未切除且复发相关组织"）等标签到位再跑。两份报告把它放 P4 是低估了——它该是**关键路径**，只是分析端被数据 gating 住。

---

## 六、纪律：先预注册一个 primary，其余全标 exploratory

最大的风险**不是测错量，是分叉路径**（garden of forking paths）。报告2 一口气提了 4 个新量 × 3 层 null × 子型分层 × 双频段——自由度爆炸。我们上一轮 replay 阴性之所以可信，靠的就是"预先锁定的门 + 一个诚实数 + max-over-time null"这套纪律。pivot 之后这套纪律**只能更严**。

**证据等级（先钉死）**：**A 是唯一 primary（跨杆 `|axis_alignment|` 赢预注册 null）；B / C / D 都是 secondary 或 exploratory；E 是临床收口（被 outcome 标签 gating）**。

**硬规定**：
1. **A 线（跨杆轴 `|axis_alignment|` 镜像不变对齐度赢 null）是唯一预注册 primary**，一个 per-病人 endpoint、一个固定门，**碰子型和漂移之前先把它锁死**。方向符号稳定性是 A 的 secondary 读出（描述单向 / 双向），不是 primary。
2. B / C / D **全部显式标 secondary / exploratory**，结果只能写"提示性"，不能写队列主张。
3. 每条线落地前，把**判读门写成数字**（赢哪层 null、`|axis_alignment|` 的百分位阈值），并配一个"坏数据应当失败"的回归对照——别让重复 / 退化 regime 把门 false-PASS（参照本仓库既有教训：验收门必须编码结论本身，不能只数个数）。

---

## 七、最小路线 / 并行时序

```
现在并行启动：
  T0  发作早期激活缓存（先 eligibility audit）  [前置，A/B/C 共吃]   —— 计算
  E1-采集  Yuquan 复发标签查病历（先冻结 schema） [关键路径限速步]      —— 人 / 临床，不抢计算
  E2-feasibility  Epilepsiae 有无触点级 resection 覆盖（先验=region-level，很可能不进主分析）

T0（先 eligibility audit，再缓存 analysis_eligible）完成后并行：
  A 线  跨杆 |axis_alignment| 镜像不变对齐（唯一 primary，预注册） ←─┐
  B 线  EI-like 早期参与梯度（secondary，per-seizure 起步）       ←─┤ 同吃 T0 缓存，互为交叉验证
  C 线  子型条件 modifier（exploratory）                          ←─┘

单独并行（低先验）：
  D 线  发作前 motif 漂移（state drift，不写预测）

E1 标签到位后：
  E1 线（Yuquan-only）触点 hub vs 未切除 / 复发（临床收口）
```

- **先跑 A+B**：现有数据最够、科学问题最干净。
- C 作为 A/B 模型里的 modifier 一起估，不单拆。
- D 单独并行，cheap，看一眼即可。
- E1 的**采集**现在就启动（限速步），分析放标签到位之后；E2 只做 feasibility，不进主分析。

---

## 八、还不知道的 / 边界

- A 线跨杆信号可能本来就弱（SEEG 结构性局限），弱 ≠ 失败；门已经允许"只到粗解剖轴"这个落点。
- 子型标签不稳，C 线只能 modifier + case-series，不升队列主张。
- D 线先验低；即便阴性也是有价值的负结果。
- E1 被 outcome 标签 gating，没标签前不能跑分析，也不能口头预告结论；E 主分析是 **Yuquan-only**（Epilepsiae 先验 region-level，触点级 hub 检验做不了，只能 E2 feasibility / region 弱版本）。
- A 线跨杆侧目前以 Epilepsiae 为主（yuquan 多数被试在这棵树里缺坐标，只有少数有）——这是真实覆盖约束，不是分析选择。
- 因果驱动层（"间期是否真把系统推向发作"）文献有争议（可能抗发作），本计划**不**把它当默认前提，只在 D 线低先验地碰一下。

---

（内部归档代号：Topic 5 ictal-template-echo 谱系 = Bridge Q1/Q1′ → Stage 1 ER-proxy echo gate → Stage 2 first-onset recruitment → Stage 2b dynamic-pattern echo（全部 negative，归档见 `docs/archive/topic5/dynamic_echo/stage2b_sentinel_2026-06-12.md`）。本计划新量：A（唯一 primary）= sign-free / mirror-invariant `|axis_alignment|` vs null（复用 `corr_pair_mirror_invariant` + `compare_model_to_cohort` + `placement_in_distribution`；间期轴复用 `compute_axis_frame` / `split_half_axis_validation`），per-patient sign-stability 降为 secondary heterogeneity readout；B（secondary）= EI-like (Bartolomei 2008) = `ER_i / (Δ_i + τ)`，delay 在分母 = 惩罚（NOT `× delay`），per-seizure 起步；C=z-ER subtype_label (`*__zer_binned.json`) as within-patient modifier (exploratory)，D=preictal motif drift (state-readout, exploratory)，E=template hub vs `template_ablation_coverage`：E1 Yuquan-only capstone（冻结 outcome schema 后采集 recurrence label，source=yuquan_doc，几年后回来=复发），E2 Epilepsiae 仅 feasibility（先验 region-level，很可能不进触点级主分析）。模板 = masked lagPat propagation template / 排队号 = template_rank；"赢过随机" = channel-shuffle / within-shaft / anchor-matched null（null 须用同一镜像不变统计）；轴 = axial stat mod π。T0 前置 = eligibility audit（attempted/cacheable/analysis_eligible，不承诺全 inventory）。复用 topic3↔4 几何读出 `results/spatial_modulation/propagation_geometry/`（archive: `docs/archive/topic3/propagation_contact_plane_readout_2026-06-11.md`）。lagpat_broad 实际 20 subject dir + figures/。文献：Smith eLife 2022（IED echo ictal）、Brain 2014（7 onset patterns）、Proix Nat Commun 2018、Schroeder PNAS 2020 / Brain Comm 2023、Bartolomei Brain 2008（EI）、Matarrese/Tamilia Brain 2023（spike propagation onset/early/late-spread）、de Curtis/Avoli（pro/anti-ictogenic 争议）、source-sink / neural fragility。纪律：预注册 A 为唯一 primary，B/C/D exploratory，门写成数字 + 坏数据回归对照。）
