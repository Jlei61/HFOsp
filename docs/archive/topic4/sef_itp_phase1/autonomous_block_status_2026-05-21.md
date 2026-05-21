# Autonomous 8h Block — Status Report (2026-05-21)

> **2026-05-21 用户回归后修订（v2）**：本文档第一版（8h 块结束时写）含**一条错误判断**——
> "/mnt 里没有 3D 坐标，需要数月工程从 CT/MRI 重做定位"。用户回归后立刻补查，
> 找到两套现成坐标源（Yuquan `patients_elecs_reGen/<sid>/chnXyzDict.npy` + Epilepsiae
> SQL `electrode` 表）。**坏味道**：autonomous 块开始时只 grep 了 EDF/SQL 主数据根
> 第一层（`/mnt/yuquan_data/yuquan_24h_edf/`、`/mnt/epilepsia_data/all_data_sqls/`），
> 漏掉影像 / 电极坐标侧的挂载（`/mnt/yuquan_data/yuquan_images/...`）。新判断：
> coord loader 是**小焦点 PR**（1-2 周），不是数月工程。详见 `docs/topic0_methodology_audits.md` §3.2 修订版。
>
> **同期修订**（基于用户审阅）：H1 检验从单 PASS/NULL/FAIL 拆成三独立层
> （descriptive / strict / envelope）。Plan archive §3.2 + `src/sef_itp_phase1.py`
> + `tests/test_sef_itp_phase1.py` 都已落字。`test_h1_strict_diffuse_null_strict`
> 改为严格断言 NULL（原版 `in ("NULL", "PASS")` 太松，抓不住回归）。
> **30 / 30 unit tests GREEN**。

> **范围**：用户 2026-05-20 → 2026-05-21 离开 8 小时期间 autonomous 推进 SEF-ITP Phase 1 基础设施
> **目标（用户原话）**："推进一些基础设置，和 advisor 交互进行。到我回来时给出基本的建模可执行框架"
> **执行口径**：plan archive doc + TDD 代码骨架 + 测试 GREEN + CLI dry-run；**不**跑真实 cohort 数据、**不**改 framework v1.0.2、**不** git commit

---

## 1. 已交付（durable artifacts）

### 1.1 文档（4 处）

| 路径 | 内容 |
|---|---|
| `docs/archive/topic4/sef_itp_phase1/plan_2026-05-21.md` | **Phase 1 plan-of-record v1**：操作细节 H1/H2/H6 数据流 + 13 TDD case 规格 + 输出 JSON schema + 复用 helper 目录 + 红线 |
| `docs/topic0_methodology_audits.md` §2 §3.1 §3.2 §5 | Phase 0 进度更新（5a/5b/5c/5d.1 完成）+ panel d 加强的科学发现 + **新增 §3.2 coord-loader gap** |
| `docs/paper_overview.md` Topic 0 + Topic 4 §一句话总论 | 未结清问题从 1 个 → 2 个；panel d 92% 加强 |
| `docs/topic1_within_event_dynamics.md` §2 | "簇内 86% identity bias → 92%" + SEF-ITP 几何前提加强 note |

### 1.2 代码（3 处）

| 路径 | 内容 |
|---|---|
| `src/sef_itp_phase1.py` | **核心模块**：H6 segregation + H1 三层 (a/b/c) + H2 set/spatial reversal + 距离度量 (3D Euclidean / shaft-ordinal) + matched null with degradation + helpers |
| `tests/test_sef_itp_phase1.py` | **26 unit tests，全 GREEN**：covers channel parsing / distance metrics / H6 cross-shaft + single-shaft EXCLUDED + H1a 紧凑 + H1c 包络 + 圆性 guard + H2 perfect swap + anti-swap FAIL + contract assertions |
| `scripts/run_sef_itp_phase1.py` | CLI runner with `--dry-run` mode（synthetic 数据） + real-data 模式 `NotImplementedError` 守门（防误调） |

### 1.3 验证

```bash
$ python -m pytest tests/test_sef_itp_phase1.py -q
.......................... 26 passed in 1.07s

$ python scripts/run_sef_itp_phase1.py --dry-run --hypothesis all --distance-metric euclidean
[dry-run] verdict summary:
  H6: PASS
  H1 cluster 0: NULL
  H1 cluster 1: NULL
  H2 pair 0<->1: PASS
```

---

## 2. 未交付（明确不做 / 受限于前置）

- ❌ **实跑真实 cohort 数字** —— autonomous 块明确不做，等 Phase 0a §5f + Phase 0b coord-loader 完成后用户手动启动
- ❌ **Phase 0b coord-loader PR 本身** —— 独立 PR，由用户决定怎么启动（见 §3 第 2 个 concern）
- ❌ **figures** —— AGENTS.md 规范"图实际生成后再写 README"，未跑数据不画图
- ❌ **git commit** —— 用户未授权
- ❌ **FHN toy 代码** —— Phase 4，远未到
- ❌ **修改 framework v1.0.2** —— 锁了

---

## 3. 三条 advisor flag 的真实问题（用户决定）

### 3.1 H1 PASS 条件比 framework 暗示的**更难**

dry-run 上 H1 cluster 0/1 都是 NULL —— 即使我设的是 perfect compact source/sink（shaft 上前 3 / 后 3）。

**根因**：matched null 用 "shaft + participation + HFO rate" 约束。如果 source/sink 都在一根 shaft 上（生物学常态），null 也被强制全部抽自该 shaft → null 也很 compact → "actual < null"  的 effect size 必须**特别大**才能 PASS。

**含义**：framework v1.0.2 §3.1 lock 的 matched null 约束**可能太严**。生物学 source/sink 多数在同一 shaft 上时，null 与 actual 接近，PASS 难度高。

**用户的选择**：

- (a) 接受 H1 需要很大 effect size 才能 PASS（保留 framework，cohort 跑出来若 NULL 就是 NULL）
- (b) 调整 matched 约束顺序：先放松 shaft 约束（同 shaft 子集换为同 hemisphere），再放松 HFO rate / participation
- (c) 把 H1c envelope 升级为主要判据（不受 shaft 约束影响）

**我没改 framework**，flag 给用户。

### 3.2 H2 set-reversal null 在 `sample_with_replacement` 模式下**有污染**

当 `union(S_A, K_A, S_B, K_B) < sum(sizes)`（合集小于角色总数 — 实际数据**几乎不会**发生，但 synthetic perfect swap 会触发），代码 fallback 到 `rng.choice(union, replace=True)`。这允许 null 抽样里**同一通道同时出现在 S_A 和 K_A**——而真实数据里 source 和 sink 在同一 cluster 里**必然不相交**。

**含义**：null 分布包含真实数据不可能产生的角色配置，cohort 跑真实数据时不会触发（因为 fwd/rev pair 的 union 必然 ≥ 4×|k|），但 synthetic 测试通过部分受此 fallback 加持。

**用户的选择**：

- (a) 加 invariant：S_A ∩ K_A = ∅ AND S_B ∩ K_B = ∅，违反则 raise（更严格）
- (b) 改 fallback 为 "constrained without replacement"：抽样时 S_A 与 K_A 必须 disjoint
- (c) 保留现状但 archive 注明 fallback 仅 synthetic edge case 触发

**我倾向 (a) + (c)**：真实数据进 H2 时 sources/sinks 必然 disjoint（按 framework endpoint 定义），违反应该 raise；synthetic 测试可单独处理。但这是设计决定，不动。

### 3.3 dry-run JSON 默认输出路径需要避开实跑路径

advisor 发现 dry-run 输出原本落在 `results/topic4_sef_itp/phase1_spatial_geometry/`，与未来真实 Phase 1 cohort 路径冲突。

**已修复**（2026-05-21 块内）：

- `--output-dir` default 改为 `results/topic4_sef_itp/_dry_run_artifacts/`
- 已生成的 dry_run JSON 已移到 `_dry_run_artifacts/`
- 空的 `phase1_spatial_geometry/` 目录已删除

**含义**：现在未来 agent 看到 `results/topic4_sef_itp/_dry_run_artifacts/dry_run_*.json` 一眼能识别是测试 artifact 而非 cohort 结果。

---

## 4. 三条 advisor 注（不阻塞）

1. **`_load_valid_mask_from_lagpat` 没写** —— 等 `load_subject_for_phase1` 实现时（Phase 0a §5f + Phase 0b 完成后）一并写。CLI 当前 `NotImplementedError` 守门
2. **dry-run JSON 含完整 null 分布** —— 真实数据跑时可能很大。未来考虑把 null arrays 拆到 `_null_distributions/` 子目录或只存 summary stats。**当前不动**
3. **figures README 未建** —— AGENTS.md 规范"图实际生成后再写"，正确做法，没建是 by design

---

## 5. 重要发现 —— Phase 0b coord-loader 是 SEF-ITP framework 隐含漏洞

autonomous 块开始时 advisor 推荐做的"coord 可用性 gating check"暴露了 framework v1.0.2 未明示的**隐含假设**：

- 全 `src/` grep `coord/electrode.*xyz/mni/montage` → **0 个** 3D coord helper
- `/mnt/yuquan_data/yuquan_24h_edf/<subject>/` → 无 coord / elec 文件
- `/mnt/epilepsia_data/` (maxdepth 3) → 无 coord 文件
- `docs/DEVELOP_PLAN.md` 把 `mni_coords` / `electrode_distance` 列为 **"Phase B 阻塞"**（已规划未启动）

**意味着**：

- SEF-ITP framework v1.0.2 §3.1 H1 列出"四种距离并列报告（Euclidean / shaft-ordinal / cortical surface / SC）"，**当前数据状态下只有 shaft-ordinal 立刻可用**
- H6 可用 shaft-ordinal 跑（不需 3D）
- **H1 / H2 主分析需要 3D Euclidean coords —— 必须先建 Phase 0b coord-loader PR 再启动**

**framework 文档级修订建议**（用户回归后做）：

- 在 `docs/topic4_sef_itp_framework.md` §0 + §6.1 加 banner "Phase 0b coord-loader 是 H1 / H2 主分析的硬前置"
- 把 v1.0.2 verdict 升级为 v1.0.3（这次是 framework-level erratum，不是 plan-level）

**新前置 Phase 0b 建议路径**（plan archive doc §1 已写）：

1. 调查 Yuquan `.fif` 文件 montage / dig points（MNE-Python `info['dig']` 字段）
2. 调查 Epilepsiae SQL `electrode` 表是否有 anatomic / x_mni / y_mni / z_mni 字段
3. 调查 `.cursor/rules` 或临床合作方是否提供过 elec.txt
4. 如果都没有 → 数月工程（重做 SEEG localization）
5. 如果任一可用 → coord-loader PR ~ 1–2 周

---

## 6. Phase 0a 实际进度（用户离开期间观察）

用户提供的 2026-05-20 status report 显示 §5a / §5b / §5c / §5d.1 已完成，§5d.2.1 在跑。autonomous 块未推进 Phase 0a（不该越权改 Topic 0 in-flight 代码）。

**已写进 Topic 0 主文档**（durable）：

- §5 路线图表更新：5a/5b/5c/5d.1 标"已完成 2026-05-20"，含数字摘要 + archive 链接
- §3.1 加 "panel d 加强" 的科学发现块（簇内 87.9% → 92.2%，p=3.17e-4）
- §2 一句话当前状态从 "1 个未结清" 升级为 "2 个未结清"

---

## 7. 用户回归后我建议优先做的事

按优先级：

| P | 动作 | 估算 | 启动条件 |
|---|---|---|---|
| **P0** | 决定 §3.1 / §3.2 / §3.3 的处理方式（接受 / 调整 / 改设计） | 30 min review | 立即 |
| **P0** | **决定 Phase 0b coord-loader PR 怎么启动** —— 这是最大的真实问题 | 1–2 小时调查 + 决策 | 立即 |
| **P1** | review v1.0.2 framework 是否应升级到 v1.0.3（加 Phase 0b 硬前置 banner） | 15 min | 立即 |
| **P2** | 让 Topic 0 §5d 系列跑完（5d.2.0 / 5d.2.2 / 5d.3 / 5d.4） | 已部分在跑 | 等 5d.2.1 完成 |
| **P3** | Checkpoint B advisor consult（PR-4B/D 方向） | 30 min | §5d 全完成后 |
| **P4** | 启动 §5f PR-6 endpoint anchoring on masked —— H1/H2 真正数据基础 | 2–3 天 | Checkpoint B 通过后 |
| **P5** | 启动 Phase 0b coord-loader PR | 1–2 周（取决于数据） | 与 P4 并行 |
| **P6** | Phase 0a + 0b 都完成后，跑 SEF-ITP Phase 1 H6/H1/H2 cohort | 1–2 天 | P4 + P5 完成 |

---

## 8. 不确定 / 我没有把握的事

1. **Phase 0b 调研路径是否能找到 coords** —— 我猜 Yuquan .fif 可能带 montage（MNE 标准），但没验证；Epilepsiae SQL 我也没查 schema
2. **H1 PASS 条件是否真的太严** —— §3.1 是 advisor 推断 + 我 dry-run 观察，没有真实 cohort 数据验证；可能真实数据 effect size 足够，PASS 自然产生
3. **H2 sample_with_replacement fallback 是否需要硬修复** —— 真实数据上几乎不触发，但 framework-level 是否要锁 "S_A ∩ K_A = ∅" invariant 是设计选择

---

## 9. 时间消耗记录

| Phase | 实际耗时（估算） |
|---|---|
| Orient + coord gating check + advisor scope | ~45 min |
| Phase 0 doc cleanup (topic0 + paper_overview + topic1) | ~30 min |
| Phase 1 plan archive doc | ~1.5 h |
| Midpoint advisor | ~10 min |
| src/sef_itp_phase1.py 模块 | ~2 h |
| tests/test_sef_itp_phase1.py（含 2 次 fixture 调试） | ~1.5 h |
| scripts/run_sef_itp_phase1.py CLI + dry-run 验证 | ~45 min |
| Final advisor + status doc | ~30 min |
| **总计** | **~7.5 h** |

---

## 10. 一句话总结

framework v1.0.2 + Phase 0a 已完成 step + Phase 1 可执行代码骨架 + 26 unit tests GREEN + CLI dry-run = **当前距离实跑 Phase 1 cohort 数字差 2 个 PR**：(a) Phase 0a §5f PR-6 endpoint on masked，(b) **新发现的 Phase 0b coord-loader PR**。后者是 framework v1.0.2 未明示的隐含前置，是 autonomous 块最有价值的发现。
