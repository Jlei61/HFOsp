# Epilepsiae Topic 3 — Stage 0 Artifact Census

> 归档日期：2026-04-27
> Stage：Topic 3 (where / SOZ spatial attribution) PR-2 准备 Stage 0
> 前序：Yuquan dual-track audit 完成（`docs/archive/yuquan_lagpat/dual_track_audit_2026-04-26.md`）
> 决策来源：用户 2026-04-27 明确指令——先 audit Epilepsiae artifact 合同，确认 legacy 能不能闭环，再决定 Track B 走不走

## 0. TL;DR

- 20/20 Epilepsiae subject 在 legacy interictal 树（`/mnt/epilepsia_data/interilca_inter_results/all_data_lns/<subject>/all_recs/`）有 **`sub_refineGpu.npz`** + **`*_lagPat.npz`**（含 `lagPatRaw / lagPatRank / eventsBool / chnNames / start_t` 全 keys）+ **`*_packedTimes.npy`**
- **20/20 subject 的 legacy 每 record `*_gpu.npz` 全是 216 bytes stub**（共 3,743 个 stub），全部不可加载
- 20/20 subject 在新 pipeline `results/hfo_detection/<subject>/` 有完整、可加载的 `*_gpu.npz` + `_refineGpu.npz`，所有 record 都有 `start_time` + `whole_dets` + `chns_names` + `events_count`
- **Track B replay 在 Epilepsiae 上 hard-impossible**：缺 legacy `<stem>_gpu.npz` 这一中间件，没法 freeze refine + detector 输入做 numerical replay
- **Topic 3 数据合同决议**：Stage 2（per-channel relaxed-refine + i/l/e 三层梯度）走新 pipeline，**不**声明 legacy lagPat 数值 parity
- 所有 20 subject 的 `new_verdict = ready`，Stage 2 batch 不需要等任何额外 detection 跑完（含 1077 — CPU 跑已于 2026-04-16 完成 189 records）

## 1. Census 数值（per-subject）

来源：`results/spatial_modulation/epilepsiae_artifact_census.csv`（由 `scripts/audit_gpu_npz.py --include-pack-lag` 生成）

| subject | legacy refine | legacy gpu real / stub | legacy lagPat | lagPat keys | new gpu loadable | new start_time | new schema | new refine | new ⊇ legacy | legacy verdict | new verdict |
|---|---:|---|---:|---|---|---|---|---:|---|---|---|
| 1096 | ✓ | 0 / 165 | 160 | ✓ | 165/165 | 165/165 | 165/165 | ✓ | ✓ | corrupt_gpu | ready |
| 1084 | ✓ | 0 / 252 | 221 | ✓ | 252/252 | 252/252 | 252/252 | ✓ | ✓ | corrupt_gpu | ready |
| 958 | ✓ | 0 / 225 | 225 | ✓ | 225/225 | 225/225 | 225/225 | ✓ | ✓ | corrupt_gpu | ready |
| 922 | ✓ | 0 / 114 | 114 | ✓ | 114/114 | 114/114 | 114/114 | ✓ | ✓ | corrupt_gpu | ready |
| 590 | ✓ | 0 / 254 | 245 | ✓ | 254/254 | 254/254 | 254/254 | ✓ | ✓ | corrupt_gpu | ready |
| 1150 | ✓ | 0 / 161 | 159 | ✓ | 161/161 | 161/161 | 161/161 | ✓ | ✓ | corrupt_gpu | ready |
| 442 | ✓ | 0 / 178 | 159 | ✓ | 178/178 | 178/178 | 178/178 | ✓ | ✓ | corrupt_gpu | ready |
| 1073 | ✓ | 0 / 231 | 231 | ✓ | 231/231 | 231/231 | 231/231 | ✓ | ✓ | corrupt_gpu | ready |
| 253 | ✓ | 0 / 268 | 268 | ✓ | 268/268 | 268/268 | 268/268 | ✓ | ✓ | corrupt_gpu | ready |
| 1146 | ✓ | 0 / 117 | **80** | ✓ | 117/117 | 117/117 | 117/117 | ✓ | ✓ | corrupt_gpu | ready |
| 916 | ✓ | 0 / 435 | 435 | ✓ | 435/435 | 435/435 | 435/435 | ✓ | ✓ | corrupt_gpu | ready |
| 620 | ✓ | 0 / 256 | 255 | ✓ | 256/256 | 256/256 | 256/256 | ✓ | ✓ | corrupt_gpu | ready |
| 583 | ✓ | 0 / 63 | 63 | ✓ | 63/63 | 63/63 | 63/63 | ✓ | ✓ | corrupt_gpu | ready |
| 548 | ✓ | 0 / 147 | 144 | ✓ | 147/147 | 147/147 | 147/147 | ✓ | ✓ | corrupt_gpu | ready |
| 384 | ✓ | 0 / 65 | 64 | ✓ | 65/65 | 65/65 | 65/65 | ✓ | ✓ | corrupt_gpu | ready |
| 139 | ✓ | 0 / 130 | 128 | ✓ | 130/130 | 130/130 | 130/130 | ✓ | ✓ | corrupt_gpu | ready |
| 1125 | ✓ | 0 / 160 | 160 | ✓ | 160/160 | 160/160 | 160/160 | ✓ | ✓ | corrupt_gpu | ready |
| 1077 | ✓ | 0 / 189 | 189 | ✓ | 189/189 | 189/189 | 189/189 | ✓ | ✓ | corrupt_gpu | ready |
| 818 | ✓ | 0 / 255 | **222** | ✓ | 255/255 | 255/255 | 255/255 | ✓ | ✓ | corrupt_gpu | ready |
| 635 | ✓ | 0 / 123 | 122 | ✓ | 123/123 | 123/123 | 123/123 | ✓ | ✓ | corrupt_gpu | ready |

观察：
- **legacy lagPat / packedTimes < legacy gpu records** 在 8/20 subject 上出现（1146 / 818 最严重，分别少 37 / 33 records）。说明 legacy pack 阶段对部分 block 已经丢过数据；这些丢失的 record 没有 lagPat artifact，即便 Track B replay 也无法对齐。
- **新 pipeline gpu records ≥ legacy gpu records** 所有 20 subject 上成立（`new_covers_legacy_records=True`）。新 pipeline 是 legacy detection 集合的严格 superset。
- 所有 20 subject 的 lagPat keys 完整（含 `lagPatRaw` / `lagPatRank` / `eventsBool` / `chnNames` / `start_t`）；但 lagPat 的存在 **不代表** legacy gpu 数据可恢复 —— lagPat 是 pack 阶段 downstream 产物，无法反推回 detection event sets。

## 2. Track B 不可行性证明

Yuquan dual-track 的 Track B replay 之所以可行，前提是**三件套同时存在**：

| 件 | Yuquan replay 用途 | Epilepsiae 现状 |
|---|---|---|
| legacy `<subject>/sub_refineGpu.npz` | refine 通道选择输入 | 20/20 ✓（2021-04~06 mtime，无覆盖） |
| legacy `<raw>/<stem>_gpu.npz` | refine + pack 阶段的 detection event 输入 | **0/20 ✓** — 全部 216 byte stub，无法加载，无法恢复 |
| legacy `<raw>/.legacy_backup/<stem>_lagPat.npz` | pack/lag 数值 ground truth | 不存在 `.legacy_backup/`；legacy lagPat 直接在 `all_recs/` 下，但**不是 backup**，是 2021 production 输出本身 |

缺第二件套（real `*_gpu.npz`）→ replay 无法 freeze "refine 输入 + detector 输出"。即便强行给 refine 喂新 pipeline 的 detection 来生成 packing，那也是**新 detection + 老 refine 决策**，与 legacy 数值不可对齐。

→ 与 Yuquan 的"replay = 旧数据 + 修过的代码"路径**不可类比**。Epilepsiae 的"replay" 只能是"新数据 + 修过的代码"，那不叫 replay，叫 re-run。

## 3. Topic 3 数据合同（Decision）

| 决议项 | 内容 | 适用范围 |
|---|---|---|
| Stage 1 Track B replay | **跳过** | 全 20 subject |
| Stage 2 per-channel relaxed-refine 输入 | `results/hfo_detection/<subject>/*_gpu.npz` + `_refineGpu.npz`（新 pipeline） | 全 20 subject |
| SOZ-AUC validation 数据源 | 新 pipeline `_refineGpu.npz` + 现有 PR-1 `scripts/plot_refine_soz_validation.py`（已发布 Yuquan AUC=0.874 / Epilepsiae AUC=0.952） | 全 20 subject |
| 数值 parity 主张 | **不声明** legacy lagPat 数值 parity | 全 20 subject |
| legacy lagPat artifacts 用途 | 仅作为**结构参照**（chnNames / start_t / packedTimes 时间戳）—— 用于 PR-1 已经产出的 lagPat-based 主张（interictal_synchrony / interictal_propagation / event_periodicity）的历史背景；Topic 3 PR-2 不再依赖 | 全 20 subject |
| 1077 detection 状态 | 已完成（`results/hfo_detection/1077/` 含 189 records，`1077_cpu_summary.json` 时间戳 2026-04-16） | 不阻塞 Stage 2 |

## 4. 与 Yuquan dual-track 的对比

| 维度 | Yuquan | Epilepsiae |
|---|---|---|
| Track B 可行性 | 部分可行（detector-comparable 13/21；strict-pass 6/14） | hard-impossible（0/20） |
| legacy gpu_npz 状态 | 21 subject 中 14 有完整 legacy `_gpu.npz`，6 无 legacy gpu，1 reference scheme 不同 | 20/20 全 stub |
| legacy `.legacy_backup/` 存在 | 是（2021 production lagPat 移到 backup） | 否（lagPat 直接在 `all_recs/`） |
| refine 是否被 new pipeline 覆盖 | 7/14 在 2026-04 被覆盖（refine drift），7/14 ε pass | refine 时间戳全在 2021，无覆盖 |
| 数值 parity 主张范围 | 6 subject strict-pass（68 records ε maxabs） | **不声明** |
| Topic 3 路径 | 走新 pipeline（PR-1 已完成 9 subject） | 走新 pipeline（Stage 2 即将开始） |

→ Yuquan 那 6 subject 的 strict parity 主张**不**适用于 Epilepsiae。Topic 3 的 Epilepsiae 部分本来就走"新 pipeline 主权"路线，与 Yuquan dual-track 是**两条独立的合同**。

## 5. 后续 Stage 路线（不在本档案内实施）

后续步骤详见 plan：`/home/honglab/leijiaxin/.claude/plans/home-honglab-leijiaxin-hfosp-docs-archi-stateful-cascade.md`

- **Stage 2**：refactor `scripts/run_spatial_modulation.py`，加 `--dataset epilepsiae` switch + i/l/e 三层 + per-metric 方向 map + 三对 paired Wilcoxon + Bonferroni + subject-level 单调 sign test。**不**引入 Page trend test（无新依赖）。subject 集合 = `results/hfo_detection/` 下 20 个数字子目录（focus_rel 缺失者照常跑，channel `region_label="unknown"`，仅在 paired stats 阶段被剔除）。
- **Stage 3**：smoke 2 subject → 全 20 batch。预期 ≥15 subject 在 paired stats valid。
- **Stage 4**：扩 `scripts/plot_spatial_modulation.py`，写 `results/spatial_modulation/figures/epilepsiae/README.md`（中文，2–4 句/图 + 关注点），归档 `docs/archive/topic3/epilepsiae_three_tier_pr2_<date>.md`，主文档 `docs/topic3_spatial_soz_modulation.md` 填实际数值 + archive 链接。

## 6. 输出 artifact

- `results/spatial_modulation/epilepsiae_artifact_census.csv`（20 行 × 23 列；每 subject 1 行；含 legacy_verdict + new_verdict）
- `scripts/audit_gpu_npz.py`（新加 `--include-pack-lag` 模式 + `audit_subject_artifact_census_epilepsiae()` + 5 个 helper）
- 本档案
