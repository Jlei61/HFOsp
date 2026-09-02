# Topic 5.1 患者特异多尺度有效传播 scaffold v0.5 执行计划

> 对应 spec：`docs/superpowers/specs/2026-08-13-topic5-patient-specific-multiscale-effective-scaffold-v0-5-design.md`
>
> 状态：**SCIENTIFIC CLOSEOUT COMPLETE。A–H 已执行完成：531/531 正式训练、Stage F target-free freeze、17 人/167 seizures
> locked internal benchmark、Figure 6/source data 与 standalone machine closeout audit 均完成；
> 仅余 commit/push 与主线 figure registry 集成。** v0.3 已由 commit
> `bd9d8621` 与 tag `topic5-lbss-full-tissue-v0.3-closeout` 独立冻结。

## 0｜v0.3 immutable closeout

1. 保持 v0.3 results、fields、Figure 6 与统计只读；
2. parent commit/tag、producer hashes、`CLOSEOUT_AUDIT.json` 写入 v0.5 provenance；
3. v0.5 只新建结果根 `results/topic5_multiscale_effective_scaffold_v0_5/`，不覆盖旧产物。

完成标准：远端 branch/tag 可解析到同一 parent commit；旧承重测试和 closeout audit 仍 PASS。

## A｜Target 物理隔离与全 parent cohort builder

1. 从干净 base commit 新建 immutable execution worktree；复制代码到 `run_snapshot/` 并只读；
2. target-free 进程只挂载 routing metadata；early-ictal energy values 通过权限或独立 mount 不可读；
3. builder 对全部 masked-rank K=2 parent 自动应用 `min_joint_contacts=6`；
4. 自动生成 34 人 contact-space、28 人/42 fits spatial 和 17 人/167 seizures exact intersection 的
   inclusion/attrition 表；不得只硬编码 recovery patients；
5. 对 dataset、plane、H、split、event-lag sidecar 和 parent outputs 记录 SHA256；
6. 输出小 montage observability QC 与 target-free cohort recovery 图。

完成标准：没有满足规则却被漏掉的患者；target access counter=0；正式分母由 builder 产物决定。

## B｜Cache、train-only modes 与单元测试

1. 生成 42-fit cache；旧 31 fits 的 rank/split arrays 与 v0.3 做逐位兼容审计；
2. 每 fixed development training partition 只用 train events 重建 K=2 modes、prefix posterior 与
   templates；modes 只作分层，不过滤任一 plane 的 prediction task；
3. 构建 3 份 split-isolated suffix-pairing null mappings，分别绑定 3 个 suffix-null seeds；
4. 冻结 prefix-template TA/TB、posterior mixture 和 train-prevalence mixture builders；
   对 non-collinear 患者，oracle own_a/own_b all-event candidates 与 non-oracle A/B-aligned
   train-mode components 必须分别构建，禁止将 geometry view 直接当作 mode component；
5. 实现 H top-90%-mass support、front distance、local graph path distance 与 latency sidecar；
6. 运行 spec §12 的全部承重单测。

阶段图：E1146 与一个 6-contact 患者的 full-tissue layout/H support；train-only mode/template 与 suffix
mapping destruction audit。图不读取 early target。

## C｜Shortcut、graph-control 与 detectability audit

1. 审计 L1/L3 candidate pool size、per-node opportunity、exposure fraction 和 proposal frequency；
2. 为每个 L3 fit/seed 构造 exact degree/reciprocity/distance-bin matched L2m mask；
3. 在 2–3 个真实患者几何上运行 functional-shortcut synthetic positive control：只要求 L3 类别在 distal
   prediction/attenuation 上可识别，不要求找回 exact edges；
4. 冻结 L2m matching algorithm、graph-null seeds、training budget 和不可行处理；
5. frozen macro rewiring 只登记为 perturbation，不参与 topology-selection inference。

阶段反思：若 L2m exact matching 大范围不可行，先修 matching contract；不得退回旧 L2 或用 frozen
rewiring 冒充重训 control。若 L1/L3 exposure 严重失衡，只删除 L3−L1 机制解释，不运行时扩模型。

## D｜Cross-fitted J 与低成本信息审计

1. 用非 f events 的 nested cross-fitting，为每个 outer test fold 构造 `J_p^{(-f)}`；
2. mode-specific nonnegative local-wave slopes，数据不足时退化 pooled slope；beta=0 保留为
   `LOCAL_WAVE_UNSUPPORTED`；
3. 计算 event-mean sparse exceedance-burden J；原 event-median 已在 pre-training feasibility 中 28/28
   退化为零，固定保留为 sensitivity，并并列报告 temporal-block median、nonzero-event fraction、
   rank-only tau 与 violation fraction；
4. 复用现有 L3 checkpoints，计算 prefix-template advantage 与连续 prefix entropy H 的关系；
5. 对 recording block/session-heldout split 做 sensitivity；
6. 冻结 28 人 J、distance denominators、NOT_IDENTIFIABLE reasons 和 primary interaction scorer；
   近一维几何 2 人保留在完整 census，并固定一份去除敏感性。

阶段图：代表患者 local-wave fit/OOF residual、28 人 J 分布、RNN-template advantage vs H。不得画成
anatomical tract 或临床 recruitment latency。

阶段反思：这一步回答测量是否有效，不因 J 分布或 interaction 方向改变模型、阈值或 cohort。

## E｜正式 target-free 训练：531 units

正式 launch manifest 在启动前一次性列全：

```text
Exact shared reuse 11 fits: C-suffix/L2m x3 = 66
Mandatory full retrain 31 fits: L0/L1/L2m/L3/C-suffix x3 = 465
Total = 531
```

1. nohup/tmux + immutable launcher；worker 数由实际显存/RSS smoke 冻结；
2. 每单元独立 model-init、graph-null、suffix-null seeds；
3. best checkpoint 仅允许 mask-freeze 后 epoch；C-suffix 只能用同一 suffix-null validation 选择，
   L2m 只能用自身 matched-graph validation 选择；
4. resume 保存 optimizer/RNG/mask/edge age/rewire counter/freeze status；
5. DONE/FAILED、峰值显存、nonfinite、retry、producer hashes 原子写入；
6. 旧 L2 只读取为 sensitivity，不替代 L2m。

Recovery fits 的执行顺序固定为先完成 L3 并冻结 added mask，再按该 mask 构造 L2m、最后训练 L2m；
这只是预定义依赖，不根据 L3 性能决定是否运行 L2m。现有 31 fits 可直接从已冻结 L3 masks 构造。

阶段图：训练收敛、L2m matching、suffix-null destruction、all/distal heldout metrics；patient-first。

聚合时不得直接使用 unit 内继承自 v0.3 的 q50/q80 `distance_bins` 作为 primary；必须从原始
`distance_decisions.json` 按冻结的 `r_local` 重算 local/nonlocal，并验证所有 arms/seeds 使用相同
decision support。q50/q80 结果仅保留为 descriptive sensitivity。

阶段反思：primary 是 `(L3-L2m) x J`，不是 L3 全队列必须获胜，也不根据阶段结果补新拓扑。

## F｜Target-free mechanism、gain 与所有 field 冻结

1. 计算 trajectory Jacobian、finite-horizon G3 和 empirical output amplification；
2. 执行预定义 gain-adjusted sensitivity；
3. 用含 postsynaptic derivative 的 signed/absolute Phi 冻结 TA/TB bundles；
4. 做 same-mode/cross-mode/matched-random attenuation；
5. 计算 shrinkage precedence、coarse endpoint density、contact-space influence 与 stability；
6. 生成并冻结全部 intact、template、mixture、C-suffix、L2m、attenuated、gain-adjusted fields；
   mixture freeze 必须另写 producer hash、14 位 own-view label parity、70 个 patient-arm repair rows，
   并证明 oracle A/B vector hashes 未改变；
7. 预生成 synchronized all-contact primary null index maps 与 geometry-eligible robustness null maps；
8. 写 `MODEL_FIELD_MANIFEST.json`、`ATTENUATED_FIELD_MANIFEST.json`、scorer hash 和 target-access=0。

阶段图：TA/TB effective-flow fields、same/cross attenuation、finite-horizon gain；exact edges 只进补图。

完成标准：target unseal 后 scorer 只能读取 frozen vectors，不能调用 model/field builder。

## G｜显式 target unseal 与 locked internal benchmark

1. 校验 source tree、run snapshot、field/null manifests 与 hashes 后显式解封；
2. 读取 17 人/167 seizures 的 0–10 s、1–150 Hz broadband energy values；
3. 统一转 earlyness，计算 signed best-mode Spearman oracle repertoire correspondence；
4. primary early test：`rho(J, C_L3-C_L2m)>0`；固定 100,000 次 patient-label permutation 与
   5,000 次 synchronized all-contact coherent spatial-null interaction 组成联合主判据，两项都必须通过；
5. 计算 train-prevalence mixture signed Spearman 和预定义 robustness endpoints/nulls；
6. patient bootstrap 每次同时重算 J/Delta；报告 LOO、高 J leaveout、6–7-contact leaveout；
7. 明确标记 `LOCKED_INTERNAL_MECHANISTIC_FOLLOWUP`，不写 independent confirmation。

阶段图：17 人 oracle 与 non-oracle field correspondence、`Delta_EI vs J`、primary null；E1146 仅作
预冻结代表患者。主图只给 primary 统计星号，secondary 报效应量/CI，不堆显著性标记。

## H｜收口与单一后续决策

1. 分开 adjudicate：primary target-free interaction、suffix information、template shortcut、mode-flow、
   cross-state interaction；
2. 更新中文 archive、Topic 5 主文档、index 和机器可读 claim summary；
3. Figure 6/Extended Data 产出 600-dpi PNG、单页 PDF、SVG、source CSV/JSON、中文 README 并逐张 QA；
4. 只按冻结决策树提出 E1/E2/E3 中一个新 spec，不直接训练；
5. 用户终审后再 commit 结果。

## 执行顺序与资源原则

```text
0 immutable v0.3 closeout
-> A target isolation + full builder
-> B train-only modes/cache/tests
-> C graph matching + detectability
-> D cross-fitted J + contract freeze
-> E 531-unit training
-> F mechanism + every field/null frozen
-> explicit target unseal
-> G early-ictal scoring
-> H closeout + one-extension decision
```

所有大规模阶段使用 immutable snapshot、nohup/tmux、原子 checkpoint 和独立 watcher。验收要求 0
unresolved OOM，而不是假装任何已恢复 OOM 从未发生。任何阶段图都必须先通过数据合同检查，再做
Nature-style 视觉 QA；不靠往图里增加文字解释科学逻辑。
