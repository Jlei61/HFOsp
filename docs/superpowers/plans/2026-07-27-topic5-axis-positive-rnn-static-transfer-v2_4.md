# Topic 5 axis-positive RNN read-back and static transfer v2.4 execution plan

## Milestone A：冻结审计

1. 指纹化 v2.3 formal runs、dataset v0.4、A/B axis artifacts 和 BB150 sidecars。
2. 只读 metadata/NPZ member names，不读取 `bb150_auc__*` 数值。
3. 冻结 n=9 collinear、n=6 reversed、n=5 strict-reversed。
4. 冻结 target-metadata eligible formal cohort、contact join 和 expected seizure count。
5. 输出 `INPUT_AUDIT_STATUS.json`，要求 `target_values_read=false`。

## Milestone B：Stage A0 cheap read-back

1. 从 v2.3 input audit 读取 transition-selected axis。
2. 从旧 axis artifact 读取 frozen `u_shared`。
3. 计算32-direction alignment null、PCA1 alignment 和已有 heldout NLL comparisons。
4. 输出 patient table 和 subgroup summary；不重训。

## Milestone C：Stage A1 RNN-selected axis

1. 增加 candidate-axis trainer，复用 v2.3 model、split 和 hyperparameters。
2. smoke：一位患者、两个方向、两个 epochs。
3. 正式：9 patients ×32 directions ×3 seeds。
4. 并行 launcher 使用 CPU 多进程；单任务 batch=2048，限制每进程线程数，避免 RAM
   峰值。
5. watcher 持续记录 complete/fail、RSS、runtime 和 target seal。
6. validation20 选方向；随后只对选定方向训练 n=6 source-full sensitivity。
7. heldout20 只在方向冻结后评分。

## Milestone D：冻结纯间期节点表征

1. 为 v2.3 formal target-metadata cohort加载已冻结 best checkpoints。
2. 对 full/no-history/isotropic/node-only 执行 paired 5000-rollout。
3. 由 train80 直接计算 empirical rank distribution。
4. 三 seed 在 contact-feature 层取中位数。
5. 写 per-subject NPZ、inventory、SHA256 manifest。
6. finalizer 逐项验证分布和为1、contact order一致、target未读。
7. 写 `TARGET_UNLOCK.json` 后停止并重新核对 hash。

## Milestone E：读取 clinical-onset BB150 target

1. target loader 必须先验证 `TARGET_UNLOCK.json`。
2. 读取 `[0,10] s`、1–150 Hz `bb150_auc__*`。
3. patient 内跨 seizure median，exact contact join，少于6 contact排除。
4. 对 full/no-history/isotropic/node-only/empirical 运行患者 LOSO ridge。
5. 每位患者5000次 all-contact permutation；within-shaft 为 sensitivity。
6. 输出 patient-level rho、null margin 和 exclusions。

## Milestone F：统计和图

1. Stage A：alignment margin、heldout NLL benefit、seed stability、reversed source term。
2. Stage S/H/X：static readout、history、axis comparisons。
3. axis-positive target-ready 小亚组只作描述性展示。
4. 严格按 spec 判 gates，不补 seeds、不换窗口、不改 ridge。
5. Figure panels：
   - A：全队列与预先冻结轴阳性亚组；
   - B：RNN candidate-axis selection；
   - C：selected axis 与 frozen A/B shared axis；
   - D：selected-axis heldout predictive benefit；
   - E：冻结 contact rank distribution 到 BB150 static field；
   - F：patient-level cross-state readout 与 gates。
6. 生成 PNG/PDF/JSON、`figures/README.md` 和正式结果报告。

## Milestone G：复现与 Git

1. 定向 tests；
2. 完整 manifest；
3. 检查 target read chronology；
4. 只暂存 v2.4 code/spec/summary/figure，不加入逐 epoch checkpoints；
5. focused commit + push。
