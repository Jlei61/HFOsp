# LEGACY — Fig3-B own-plane spatial null

本目录由旧 `spatial_null/` 原样迁入。producer 使用 legacy `_compute_values/_scorer` own plane，与当前 frozen shared-gradient Fig3-B 的 subject、seizure 集和 observed statistic 均不匹配；只保留作历史审计。

**交付状态**：本目录二进制是当前机器上的 `local-only legacy artifacts`，Git 只跟踪本 README，不把它们作为正式 archive evidence。旧 null 还采用逐窗口独立 permutation，不能支持合法的 cluster/maxT 联合校正；新 checkout 中允许整套二进制不存在。

### `<subject>_maxab_spatial_null.png / .pdf`

在 Fig3-B 的 maxAB 面板上叠加**两个被试内空间置换 null + 时间维多重比较校正**。两个 null 都：保持同一批 seizure / 时间窗 / A|B 模板 / 场平滑 / maxAB 逻辑，只打乱每窗 per-channel 能量值，完整重跑读出、对 seizure 取中位（每次 seizure 独立置换）。两档强度：
- **all-contact**（弱，灰点线=null 中位）：值在**所有触点**间打乱。
- **within-shaft**（强，主对比，蓝带）：值只在**每根杆(shaft)内**打乱，保留'哪根杆热'的植入几何。

图元：粗 rust=观测中位、浅 rust 带=观测 IQR；蓝虚线+蓝带=within-shaft null 中位+95%；灰点线=all-contact null 中位；浅 rust 竖带=within-shaft **cluster 校正显著区间**；蓝三角=within-shaft **maxT 校正显著窗**；0 s 虚线=onset。副标题给 within-shaft 的 cluster/maxT/pointwise 显著窗数 + all-contact pointwise。

**三档显著性（都在 stats CSV，两个 null 各一套）**：pointwise（逐窗，未校正）< maxT（逐窗 FWER）< cluster（时间维、对持续抬升敏感，= paper-facing '显著区间'）。

**关注点**：不得把本目录的 `13/20`、`7/20`、`2/20` 或任一 subject null 结果接到新版 shared-plane trajectory。新版匹配 null 只认重新生成的 canonical `spatial_null/` 目录及其 shared fingerprint provenance。
