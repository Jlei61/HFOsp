# Fig3-B shared-gradient maxAB — 匹配空间置换 null（两档）+ 时间维校正

### `<subject>_maxab_spatial_null.png / .pdf`

仅对通过二维几何门的 shared-gradient Fig3-B 病例叠加**两个被试内空间置换 null + 时间维多重比较校正**。两个 null 与主图保持同一批成功 seizure、同一 66-window grid、同一冻结 `shared_a/shared_b`、同一 fingerprint 和 maxAB 选择；每个 seizure×replicate 只抽一次空间映射，并将同一映射贯穿全部 66 个窗口，以保留 null trajectory 的时间依赖。每次 shuffle 都完整重跑 shared scorer，再对 seizure 取中位。两档强度：
- **all-contact**（弱，灰点线=null 中位）：值在**所有触点**间打乱。
- **within-shaft**（强，主对比，蓝带）：值只在**每根杆(shaft)内**打乱，保留'哪根杆热'的植入几何。

图元：粗 rust=观测中位、浅 rust 带=观测 IQR；蓝虚线+蓝带=within-shaft null 中位+95%；灰点线=all-contact null 中位；浅 rust 竖带=within-shaft **cluster 校正显著区间**；蓝三角=within-shaft **maxT 校正显著窗**；0 s 虚线=onset。副标题给 within-shaft 的 cluster/maxT/pointwise 显著窗数 + all-contact pointwise。

**三档显著性（都在 stats CSV，两个 null 各一套）**：pointwise（逐窗，未校正）< maxT（逐窗 FWER）< cluster（时间维、对持续抬升敏感，= paper-facing '显著区间'）。

**关注点**：观测中位数是否**离开蓝色 within-shaft null 带**并形成 cluster 显著区间。⚠️within-shaft null 的分辨力取决于每根杆的触点数（见 summary.shaft_structure）；单触点杆无法打乱，若相似度完全由'哪根杆热'解释，within-shaft null 就贴着观测、几乎无显著窗——这是诚实的强 null 结果。只检验 maxAB scaffold，不做 onset increment / signed A/B / 多频带。旧 own-plane null 已移入 `legacy_own_plane_spatial_null/`，不得与本目录结果混用。**单被试二维素材，非 formal cohort spatial gate。**
