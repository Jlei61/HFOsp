# Topic 5 gradient 多频带显著性图

### gradient_multiband_significance_onset_0_10s.png / gradient_multiband_significance_onset_0_10s.pdf

严格复用旧 F2 的 7 个 primary 频带、单轴 violin、逐患者点、黑色 cohort Δ 横杠和 7-band max-over-bands FWER 标注。激活量来自旧 producer 已接纳的 onset 后 `[0,10] s` 事件窗口；冻结 gradient field 按结果无关的可用性固定路由：有完整 `shared_a/shared_b` 时取 shared maxAB，否则取 own maxAB，不在 shared 与 own 之间择优。Null 复用旧 producer 的三层空间置换：优先杆内洗牌，杆内触点不足时依次退到 distance-bin 与 subject-wide；每个发作 1000 次，每次都重新做平滑、mirror 与 A/B max，随后按 seizure→subject→cohort 折叠。

**关注点**：n=19（Epilepsiae 17 + Yuquan 2；shared 8、own fallback 11），1/7 个频带通过 FWER。每频带的 spatial-null 强度构成为 within-shaft strong 2、distance-bin fallback 4、subject-wide weak 13，因此这是沿用旧 F2 合同的弱空间-null cohort 证据，不能写成 19 人都通过纯杆内 null。delta_HYP_slow: pFWER=0.8781, Δ=0.014; theta_preictal_PAC: pFWER=0.9451, Δ=0.008; alpha_sharp_leq13: pFWER=0.6064, Δ=0.026; beta_LVFA_low: pFWER=1, Δ=-0.017; gamma_LVFA: pFWER=0.02797, Δ=0.054; hg_low_ripple: pFWER=0.967, Δ=0.006; ripple_high: pFWER=1, Δ=-0.013。Epilepsiae 的 0 点为 clinical onset；Yuquan 没有 clinical-onset 标注，只保留真实 EEG onset，因此图题写 `onset 0–10 s`，不把 19 人统称为 clinical onset。

### gradient_multiband_significance_original_f2_windows.png / gradient_multiband_significance_original_f2_windows.pdf

这是旧 narrow F2 的受控 gradient-field 替换版。原 producer 接纳的五类重叠窗 `[-5,5]`、`[0,10]`、`[5,15]`、`[10,20]`、`[15,25] s`、`ictal_fraction≥0.5`、window→seizure→subject 中位数、旧 endpoint 二维坐标上的空间置换分组、1000 次置换与七频带 max-over-bands FWER 均保持不变；只把 endpoint field scorer 换成 frozen gradient field，有完整 shared A/B 时使用 shared maxAB，否则使用 own maxAB。

**关注点**：gradient field 可用分母为 n=19，E916 因 `axis_not_available` 排除；这属于新轴的可用性边界。3/7 个频带通过 FWER。delta_HYP_slow: pFWER=0.000999, Δ=0.058; theta_preictal_PAC: pFWER=0.003996, Δ=0.048; alpha_sharp_leq13: pFWER=0.1209, Δ=0.031; beta_LVFA_low: pFWER=0.4935, Δ=0.018; gamma_LVFA: pFWER=0.02098, Δ=0.040; hg_low_ripple: pFWER=0.985, Δ=0.002; ripple_high: pFWER=1, Δ=-0.018。

### gradient_multiband_significance_original_f2_fixed_sigma.png / gradient_multiband_significance_original_f2_fixed_sigma.pdf

这是旧 narrow F2 的 fixed-sigma gradient-field 替换版。五类原始重叠窗、`ictal_fraction≥0.5`、window→seizure→subject 折叠、旧 endpoint 二维坐标上的 spatial-null 分组、1000 次置换及七频带 FWER 全部保留。每名患者只从结果无关的 selected A plane 读取一次最近邻带宽；shared 患者使用 shared-plane sigma，own fallback 患者使用 own-A sigma，并强制 TA、TB、全部频带、窗口与 null draw 共用该值。

**关注点**：n=19（E916 gradient axis 不可用），3/7 个频带通过 FWER。delta_HYP_slow: pFWER=0.000999, Δ=0.064; theta_preictal_PAC: pFWER=0.002997, Δ=0.057; alpha_sharp_leq13: pFWER=0.1279, Δ=0.031; beta_LVFA_low: pFWER=0.003996, Δ=0.055; gamma_LVFA: pFWER=0.08292, Δ=0.035; hg_low_ripple: pFWER=0.982, Δ=0.002; ripple_high: pFWER=1, Δ=-0.020。

### gradient_multiband_significance_original_f2_fixed_sigma_shared_only.png / gradient_multiband_significance_original_f2_fixed_sigma_shared_only.pdf

这是 fixed-sigma gradient F2 的 shared-only subgroup 版本。分母在读取频带结果前由 frozen routing 的 `field_plane=shared` 固定为 n=8；own fallback 11 人不进入 delta、cohort median 或任何 FWER null draw。原五类窗口、window→seizure→subject 折叠、endpoint-plane spatial-null 分组及 1000 次七频带 max-T FWER 均继承自 canonical fixed-sigma run。

**关注点**：4/7 个频带通过 shared-only FWER。delta_HYP_slow: pFWER=0.000999, Δ=0.119; theta_preictal_PAC: pFWER=0.09091, Δ=0.075; alpha_sharp_leq13: pFWER=0.01099, Δ=0.097; beta_LVFA_low: pFWER=0.02897, Δ=0.088; gamma_LVFA: pFWER=0.000999, Δ=0.115; hg_low_ripple: pFWER=0.2278, Δ=0.062; ripple_high: pFWER=0.2328, Δ=0.062。该图是预先按 field availability 定义的 subgroup/sensitivity，不替代 n=19 主分析。

### gradient_multiband_significance_original_f2_fixed_sigma_cohort_matched_n17.png / gradient_multiband_significance_original_f2_fixed_sigma_cohort_matched_n17.pdf

这是 canonical n=19 fixed-sigma gradient F2 的 cross-panel cohort-matched sensitivity。分母在读取七频带 cohort 结果前，由 `clinical_onset_gradient_field_cohort_stat_subject.csv` 的 `group_id=all_phenotype_matched` 独立固定为 n=17；随后仅在这 17 人内重新计算 cohort median 和七频带 Westfall–Young max-T FWER。E583 与 Yuquan zhangkexuan 因不在该 cross-panel phenotype-matched group 而排除；E139 与 E1146 明确保留。

**关注点**：2/7 个频带通过 cohort-matched FWER。delta_HYP_slow: pFWER=0.005994, Δ=0.050; theta_preictal_PAC: pFWER=0.003996, Δ=0.058; alpha_sharp_leq13: pFWER=0.1179, Δ=0.032; beta_LVFA_low: pFWER=1, Δ=-0.013; gamma_LVFA: pFWER=0.08891, Δ=0.034; hg_low_ripple: pFWER=0.981, Δ=0.002; ripple_high: pFWER=0.995, Δ=-0.002。这不是把两名排除者描述成“单杆/单 shaft”患者的几何剔除，名单也不是按本图 multiband concordance 结果挑选；它由预定义的 T_spectral 频谱表型合同给出，仅用于跨 panel 同分母敏感性，不替代 n=19 主分析。
