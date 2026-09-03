# Fig3-A/B 代表性频谱表型 signal context 验收（2026-09-03）

## 1. 结论

Fig3-A/B 是两个描述性频谱表型的代表性对照。A 内并列 `E10 | SZ8` broadband-type 与 Supplementary Fig. 5 已接受的 `E20 | SZ8` gamma-type raw SEEG/TFR 示例；B 用未经断轴的完整时间轴比较二者的四频带能量轨迹。两例患者和代表通道均不同，只用于清楚传达两类可见形态，不是患者内统计检验或 cohort 结论。

## 2. 数据与表型合同

- primary：`epilepsiae_1146` seizure index 7，公开标签 `E10 | SZ8`，`simple_phenotype=broadband_1_150`，classification reason 为 `tspectral_anchored_band_support`。
- comparison：`epilepsiae_635` zero-based seizure index 7，公开标签按正文统一的一基编号写为 `E20 | SZ8`，`simple_phenotype=gamma_nonbroadband`，classification reason 为 `fast_specific_change_point`。
- 两次发作均使用 CAR；E10/SZ8 固定 `SCL9`，E20/SZ8 固定 `HRB1`。表型来源为 `results/topic5_ictal_recruitment/peri_onset_energy_timing/early_spectral_phenotype/per_seizure_spectral_overlap_state.csv`。
- E20/SZ8/HRB1 的 clinical 0–10 s 均值为 gamma `+12.55 dB`、high-gamma `+9.15 dB`、low bands `−1.50 dB`、broadband `−1.14 dB`。该事件沿用 Supplementary Fig. 5 已接受的典型 gamma-dominant morphology，不再使用 TFR 视觉上不典型的 E10/SZ21/POR6。
- baseline-normalized dB 的参考窗仍是 `[-120,-90) s`，early clinical shading 仍是 `[0,10) s`。

## 3. 视觉合同

- A 内部横向并列两个发作示例；每个示例都包含 raw SEEG 与 TFR，只显示 20 s baseline `[-110,-90] s` 与 `[-10,+20] s`。`[-90,-10] s` 不画数据，并在每例 raw/TFR 使用一致的空白间隔和成对斜线断轴。`BASELINE` 按可见 20 s 段居中；断轴不改变 `[-120,-90) s` baseline、TFR 或 band-power 的计算。
- B 保留 low bands、gamma、high-gamma、broadband 2×2 结构，每格叠加两次发作，并连续显示 `[-120,+20] s`，不使用省略号。浅蓝紫 `#8D9FCD` 只表示 broadband-type，青绿 `#62BE9F` 只表示 gamma-type；A 的类型标签沿用相同语义色，频带由小图短标签识别。legend 固定在 low-bands 图左上角的无曲线区、两项纵向排列，只写 `Broadband` / `Gamma`；病例与通道身份保留在 A 标题。四图在 0 s 统一画黑色竖直虚线，左列 ylabel 简写为 `dB`。
- 顶行只重新分配 A/B 宽度；C–F 的位置、尺寸、数据和统计未修改。

## 4. 科学边界

该图可以说明两个代表性发作在各自代表通道上具有不同的早期频谱增强结构。E10/SZ8/SCL9 为宽频增强；E20/SZ8/HRB1 为 gamma-dominant 快活动增强。由于患者与通道均不同，不能把差异归因于发作模式本身；也不能据此推断表型 prevalence、发作机制、患者内稳定性或 cohort-level 差异。

## 5. 验收项

- PNG/PDF 由同一 producer 和同一数据状态生成。
- A 两个内部示例的 raw/TFR 断轴严格对齐，成对斜线位于空白间隔中央且不与 `−90`、`−10` ticks 重叠，最低通道未裁切。
- B 的 legend 不遮挡曲线，颜色没有继续编码频带；时间轴连续且无省略号，两行仍共享各自行内 y limits。
- 完整拼板中 C–F 未重排，匿名标签为 E10/SZ 编号，不显示 E1146 或 E548。
