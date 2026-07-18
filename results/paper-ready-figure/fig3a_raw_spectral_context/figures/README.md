# Fig3-A Raw Spectral Context

### epilepsiae_1146_seizure_07_raw_spectral_context.png / epilepsiae_1146_seizure_07_raw_spectral_context.pdf

这张图使用 `epilepsiae_1146` 的 seizure `7`，在进入 z-ER、field projection 和 maxAB 相似性之前，展示远端 baseline 与 clinical onset 附近的原始发作信号。左侧上排是连续时间轴上的 lagPat 电极原始波形，左侧下排是同一时间轴上的代表性 lagPat 单通道 baseline-normalized TFR；右侧 2×2 小图依次展示同一代表通道 low bands (1-30 Hz)、gamma (30-80 Hz)、high-gamma (80-150 Hz) 和 broadband (1-150 Hz) 相对 baseline 的能量增强轨迹。右侧同一行共用 y 轴范围，数值 ticks 与 dB 标签只放在每行左图。图面不标 EEG onset，也不画 onset 虚线；alpha 与 beta 只保留在 summary JSON 的通道选择审计中。它只承担解释和质控作用，不是 cohort 统计，也不证明 timing-order replay 或机制。

**关注点**：raw SEEG 与 TFR 的时间轴必须严格对齐；baseline 是标准化参考，不等于发作前最后几秒；clinical-onset 阴影表示早期 ictal field input，而不是原始 z-ER 图本身。
