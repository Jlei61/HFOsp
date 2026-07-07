# Fig3-B peri-onset field similarity

### `<subject>_peri_onset_field_similarity_paper_ready.png / .pdf`

这类图把单个 subject 的合格 seizures 限定在 -120 到 +20 s 的共同 10 s 时间窗内,并以 2 s 步长滑动。Panel a 显示 `max(|r_A|, |r_B|)` 的 sign-free scaffold similarity; Panel b 分别显示 template A 和 template B 的 signed similarity。浅线是单次 seizure,粗线是跨 seizure median,阴影是 IQR; 0 s 虚线标记临床 onset。诊断用的方差和 seizure 数不放在图面,写入 summary JSON。

**关注点**:Panel a 回答发作前能量场是否像 A/B 任一间期模板;Panel b 检查这种相似性是否具有稳定 polarity。加入 +20 s 后只解释 onset 附近早期变化,不解释完整发作期轨迹;完整发作期仍需要 duration warping。
