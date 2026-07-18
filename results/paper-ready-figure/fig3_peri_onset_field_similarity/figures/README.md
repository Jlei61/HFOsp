# Fig3-B peri-onset field similarity

**发布合同**：本目录是 2026-07-18 的兼容性快照；当前权威 trajectory 由上一级 `fig3_peri_onset_run_manifest.json` 指向 immutable `runs/<run_id>/artifacts/figures/`。subset 或中断运行只写自己的 run 目录，不更新这里或 canonical pointer。当前 canonical run 为 `20260718T071020Z_d99c96ec`。

### `<subject>_peri_onset_field_similarity_paper_ready.png / .pdf`

这里仅保留 fingerprint 有效、`shared_a/shared_b` 完整且 `geometry_2d_supported=true` 的二维病例；不回退到 `own_a/own_b`。每个 seizure 使用 `[-120,+20]s` 的 66 个共同窗口（10 s window、2 s step）；Panel a 展示 raw shared-plane `max(|r_A|,|r_B|)` trajectory，Panel b 展示 signed A/B polarity sidecar。浅线是单次 seizure，粗线是跨 seizure median，阴影是 IQR，0 s 虚线标记临床 onset。

**关注点**：这些图是个体级描述性素材，不证明相似度超过 shaft geometry，也不证明 onset 时新出现 scaffold alignment。coverage 必须结合 index 中的 `complete_ok / partial_ok / severely_partial` 阅读；E583 只能作为严重不完整个案，不能承担 polarity 稳定叙述。旧 own/unproven 图、E139 单杆 sensitivity 和旧 own-plane null 已分别移入独立目录，不属于本目录二维证据。
