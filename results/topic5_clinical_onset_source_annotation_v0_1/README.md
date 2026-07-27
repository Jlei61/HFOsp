# Clinical-onset source annotation v0.1

本目录只准备逐发作 clinical-onset contact 的盲法人工标注表，不包含也不读取 early-ictal energy 数值。`annotation_registry.csv` 已预填 13 人 71 次发作的 patient/seizure/time metadata，source contacts、两位 reviewer 和 consensus 仍为空。

只有 `consensus_status=CONSENSUS_EXACT` 且 `exact_contact_join_status=EXACT_JOINED` 的 seizure 才能进入 primary transfer。SOZ、患者级 focus、A/B source 和 energy-top contacts 均不能补位。
