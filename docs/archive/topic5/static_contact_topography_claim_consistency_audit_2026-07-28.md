# Topic 5 static contact topography 全文 claim consistency audit

- 状态：`PASS`
- 扫描 manuscript-facing 文件：21
- 命中需审阅短语：20
- unsafe current claims：0

## 冻结口径

> Patient-specific interictal contact topography corresponds to early-ictal energy up to polarity, while ordered GRU shows no detectable independent heldout or cross-state increment.

`abs(rho)` 只表示相同或反向的 contact ordering，不能单独写成 positive replay、shared signed field 或方向性传播。RNN 在当前论文中是 Supplementary boundary control，不是主文机制。

## 分类汇总

- `DIFFERENT_EMPIRICAL_CONTRACT`：3
- `HISTORICAL_MODEL_STAGE`：3
- `SAFE_BOUNDARY_OR_NEGATION`：14

## 产物

- `results/topic5_static_scaffold_fixed_readout_validation/claim_consistency_audit.csv`
- `results/topic5_static_scaffold_fixed_readout_validation/CLAIM_CONSISTENCY_AUDIT.json`

逐条文本、行号和判定理由见 CSV。历史模型文件仍保留用于 provenance，但其标题头已明确标为 superseded/historical，不能作为当前 manuscript source。
