# Topic 5 static contact topography Supplementary closeout v1.0

## 1. 目的

本合同只负责把当前已完成的 RNN/static-field 分析收束成可进入论文 Supplementary 的证据包。
它不等待外部复制，也不再训练、选择或解释新的模型。

唯一中心结论是：

> Interictal contact participation delineates a patient-specific spatial
> organization that shows orientation-free correspondence with early-ictal
> broadband energy in the current dataset; recurrent order modelling provides
> no detectable held-out or cross-state increment over strong controls.

## 2. 术语

- `interictal contact topography`：train-only interictal participation；
- `orientation-free spatial correspondence`：允许同序或逆序的 \(|\rho|\) 对应；
- `signed spatial correspondence`：预先固定正方向的 Spearman \(\rho\)；
- `recurrent-order increment`：full GRU 相对 rank-shuffle、first-order 或强静态估计器的增量；
- `one-confound-at-a-time sensitivity`：一次只调一个 contact covariate，不称为因果校正。

不再把本结果称为 `shared field`、`replay`、`directional scaffold` 或 `latent mechanism`。

## 3. 冻结证据

### 3.1 纯间期

- formal cohort：34 人，3 seeds；
- full vs rank-shuffle heldout NLL：未建立增益；
- matched order perturbation：支持模型使用顺序；
- 两项必须并列报告，不能互相替代。

### 3.2 跨状态

- strict clinical-onset cohort：16 人、106 seizures；
- target：clinical onset 后 `[0,10] s`、`1–150 Hz`；
- 当前预设 signed direction：未建立；
- orientation-free morphology：在 all-contact、within-shaft 和 geometry-smooth null 下保留；
- raw/best regularized field：已复现同类 morphology；
- GRU-specific static increment：未建立；
- target 已多轮读取，只能称 same-dataset internal validation。

## 4. 论文位置

- 主文：最多一句 bounded computational result；
- Supplementary Methods：数据切分、target-free baseline、teacher/free、spatial null、统计；
- Supplementary Results：按 order use、heldout gain、signed direction、orientation-free morphology、
  model increment、confound sensitivity 的顺序；
- Supplementary Discussion：shared spatial organization 不等于 shared trajectory；
- Supplementary Figure 6：六块固定图。

当前 manuscript-facing source：

`docs/paper-draft/figure6_static_contact_topography_bounded_result.md`

所有更早的 Figure 6 RNN 文稿标记为 historical model-stage records。

## 5. Figure 6

图题冻结为：

> **Interictal contact topography shows static cross-state correspondence
> without a detectable recurrent-order gain**

Panel D 必须写明 \(|\rho|\) 同时容许同序与逆序，不等于 positive replay。Panel F 必须写明
one-confound-at-a-time sensitivity，不是 multivariable causal adjustment。

## 6. Claim consistency gate

对 manuscript-facing 文档搜索：

- replay；
- predicts seizure propagation；
- shared field；
- latent state transition；
- RNN recovered axis；
- ordered history drives transfer。

每个命中必须被分类为：

1. `SAFE_NEGATION_OR_BOUNDARY`；
2. `DIFFERENT_EMPIRICAL_CONTRACT`；
3. `HISTORICAL_MODEL_STAGE`；
4. `UNSAFE_CURRENT_CLAIM`。

只有 `UNSAFE_CURRENT_CLAIM=0` 才完成收口。

## 7. 完成门

- `FINAL_ACCEPTANCE.json=PASS_WITH_BOUNDED_STATIC_CONCLUSION`；
- current Supplementary Methods/Results/Discussion/caption 完整；
- Figure PNG/PDF/metadata/README 完整且目检通过；
- claim consistency audit 无 unsafe current claim；
- docs/archive index、paper-draft README 和 Figure index 指向 current source；
- 不再在当前 16 人 target 上运行新 readout、亚组或 hidden-state 分析。
