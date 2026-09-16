# Clinical SOZ 触点三维空间紧凑性（2026-08-19）

> 2026-08-20 补充：完成既有 28 人 real-geometry cohort 与本分析 29 人 clinical-SOZ-coordinate cohort 的逐患者分母审计，并在图中加入 cohort-vs-null 括号、星号和精确 P 值。

## 1. 科学问题与结论边界

本分析回答：在患者自身的植入几何中，clinical SOZ 触点是否比等数量随机颅内触点更集中。

它不直接回答：参与间期群体事件的触点是否更集中、传播 endpoint 是否落在 SOZ，或 SOZ 是否构成单一连续脑区。因此本文只使用“clinical SOZ 触点在三维接触点空间中更紧凑”，不单独据此写“群体事件局限于致痫网络”。

## 2. 锁定分析合同

- 坐标：患者内三维 invasive-contact 坐标；禁止 voxel-index fallback。Yuquan 使用患者原生毫米坐标，Epilepsiae 使用 MRI affine 转换后的 MNI152 1 mm 坐标。所有比较只在患者内进行，不混合不同坐标空间。
- 独立统计单元：subject。
- 纳入条件：至少 2 个可映射 SOZ 触点、至少 1 个可映射 non-SOZ 触点，且已标注 SOZ 的坐标覆盖率不低于 80%。
- 主指标：SOZ 触点到其自身质心的 RMS 半径

  \[
  R_{\mathrm{SOZ}}=\sqrt{\frac{1}{k}\sum_{i=1}^{k}\lVert x_i-\bar{x}\rVert^2}.
  \]

- 主 null：在同一患者的全部可映射颅内触点中随机抽取 `k` 个触点，`k` 等于该患者可映射 SOZ 触点数；每名患者 20,000 次。
- 植入几何敏感性：按通道名前缀定义 electrode lead/array，在每个 lead/array 内保留观察到的 SOZ 触点数，仅随机其组内位置。该 null 检查效应是否只是 SOZ 落在少数 leads/arrays 上造成。
- 稳健性指标：SOZ 触点两两距离的中位数。
- 患者内经验 P 值：`(1 + # null <= observed) / (B + 1)`，为单侧紧凑性检验。
- Cohort 检验：对每位患者的 `log2(observed / median null)` 做单侧 Wilcoxon signed-rank；中位比值的 95% CI 由患者 bootstrap 给出。
- 随机种子：全局 seed `20260819`，再由 dataset/subject 通过 SHA256 派生稳定患者种子。

## 3. Cohort 与排除

35 名具有 clinical SOZ 标签的患者中，29 名进入主分析：Yuquan 15 名、Epilepsiae 14 名。

这里的 `n=29` 与既有“28 名 real-geometry”不是同一个分母。既有 28 名是从 34 名 masked stable-K2 propagation/model cohort 中筛出的 real-geometry sensitivity arm；其中 Epilepsiae `384`、`620`、`1125` 没有 clinical SOZ 标签，`635` 的 SOZ 坐标覆盖率仅 0.70。当前分析则从 FigS2 对应的 35 名 clinical-SOZ-labelled cohort 出发，只按本分析的坐标合同筛选。两个集合仅重叠 24 名，因此不能为了沿用“28”而删去当前合格患者。主分析保留 `n=29`；within-lead/array 恰为 `n=28` 是因为 `zhangjiaqi` 没有组内置换自由度。

排除 6 名：

- Yuquan `chenziyang`、`gaolan`、`hanyuxuan`、`sunyuanxin`、`wangyiyang`：无可用坐标源。
- Epilepsiae `635`：10 个已标注 SOZ 触点中仅 7 个可映射，覆盖率 0.70，低于预设 0.80 门槛。

within-lead/array 敏感性纳入 28 名；Yuquan `zhangjiaqi` 的 lead/array 内没有可交换位置，因此该敏感性按合同记为不可检验，而非赋予 P 值。

## 4. 正式结果

### 4.1 主分析：全部植入触点 null

- `n=29`，SOZ/null RMS 半径比中位数 `0.375`，患者 bootstrap 95% CI `0.261–0.478`。
- `28/29` 名患者比值小于 1；`27/29` 名患者的患者内经验 `P<0.05`。
- cohort 单侧 Wilcoxon：`P=3.73×10^-9`。
- 两数据集方向一致：Yuquan `15/15` 比值小于 1，Epilepsiae `13/14` 比值小于 1。

### 4.2 植入几何敏感性：within-lead/array null

- `n=28`，SOZ/null RMS 半径比中位数 `0.653`，95% CI `0.530–0.744`。
- `26/28` 名患者比值小于 1；`22/28` 名患者的患者内经验 `P<0.05`。
- cohort 单侧 Wilcoxon：`P=1.23×10^-7`。

效应在保留每个 lead/array 的 SOZ 数量后仍存在，说明主结果不能只用“SOZ 恰好集中在少数电极杆或阵列”解释。但通道名前缀只是 lead/array 的工程代理，并不等于完整的植入规划模型。

### 4.3 稳健性：median pairwise distance

- 全部植入触点 null：`n=29`，比值中位数 `0.356`，95% CI `0.231–0.444`，`29/29` 同向，`P=1.86×10^-9`。
- within-lead/array null：`n=28`，比值中位数 `0.646`，95% CI `0.564–0.723`，`24/28` 同向，`P=6.14×10^-6`。

因此结论不依赖 RMS 半径这一种空间离散度定义。

### 4.4 异质性

结果不是逐患者普遍成立。Epilepsiae `E922` 在两个 RMS null 下均为反方向（all-contact 比值 `1.192`；within-lead/array 比值 `1.312`），`E1146` 在 within-lead/array null 下接近无差异（比值 `1.012`）。正文应报告总体效应和同向人数，不应写成“所有患者均紧凑”。

## 5. 可接受的论文表述

> 在具有足够 clinical SOZ 标注和三维坐标覆盖的 29 名患者中，SOZ 触点相对于患者特异性全部可映射颅内触点 null 呈显著空间紧凑（SOZ/null RMS 半径比中位数 0.375，95% CI 0.261–0.478；28/29 同向；单侧 Wilcoxon P=3.7×10^-9）。保留每个 electrode lead/array 上的 SOZ 触点数后，该效应仍然存在（n=28；中位比值 0.653，95% CI 0.530–0.744；26/28 同向；P=1.2×10^-7），说明结果并非仅由 SOZ 分布在少数 leads/arrays 上造成。

若要与间期群体事件连接，只能把它与 SOZ-AUC 结果并列为两条证据：群体事件相关 HFO 负荷具有 SOZ 富集，而 clinical SOZ 触点本身占据植入 montage 中较局限的三维子集。不能把本分析单独解释为“群体事件触点更集中”。

## 6. 复现与产物

运行：

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/run_soz_spatial_compactness.py --n-null 20000
```

代码与测试：

- `src/soz_spatial_compactness.py`
- `scripts/run_soz_spatial_compactness.py`
- `tests/test_soz_spatial_compactness.py`

结果：

- `results/spatial_modulation/soz_contact_compactness/cohort_summary.json`
- `results/spatial_modulation/soz_contact_compactness/subject_compactness.csv`
- `results/spatial_modulation/soz_contact_compactness/exclusion_inventory.csv`
- `results/spatial_modulation/soz_contact_compactness/per_subject/`
- `results/spatial_modulation/soz_contact_compactness/figures/soz_contact_spatial_compactness.{png,pdf}`
- `results/spatial_modulation/soz_contact_compactness/figures/README.md`

当前图只展示 all-contact 主分析；within-lead/array 敏感性保留在统计文件和本报告 §4.2，不再单独占一个 panel。图中虚线为 `null=1`；Yuquan 与 Epilepsiae 各自右侧的括号分别连接该数据集的中位比值与 null，星号和精确 P 值对应数据集内患者级 `log2(observed/null)` 对 0 的单侧 Wilcoxon 检验。所有括号都不是 Yuquan 与 Epilepsiae 之间的比较。

当前图的身份是 `candidate supplementary figure`；补图编号需通过 `docs/paper_figure_registry.md` 另行锁定，不能占用已有 FigS4 身份。
