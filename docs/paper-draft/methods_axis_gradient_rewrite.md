# 共同病理传播轴：D_AB 三维梯度定义

> 状态：建议替换稿。用于替换原草稿中的“空间端点与病理轴投影”和“共同病理传播轴”定轴部分。
> 一句话方法论：用全部有坐标的 A/B 联合触点拟合连续的 A/B 相对早晚偏好梯度，以该三维梯度定义轴；极端三分位触点仅用于显示两极和报告两极间距。

## 可直接替换的方法正文

### 共同病理传播轴

共同病理传播轴由模板 A 与模板 B 在全部联合触点上的相对早晚偏好估计。对同时具有模板 A rank、模板 B rank 和有效三维坐标的触点 \(i\)，分别记其模板 rank 为 \(r_A(i)\) 和 \(r_B(i)\)。由于较小的 rank 表示较早参与，我们首先在联合触点内将两个模板的 rank 分别标准化，并定义触点在模板 \(T\in\{A,B\}\) 中的早参与偏好：

\[
e_T(i)=-\frac{r_T(i)-\overline{r_T}}{\operatorname{sd}(r_T)}.
\]

由此，较大的 \(e_T(i)\) 表示触点在模板 \(T\) 中相对更早。每个联合触点的 A/B 相对早晚对比定义为

\[
D_{AB}(i)=e_A(i)-e_B(i).
\]

\(D_{AB}(i)>0\) 表示该触点相对于模板 B 更偏向在模板 A 中早参与，\(D_{AB}(i)<0\) 表示相反。该定义使用两个模板共同有效的触点，并避免将任一单模板的绝对 rank 直接解释为患者内共同空间轴。

对 \(n\) 个具有有效坐标的联合触点，记其三维坐标为 \(x_i\in\mathbb{R}^3\)，对应的标量值为 \(y_i=D_{AB}(i)\)。坐标和 A/B 对比值分别去中心：

\[
\overline{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,\qquad
\overline{y}=\frac{1}{n}\sum_{i=1}^{n}y_i,
\]

\[
X_c=
\begin{bmatrix}
(x_1-\overline{x})^\top\\
\vdots\\
(x_n-\overline{x})^\top
\end{bmatrix},
\qquad
y_c=
\begin{bmatrix}
y_1-\overline{y}\\
\vdots\\
y_n-\overline{y}
\end{bmatrix}.
\]

随后通过最小二乘拟合 A/B 相对早晚偏好在三维接触点空间中的线性梯度：

\[
\widehat{\beta}
=\arg\min_{\beta\in\mathbb{R}^3}
\left\|y_c-X_c\beta\right\|_2^2
=X_c^{+}y_c,
\]

其中 \(X_c^{+}\) 表示 Moore–Penrose 伪逆，\(\widehat{\beta}\) 为 \(D_{AB}\) 在三维接触点空间中的梯度向量。共同病理传播轴的单位方向定义为

\[
\widehat{u}_{AB}
=\frac{\widehat{\beta}}{\|\widehat{\beta}\|}.
\]

该方向按照 \(D_{AB}\) 增大的方向取向，因此由偏向模板 B 早参与的一极指向偏向模板 A 早参与的一极。若数值实现中出现符号不确定，则选择使触点轴向投影与 \(D_{AB}\) 呈正相关的方向。若 \(D_{AB}\) 在联合触点间无变异，或 \(\|\widehat{\beta}\|\) 低于预设数值阈值，则不定义传播轴。

每个触点相对于该轴的轴向位置定义为

\[
s_i=(x_i-\overline{x})^\top\widehat{u}_{AB},
\]

其到轴线的垂直距离定义为

\[
d_i=
\left\|
(x_i-\overline{x})-s_i\widehat{u}_{AB}
\right\|.
\]

因此，轴向坐标 \(s_i\) 由全部联合触点共同决定，且不依赖预先选定的 source、sink、端点数或 \(k\) 值。后续二维 contact-plane 投影以 \(\widehat{u}_{AB}\) 作为轴向方向，并在去除轴向分量后的三维残差上估计横向主方向。

为显示 A/B 相对早晚偏好的两个空间极端，我们将联合触点按 \(D_{AB}\) 从小到大排序，并分别取最负和最正的三分之一触点。两组触点的空间质心记为

\[
\mu_B=\frac{1}{|\mathcal P_B|}
\sum_{i\in\mathcal P_B}x_i,\qquad
\mu_A=\frac{1}{|\mathcal P_A|}
\sum_{i\in\mathcal P_A}x_i.
\]

两极分离距离定义为

\[
L_{\mathrm{poles}}=\|\mu_A-\mu_B\|.
\]

\(\mu_A\)、\(\mu_B\) 和 \(L_{\mathrm{poles}}\) 仅用于可视化及报告两极空间分离，不参与 \(\widehat{\beta}\) 或 \(\widehat{u}_{AB}\) 的估计。绘图时，将两极质心分别投影到已经拟合的梯度轴：

\[
p_T=\overline{x}
+\left[(\mu_T-\overline{x})^\top\widehat{u}_{AB}\right]
\widehat{u}_{AB},
\qquad T\in\{A,B\}.
\]

轴箭头由 \(p_B\) 指向 \(p_A\)，因而始终与 \(\widehat{u}_{AB}\) 平行；原始两极质心 \(\mu_A\) 和 \(\mu_B\) 作为独立标记显示，并用 \(L_{\mathrm{poles}}\) 报告两极的实际三维分离距离。

### 轴估计的数值质量控制

轴估计至少需要 6 个具有有效坐标的联合触点。每名患者同时记录 \(n\)、\(D_{AB}\) 标准差、\(\|\widehat{\beta}\|\)、线性拟合 \(R^2\)、\(X_c\) 的矩阵秩和条件数。若 \(X_c\) 非满秩，则 \(\widehat{u}_{AB}\) 解释为在已采样接触点子空间内的最小范数梯度方向，而不表述为完整三维组织中的唯一梯度。

**病态截断（2026-07-13 实现）**：仅当 \(X_c\) 严格奇异时才出现"非满秩"，但**近共线**（如仅一根电极杆参与）时 \(X_c\) 虽满秩却极度病态，直接 Moore–Penrose 会在采样极差的垂直方向上过拟合 \(D_{AB}\)，得到与电极几何几乎垂直的伪轴（实测 E139 单杆：\(R^2=0.97\) 但轴垂直于电极）。因此最小二乘按相对奇异值阈值 `RCOND=0.05` 截断：展布小于最大展布 5% 的方向不参与梯度。这实现 spec 所述"已采样子空间内最小范数梯度"，使单杆患者的轴退回为沿电极杆方向；对良态多杆患者（条件数小）不产生截断、轴不变。每名患者额外记录 `effective_rank`（截断后保留的方向数）。

为区分稳定的跨触点空间组织与由单根电极杆采样产生的退化梯度，患者级质量控制进一步记录 \(D_{AB}\) 的空间自相关、杆内方差比例、参与电极杆数，以及 leave-one-shaft-out 或触点 bootstrap 后轴方向与全数据轴方向的余弦相似度。这些量用于判断轴的可读性和采样依赖性，不改变 \(\widehat{u}_{AB}\) 的定义。

**单杆退化患者处理（2026-07-13 用户定）**：`effective_rank=1` / `n_shafts=1` 的患者 D_AB 跨杆轴退化为杆内梯度。政策 = **保留但标记/分层**：这些患者仍进入分析，但结果按"跨杆 vs 单杆退化"分层报告，不因单杆而丢数据。跨杆病理轴的强解释只落在多杆患者。**通道池 = narrow 主 + broad 敏感性**（broad 多杆患者更多、跨杆轴更良定义；narrow 与冻结 source→sink maxAB 同池、可 before-after 对比）。

## 与当前实现的对应关系

| 环节 | 当前实现 | 审阅结论 |
|---|---|---|
| \(e_A,e_B,D_{AB}\) | src/topic5_scaffold_ab_contrast.py::build_D_AB | 已实现；对 joint-valid rank 分别 z-score 后取 eA−eB |
| 联合触点与坐标 | scripts/plot_topic5_dab_axis_subject.py::compute | 已实现；先 joint_valid，再保留有坐标触点 |
| 三维最小二乘梯度 | 同一 compute 函数中的 np.linalg.lstsq(Xc, yc) | 已实现原型 |
| 轴取向 | 以 along 与 D_AB 正相关为正方向 | 与 A-lead 正极定义一致 |
| 极端三分位质心 | k=max(2, floor(n/3)) | 已实现，且不进入 beta 拟合 |
| 正式 geometry/readout producer | run_propagation_skeleton_geometry.py 与 run_contact_plane_readout.py | 尚未迁移；仍使用 source/sink 端点质心轴 |
| 图中轴箭头 | 当前连接正负三分位质心 | **需要修改**：箭头方向必须平行于 beta；质心只能决定显示跨度 |
| 数值诊断 | 当前有 R²、Moran's I、within-shaft fraction | 缺 matrix rank、condition number、轴 bootstrap/leave-one-shaft-out 稳定性 |
| 自动化测试 | 未发现针对 D_AB 三维梯度轴的独立测试 | 需要补旋转/平移不变性、符号取向、退化坐标和极端质心不影响定轴测试 |

> **与单模板传播轴的方向合同不要混用。** 本文的 `u_AB` 是 `D_AB` 对比轴，正号按 A-leading contrast 定义；它不是模板 A 或 B 的 early→late 传播正向。对单模板 `e_T=-z(rank_T)`，`gradient(e_T)` 指向晚→早，故传播向量必须写成 `u_T=-gradient(e_T)/||gradient(e_T)||`。producer 中单模板 `u` 只允许表示早→晚，原始梯度另存为 `earliness_gradient_u`。

## 实现 TODO

1. 将 D_AB 梯度轴抽成 src 下的纯函数，避免只存在于绘图脚本。
2. 让正式三维 geometry 和二维 contact-plane producer 显式选择并记录 axis_definition=dab_gradient_v1。
3. 重新计算 axis projection、横向残差、field、null、cohort summary 和所有依赖轴的图。
4. 修正灰色轴箭头，使其严格平行于 \(\widehat{u}_{AB}\)；另画 \(\mu_A\) 与 \(\mu_B\) 及其原始分离距离。
5. 增加 matrix-rank、condition-number、bootstrap/leave-one-shaft-out 诊断和 fail-closed eligibility。
6. 在新旧轴上做同一队列的 sensitivity comparison；锁定后再删除正文中的 endpoint-based 定轴描述。
