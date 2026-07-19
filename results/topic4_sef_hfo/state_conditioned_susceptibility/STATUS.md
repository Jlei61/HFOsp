# STATUS — Topic 4 state-conditioned spatial susceptibility (2026-07-19)

> Model-side mechanism/readout diagnostic on the E1146 SNN scaffold. **NOT a seizure, not a patient
> claim.** Base: `codex/topic4-mz-slowvars` @ `66a4d93`. Candidate `zA_q50_tz10000`, seeds 1/3/4.

## 一句话结论

在这个仿真皮层小片里，随着"抑制刹车效力"沿一条固定轨迹慢慢失效、逼近失控点，网络对空间戳一下的短时放大**急剧上升、偏好尺度掉到沿轴大尺度、真正的主本征模式从全局转成沿轴**，三个随机种子高度一致；但放大整体偏"全局"，"沿轴略强于垂直"这点差别是**固定骨架**给的（不是失效图案给的），失效图案决定的是**放大的强度**。所以刹车失效是在**放大一条本来就存在的骨架轴**，不是创造或转动它。

## 三段式朴素话（测了什么 / 怎么测的 / 揭示了什么）

**测了什么**：一个 3.2 万兴奋细胞、按固定带方向骨架连起来的仿真皮层片。戳一下，活动会沿一条固定横轴短暂传一小段再自己停下（间期传播轴）。另有一个慢变量——每个细胞的抑制刹车效力：细胞持续被强抑制时，这个刹车会慢慢失效（去抑制）。沿一条已经算好、固定不变的失效轨迹，网络从反复小事件走向失控（runoff）。我们测：走向失控前，随刹车失效，这张固定骨架"对空间扰动的短时放大能力"怎么变。

**怎么测的**：把失效轨迹重放，在五个时刻（基线、中段、失控前 500/100 毫秒、失控点）抓下每个细胞的刹车效力，铺成一张粗地图，喂进一个已有的率场算子求工作点、线性化，再用一批不同波长/朝向的探针去戳，量 30 毫秒内被放大多少（沿轴 / 垂直 / 全局）。关键是对照：如果把失效图案抹平成同一平均值、旋转 90 度、随机打乱、完全不失效，或者把骨架方向性去掉，放大还一样吗？——如果一样，说明那个效应不是这张真实图案/这个方向带来的。

**揭示了什么**：在这个尺度上，随刹车失效逼近失控，短时放大从约 0.2 涨到约 0.75，偏好尺度从细小变成整片沿轴的大尺度，主本征模式从"全局均匀（globality≈0.99）"转成"沿轴（axis≈0.9）"，三种子高度一致。分两层看：**放大强度**由失效的空间图案决定（真实图案≈抹平平均的两倍，失效正压在轴上两个核团时最强）；**沿轴>垂直**这点方向偏好（约 +0.09）主要由骨架的各向异性给——旋转/打乱失效图案，这点差几乎不变；去掉各向异性（各向同性骨架），这点差减半；完全不失效时没有方向偏好。失控点本身工作点已越过边界（数值上解不出/饱和），与真实网络在失控点走向 runaway 一致。

（内部归档代号：candidate `zA_q50_tz10000`；M4-MZ per-neuron `z_i` 轨迹；M3B finite-Jacobian 率场算子 + 非正规有限时响应 `C exp(J T) B`；probe = Gabor/Fourier 探针；seeds 1/3/4；backdrop `w_ee_mult=1.05, mu_core=0.6`。）

## 完成层次

- **engineering（工程）= 完成**：off-by-default 快照观测器（`mz_slow_vars.py`，6 个受保护引擎文件零改动）；三种子重放 onset 与锁定值**逐 ms 精确一致**（4937.0/4706.8/4861.5），观测器不扰动轨迹（真实 3.2 万神经元 substrate 上 byte-parity 实测），E 细胞界内、I 细胞钉住、z-only ⇒ m≡0，五状态全捕获。
- **numerical（数值）= 完成**：batched vs single 响应差 8e-16；工作点 resolved/saturated/unresolved 严格 fail-closed（saturated/unresolved 一律不给易感性、不贴 axial 标签）；n=8 与 n=12 在失控前一致（沿轴 0.715 vs 0.747）；线性区自检通过。
- **scientific（科学观测）= 完成（模型侧候选）**：见上"揭示了什么"。方向-中性描述用的是 design §11 词表，不设 PASS/FAIL。
- **bridge（到真实网络/患者）= 未完成**：这是**粗率场代理**的结论，尚未在真实 spiking 网络上、经虚拟-SEEG 读出确认；不能外推到患者机制。

## primary observation（三种子，基线 → 失控前 100 毫秒，中位数）

| 量 | Δ 中位数 | per-seed | 说明 |
|---|---|---|---|
| 沿轴放大 axial_gain | **+0.569** | 0.561/0.605/0.569 | 沿轴放大随失效大涨 |
| 垂直放大 perp_gain | +0.468 | 0.454/0.486/0.468 | 垂直也涨 |
| 全局放大 global_gain | **+0.585** | 0.572/0.616/0.585 | 全局涨得最多（放大偏全局） |
| 沿轴−垂直 axis−perp | **+0.106** | 0.106/0.119/0.102 | 沿轴略强于垂直，跨种子稳定 |
| 峰值波数 peak_k | **−5.85** | 全 7.11→1.26 | 偏好尺度掉到大尺度 |

真本征模式：baseline globality≈0.99 / axis≈0.05 → pre-500/pre-100 globality≈0.15 / axis≈0.9（主本征模式随工作点逼近边界从全局转轴向；这是直接量在本征向量上的，不是从响应推的）。

## 复查修正（review 2026-07-19，精度而非结论翻转）

一位仔细的评审指出图叙事顺序问题：没画输入探针/真本征模式/最优输入，且把"沿轴"（输入波矢方向）当成了输出传播方向。已修正——图重构成 **z→本征模式→V1 最优输入→U1 最优输出→G(k∥,k⊥)** 五行；并补严格的非正规分解（对率场传播算子 E→E 块 `C e^{JT} B_E` 做 SVD，不依赖探针字典）：

- **σ1（真正的最大有限时放大，跨所有输入图案）**：0.24→0.28→0.65→**1.01**——真正能放大（>1）的图案只在失控前才出现。
- **渐近本征模式沿轴度**：0.06→**0.90**（强沿轴）；**有限时 30 毫秒最优输出 U1 沿轴度**：0.06→**+0.55**（中等沿轴）。二者的差=非正规系统的关键区别（30 毫秒瞬态尚未对齐到渐近模式）。
- 所以"沿轴放大增强"要分清：**输入 k∥ 偏好上升 + 渐近模式强沿轴**成立；**有限时输出只是中等沿轴**——"响应沿轴传播"在 30 毫秒尺度是中等、非强。本征模式/V1/U1/探针是四个不同对象，不混叫。
- **peak_k 7.11→1.26 是撞轨**（1.26=2π/L=整片尺度，固定 p_max=4 下的域限制），不是找到了真实最优波长；已补 n=8→24 网格收敛检查（见 `convergence_summary.json` + 收敛图）。

## required controls（失控前 100 毫秒，3 种子中位数）

| 对照 | axial | perp | ax−perp | 读出 |
|---|---|---|---|---|
| real | 0.756 | 0.668 | +0.088 | 参照 |
| uniform_mean | 0.363 | 0.307 | +0.056 | real≈两倍 → **空间图案决定放大强度** |
| rotated_90 | 0.436 | 0.347 | +0.089 | 强度掉、**沿轴差不变** |
| spatial_shuffle | 0.405 | 0.322 | +0.083 | 同上 → **方向不由图案朝向给** |
| z_blocked（不失效） | 0.184 | 0.192 | −0.007 | 平 → **没失效就没放大/无方向** |
| AR1_isotropic | 0.65/0.72/0.62 | 0.61/0.66/0.59 | +0.037（中位） | 去各向异性 → **沿轴差减半 → 方向主要由骨架各向异性给** |

（AR1 在失控前仍留 +0.037 的小沿轴差：来自两核沿横轴放置 + 失效横带几何的残余，非核方向偏好来源主体。）

## tests（精确计数）

- `tests/test_mz_slow_vars.py` — **24 passed**（16 原有 + 8 新 snapshot 观测器 Gate B）。
- `tests/test_topic4_mz_slowvars.py` — **18 passed**（观测器改动无回归）。
- `tests/test_topic4_state_conditioned_susceptibility.py` — **12 passed**（Gate C 映射 + Gate D 算子/探针）。
- `tests/test_topic4_m3b_spectral_phase.py` — **81 passed / 7 failed**。7 个失败**全部**是 `results/topic4_sef_hfo/m3b_spectral_phase_map/`（git-ignored 构建产物目录）下 STATUS/verdict/figure 存在性合同测试，`FileNotFoundError`：该目录在 Gate A（本任务开始时全绿）之后被**同一 worktree 内另一并行 session**（`topic4_mz_early_field_bridge` / `mz-onset-dynamics` 线，其未跟踪文件在本 session 中途出现）清掉。**与本改动无关**：我未触碰该目录；我复用的 M3B 算子/本征/有限时机制由 81 个逻辑测试全数验证。需要时 `python scripts/build_m3b_spectral_outputs.py` 可重建这些产物。

## not-run / failed

- **completed**：snapshot 观测器 + 映射/探针模块 + 3 种子捕获 + n=12 real-field atlas + 5 required controls（real/uniform/rotate/shuffle/z-blocked）+ AR1 + n8/n12 resolution + 线性区自检 + 主诊断图 + controls companion 图 + **第二候选 `zA_q75_tz5000` 稳健性（P2；同一冻结背景、不同失控轨迹 onset~9.5s，同模式：放大随失效上升、偏好掉到沿轴低波数、主本征模式转轴向、强度由图案定 real≈3×uniform、方向由骨架定；见 report §4.3 + `second_candidate_sensitivity.json`）**。
- **not_run（P4）**：dirty `topic4-early-readout` worktree 的只读对比（未依赖、未纳入结论）。
- **failed（环境，非本改动）**：上述 7 个 M3B 构建产物合同测试。
- **boundary（设计内）**：onset 工作点 unresolved/saturated（= runoff 边界），故最贴近失控点的易感性未测；primary estimand 端点取最后一个 resolved 状态（失控前 100 毫秒）。

## largest scientific gap & next single step

- **最大缺口**：这是**粗率场代理**上的结论；且固定背景的失控裕度被从 M3B 单核锚点（`w_ee_mult=1.3`）重标到 `1.05`，使代理在失控前保持 resolved、恰在 onset 触界（匹配真实网络间期-至-onset 的裕度）——绝对工作点是代理的自由归一化。沿轴放大的预测**尚未在真实 spiking 网络上确认**。
- **下一步唯一建议**：在真实 SNN 上、用捕获的失控前慢状态（vs 基线）给源核一个 kick，经虚拟-SEEG 平面读出有限时 E 场响应，检验"沿轴放大随失效上升"是否如代理预测——把率场代理桥到 spiking 网络（design 明确的 bridge 缺口）。

## reproducibility

git `66a4d93`（+ 本地未提交改动）；6 个受保护引擎 SHA 与锁定 MZ 多种子 provenance **逐一匹配**（`kick_probe 5faaedab37ab` …），故重放逐 ms 复现锁定 onset。`snapshot_contract.json` / 每 seed `.json` / `susceptibility_atlas.json` / `control_summary.json` / `numerical_audit.json` 均带 schema 版本、上游路径、git/engine SHA、config、candidate/seed/state 列表。
