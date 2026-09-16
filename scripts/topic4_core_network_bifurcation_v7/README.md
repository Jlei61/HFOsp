# Core 网络分岔与原生放电对应 v7

冻结 v2–v6 模型与拓扑2511、阈值深度1。新增输出位于 `results/topic4_sef_hfo/core_network_bifurcation_v7_20260916`；科学任务与条件编号见 `execution_plan.md`。

- `extend.py` / `continue_fast.py`：继续已有早期周期族；`scan_folds.py`：按弧长切向过零精化周期折点。
- `flip_local.py`：长周期反周期谱采用更大的特征子空间；`extend.py flip2 --N 4096/8192`：定位2T→4T临界点。
- `double_branch.py`：幅度约束的真实倍周期边值解；不把冗余切向求解失败当作轨道不存在。较大的试探振幅和未收敛结果保留在单独目录，不进入被验证分支。
- `homoclinic_tail.py` / `homoclinic_audit.py`：固定更长周期求轨道、检验接近同一鞍点及参数指数收敛；两套傅里叶网格核查。所得HC为长周期极限证据，尚非无限时长连接轨道的直接求解。
- `native.py`：新增原生SNN与真实spike记录器；初始100ms与旧执行器逐位比对，全部E/I计数与六群体分区守恒检查。本轮最初14条批量运行、19/20单独补充；最终`--batch`入口包含全部16条新条件，并跳过已完成输出。
- `correspondence.py`、`fixed_arc.py`、`condition20arc.py`：与同编号原生SNN一致参数的周期/平衡读出；陡峭区域用弧长图求交代替失效的固定J Newton。
- `gallery.py`：每个原生条件独立的全网络/A/B放电率+raster及300ms放大图；`figures.py`：近方形A/B主分岔图及同编号确定性波形。`onset_figures.py` / `global_figures.py`：新增起振区与远端周期族诊断。
- `critical_gallery.py` / `critical_table.py`：12个周期临界轨道的全网络/E/I读出、A–B相图、左右率模及14项临界结果表（含低率平衡折点、HC极限估计）。
- `shifted_monodromy.py` / `shifted_floquet.py`：对强增长的变分方程作指数变量替换；改变指数移位与积分步长核查领先增长率，不解释失准的次主乘子。
- `analytic_gains.py` / `analytic_poincare.py` / `analytic_critical_check.py`：解析微分同一冻结传递函数及其数值求积，并复核已存临界模。`long_tail_validation.py`区分鞍点附近有限差分增益误差造成的假失稳与真实稳定性；原相位和burst相位、两种积分步长均保留。
- `audit.py` / `fold_validation.py` / `secondary_audit.py` / `homoclinic_audit.py` / `validate.py`：原生记录器、轨道、左右模、局部临界与交付范围验证。`figure_qa.py`检查PNG/PDF可读性、页边界和40/24/12页图册，并在实际生成后写完整逐图说明。
- `report.py`从完成的结果生成中文科学报告和archive入口；数值PASS不表示远端周期族的全局连接已经穷尽，也不表示通过了用户人工图检。

Python为 `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`，从项目根目录运行。`common.py`在NumPy/SciPy导入之前限制BLAS线程。原生SNN模块不导入确定性模型的 `model` 模块，以免与旧执行器同名模块冲突。

模型层次：确定性波形没有人为生成raster；编号1–4是旧SNN记录（2ms占据采样），5–20是新记录（0.1ms积分时刻spike）。原有1–4主图均值保留2–20秒坐标；新结果另统一提供2–12秒比较窗。新SNN样本仅为单拓扑、单噪声的参数对应图库，不构成原生随机网络的分岔证明。

只重画已完成结果的顺序为：`gallery.py` → `figures.py` → `critical_gallery.py` → `onset_figures.py` → `secondary_figures.py` → `global_figures.py` → `figure_qa.py`。绘图脚本顺序运行，避免共享图说明元数据同时写入。`figures.py --main-only`只重画8张主/局部参数对应图；`--waveforms-only`只重画24页确定性波形。
