# Core A/B周期轨道投影与局部尖点

冻结模型和原周期轨道来自v2–v5；此目录不改变网络、阈值场或原生SNN四状态样本。

- `solve.py`：给定已有完整周期轨道，在固定J求周期边值解，并检查倍网格离点残差。陡峭分支不收敛时不将失败位置认作分岔。
- `peak_exchange.py`：用伪弧长图定位同周期内两个A峰等高的位置；`projection.py`：定位不同轨道的均值/谷值坐标交点。
- `refine_left_folds.py --which 1/2`：将两处近邻周期折点加倍到4096网格。
- `microflip.py --N 2048/4096`：用双模态反周期测试定位PD0；`microflip.py --double`：求实际2T子支。`micro_neighborhood.py`求LP0b至PD0附近轨道及局部Floquet样本。
- `dense_micro.py`：求67个额外周期边值解，将微区折点曲线加密；不使用绘图平滑代替求解。`grid_audit.py`核查临界轨道和左右率模网格收敛，PD模态按反周期边界插值。
- `stability.py PATH --dt .025`：完整延迟历史的RK4/Poincaré横向Floquet计算。临界点另使用`.0125`。
- `transition.py --name NAME --source PATH --g J --dt .025`：给定完整周期历史，改变J后积分8秒；步长核查使用`.0125`。
- `figures.py`：从已保存解生成A/B独立近方形主图、对应四状态、招募区、同J共存波形、均值/峰谷六联图、投影交点、切换与微区临界模态图。无手工跨断口补线。
- `validate_report.py`：检查同轨道min≤mean≤max、同J稳定共存、峰身份交换的非临界性、投影交点的不同轨道身份、网格/步长精度以及PD0和2T支，生成中文报告。

输出：`results/topic4_sef_hfo/core_observable_bifurcation_v6_20260915/`。已完成数值结果后依次运行`figures.py`、`validate_report.py`、`figures.py`；最后一次重画增加依赖已验证谱数据的微区/模态图。Python为`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`，从项目根目录运行。

精确种子路径、参数和残差保存在每个结果JSON，V5原始分支身份保留在`displayed_curve_sequences.json`及逐点`path`。本目录中的PD0是R微区新增标签，沿用右侧PD1–PD3编号。
