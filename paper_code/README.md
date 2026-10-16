# 论文计算代码

三个 notebook 对应 Fig1、Fig2 和 Fig4–5，各代码块用英文注释说明计算及面板含义。所有实现都在本目录；输入和结果放在旁边。

```text
paper_code/
├── notebooks/          01 Fig1 · 02 Fig2 · 03 SNN/Fig4–5
├── data_processing/    电压、参考、滤波、检测、事件质心
├── analysis/           rank、K-means、统计、投影、空间场
├── modeling/           LIF网络、core输入、目标函数、优化、Z/M
├── plotting/           各面板接口；paper/ 是论文 producer 的本地绘图副本
├── workflows/          输入、结果保存及队列入口
├── config/             E1146 demo参数、SNN基础/论文规模参数
├── tools/              notebook执行、EDF导出、科学计算检查
└── METHODS_AND_PANELS.md

paper_inputs/E1146_demo/ 输入数据
paper_outputs/          新结果和带输出notebook
```

## 运行

将 `E1146_demo.zip` 解压到 `paper_inputs/`，应得到 `paper_inputs/E1146_demo/`。Python 3.11：

```bash
cd paper_code
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/
```

按顺序打开：

1. [01 · 重复顺序（Fig1）](notebooks/01_raw_seeg_to_fig1.ipynb)：从原始电压得到事件与模板，再与随机顺序比较。
2. [02 · 空间信息（Fig2）](notebooks/02_ranks_to_spatial_fields_fig2.ipynb)：计算传播方向，在独立记录上比较有无空间信息，最后投影成场。
3. [03 · 传播模型（Fig4/5）](notebooks/03_snn_model_fit_fig4_fig5.ipynb)：建立网络和拟合目标，分别运行core搜索、方向实验、传播动画与Z/M仿真。可单独运行。

面板编号和方法说明写在对应代码块的英文注释中。参数细节与扩展接口见 [方法说明](METHODS_AND_PANELS.md)，输入和文件保存由 `workflows/tutorial.py` 处理。

命令行执行：`python tools/execute_notebook.py --notebook 01_raw_seeg_to_fig1.ipynb`，替换文件名即可运行另外两个。结果和带全部cell输出的IPYNB/HTML写到相邻 `paper_outputs/`。输入/输出位置可通过 `HFO_PAPER_INPUTS`、`HFO_PAPER_OUTPUTS` 设置。

E1146 demo含4×200秒、15个分析触点的原始电压和完整114通道的CAR参考信号，以及模型需要的19,770个FIT、4,981个CAL观测及坐标。FIT/CAL来自较完整记录，不能当作这800秒短例的检测结果。另含23次临床候选发作的原始片段、完整CAR参考和时间标注，Fig5E/F重新计算频谱、六带阶段、场及选择校正统计。数据包不含旧模型结果、功率场或图片；本机数据包未随代码推送。

SNN默认40,000神经元、12秒/候选，1–4 core各运行2次重启、每次10个候选，共80次搜索仿真，另有13点方向扫描及Z/M轨迹。默认16个worker，CPU完整运行需数小时；内存较少时可减小配置中的max_workers。Z/M最多运行60秒，并保留原高率进入后的停止规则。优化使用原Matérn-5/2 GP、qLogEI及后验不确定性提议；不足8个可评分观测时按原规则继续空间填充。用 `HFO_SNN_CONFIG=snn_paper.json` 扩大到每次31候选、4次重启、60秒/候选；搜索共496次仿真。

本机Python为 `/data/hfosp/envs/paper_code_20261015/bin/python`，启动时需 `LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib`。建议设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1`，避免重复占用CPU。本机命令行导出HTML还需 `JUPYTER_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/share/jupyter`。其他机器不应照搬这些本机路径；联网全新安装尚未验证。

[方法与面板定义](METHODS_AND_PANELS.md)说明每一步计算，[绘图来源](plotting/paper/SOURCES.md)列出对应论文函数。E1146单患者demo展示Fig4I的两个模板相似度；Fig5E/F使用E1146原始发作片段；一位患者不等于13人队列复现。Fig1/2/4/5的完整队列统计和Fig1 MRI仍需相应数据。Fig5按2026-10-11六面板候选的绘图定义整理，未更改论文正式图。科学计算检查：`python tools/check_computation.py`。
