# 论文计算代码

三个 notebook 从原始电压或模型参数开始计算，方法说明放在代码块的英文注释里。所有实现均在本目录，输入和输出放在旁边。

```text
paper_code/
├── notebooks/          01 Fig1 · 02 Fig2 · 03 SNN/Fig4–5
├── data_processing/    参考、滤波、检测、群事件与质心
├── analysis/           rank、聚类、统计、投影、空间场
├── modeling/           LIF网络、core输入、目标函数、优化、Z/M
├── plotting/           面板接口及论文绘图函数的本地副本
├── workflows/          输入和结果保存
├── config/             患者、模型和实验参数
└── tools/              notebook执行和计算检查

paper_inputs/
├── Y1_demo/            Fig1的原始SEEG
└── E1146_demo/         Fig2原始SEEG、SNN患者观测、临床发作片段
paper_outputs/          新计算结果和带输出的notebook
```

## 运行

将 `Y1_demo.zip`、`E1146_demo.zip` 解压到 `paper_inputs/`。Python 3.11：

```bash
cd paper_code
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/
```

三个 notebook 可以独立运行：

1. [01 · Fig1](notebooks/01_raw_seeg_to_fig1.ipynb)：Y1原始电压 → HFO群事件 → 顺序 → K-means → 一致性统计。分析保留26通道，C/E沿用论文的18通道显示及1起算的参与通道rank。
2. [02 · Fig2](notebooks/02_ranks_to_spatial_fields_fig2.ipynb)：从E1146原始电压重新检测，计算rank梯度、空间信息增益及动态/模板场。
3. [03 · Fig4–5](notebooks/03_snn_model_fit_fig4_fig5.ipynb)：构网、积分、定义J与J_joint、优化core、分析波形和传播，再启用Z/M并与临床场比较。

命令行：`python tools/execute_notebook.py --notebook 01_raw_seeg_to_fig1.ipynb`，替换文件名即可运行另外两个。完整输出保存在 `paper_outputs/`；位置可用 `HFO_PAPER_INPUTS`、`HFO_PAPER_OUTPUTS` 更改。

## SNN实验

40,000神经元、20×20mm、dt=0.1ms。连接核从 `l_EE=0.38mm, AR=2` 开始，垂直宽度只乘一次1.5；最终平行/垂直宽度为0.5374/0.4031mm。

短优化默认1–4 core各2次重启、每次10个候选、每次12秒，16个worker。本机并行运行的总内存占用约100GB；内存较少时可在配置中调低 `max_workers`。Fig4E只画这批候选的位置散点及J_joint，不用小样本拟合正式图的位置密度。`snn_paper.json`提供31候选×4重启、60秒/候选的core数量比较配置。

短demo压缩的是搜索预算，仍保留40,000神经元模型。本机一条60秒间期轨迹约需22分钟；完整Notebook 03需数小时，运行时间取决于CPU和并行内存带宽。

Fig4F–H和Fig5重新模拟论文的 `xy_left_20` 工作点；Video2另模拟 `lf_bo04_07`。两条间期轨迹各60秒，Fig5启用同一网络的Z/M并使用原30秒上限和高率进入后的停止规则。短搜索的最优点不会替换这两个论文工作点。代码不读取历史模型轨迹、loss或图片。

Fig5F演示将本次模型与E1146发作进行场比较。论文F来自另一批按患者拟合的模型和13人队列；这里保留本次单患者计算的实际结果，不复制该队列中的点。

## 数据范围

Y1和E1146间期demo各含4×200秒原始电压。Y1按配置做双极参考；E1146另附原完整114通道的CAR参考。模型目标使用E1146的19,770个FIT、4,981个CAL观测；模板从全部46,683个间期观测重新发现，G/H使用其中30,049个development事件。输入包含观测、参与掩码和索引，不含旧模板标签。Fig5临床输入为23次发作的原始片段、时间及基线标注。

这是单患者演示。短窗口不等于整天记录；Fig1 MRI和论文各面板的完整患者队列仍需相应数据。所有临床数据包、本次输出和论文终稿留在本机，不随代码推送。

[方法与面板定义](METHODS_AND_PANELS.md) · [绘图来源](plotting/paper/SOURCES.md)

本机环境：`/data/hfosp/envs/paper_code_20261015/bin/python`，需设置 `LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib`；HTML导出还需 `JUPYTER_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/share/jupyter`。建议将OPENBLAS/OMP/MKL/NUMBA线程数设为1。其他机器使用自己的环境；联网全新安装尚未验证。
