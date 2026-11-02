"""Bounded completion/export, native-panel validation and reproducible delivery."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
SOURCE=Path('/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig4/candidates/snn_layout_20261024/source')
PAIR=HERE/'run02'
D0=HERE/'run03_d0'


def main():
    deadline=time.monotonic()+900
    while not (PAIR/'source_manifest.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Native E did not complete within bounded wait')
        time.sleep(2)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(SOURCE/'mpl'),OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,'-B',str(HERE/'export_native_de.py'),str(PAIR),str(SOURCE/'native_de_data'),'--d0',str(D0)],check=True,env=env)
    for name in ['run_native_de.py','run_native_d0.py','export_native_de.py','finish_native_de.py']:
        target=SOURCE/name
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(HERE/name,target)
    subprocess.run([sys.executable,'-B',str(SOURCE/'render_native_de.py'),'--output-dir',str(SOURCE/'figures')],check=True,env=env)
    subprocess.run([sys.executable,'-B',str(SOURCE/'validate_native_de.py'),'--output',str(SOURCE/'native_de_validation.json')],check=True,env=env)
    m=json.loads((SOURCE/'native_de_data/observables.json').read_text())
    rows=[]
    for arm,label in [('D','D0 zero-core'),('E','E one-core'),('diagnostic_threshold_D','Threshold-only diagnostic')]:
        a=m['arms'][arm]
        rows.append(f"| {label} | {a['E_rate_hz']:.6g} | {a['I_rate_hz']:.6g} | {a['patch_E_rate_hz']:.6g} | {a['patch_I_rate_hz']:.6g} | {a['native_population_events']['global_E']['count']} | {a['native_population_events']['input_patch_E']['count']} | {a['frozen_contact_events_primary']} |")
    report='''# 原生 SNN D/E 候选交付

父图入口为 `de_panel.py::render_de(fig, rectD, rectE)`。rect 是归一化 `(left, bottom, width, height)`，不改变父图画布、字体全局设置或其他 panel。按 320×245 mm 父图、每 panel 147×66 mm 设计；左侧 44.982 mm 正方形阈值降低场，右侧同中心 LFP proxy 与局部 raster。E 红、I 蓝，D0 空 raster 明写 `0 spikes`。

主图 D 是完整 **D0 真零核背景**，E 是既有单核工作点。D0 同时撤去 E 降阈值、core 内 OU/Poisson，以及核内 EE×0.85 和核向外 EE×1.453449592 的权重倍率（均恢复×1）。全局 EE 核的角度与宽度、神经元位置、连接伙伴、时延、GABA 权重、动力学常数和 ROI 外背景输入保留。ROI 内由 E 的 `max(0.95ν+OU,0)` Poisson 输入恢复为 D0 的 `ν` 确定性输入，因此局部均值与波动均改变，并非均值匹配对照。D0 不保留局部随机输入区；虚线圈仅表示与 E 匹配的 readout ROI。E 的实线圈为实际 core。D0 的静默不能称作 E/I 平衡背景。原执行记录关于相同 tonic mean 的措辞在 `D0_frozen_execution_design.json` 原样保留；`D0_design.json` 和主 `design.json` 有明确更正，数值轨迹未变。

另外保留 `diagnostic_threshold_D_native.npz`：只撤降阈值，保留 E 的输入与局部权重。这一诊断与 E 的所有随机到达量逐步一致，绝不能用它替代主图 D0。主图 D0/E 因撤去整个随机输入区而不具有相同外部随机流，二者为完整 core 组件对照，不是纯阈值因果对照。

原生引擎为 `src/snn_engine/kick_probe_core_input_v2.py::simulate_kick`，真实电流型逐细胞 LIF，32,000 E+8,000 I、20×20 mm、dt=0.1 ms。拓扑 seed 2511、噪声 seed 847401。E 直接采用历史 `e1146_core_count_optimization_20260928/canary_candidates/core1.json` 的已存在物理 canary，未按本次 burst 结果筛选、调参或重选 seed；不是优化最佳单核。原始 1 秒 canary 的 E/I rates 与空间活动逐点核验。所有 spike 时间及 ID 均来自引擎，未按 rate 合成。

三条轨迹均完整 10 s。统计与默认显示统一使用 1.5–10 s（8.5 s，去掉预定启动段）；raster 显示同一圆形 ROI 的全部 **847 E+191 I**，按 E/I、原始 ID 排序，不按是否活动选细胞。原始文件保留全网 spike，用于总体统计。LFP 位于 (13.5352235294, 7.7596793568) mm，采样周围 E 突触电流幅值的空间加权均值；同一 LFPRecorder、同一位置、同一纵轴范围，无滤波、幅度归一或逐条件 gain。它是模型 current-LFP proxy，不是实测 SEEG 或已标定的细胞外电压。

| 条件 | 全网 E Hz | 全网 I Hz | ROI E Hz | ROI I Hz | 全局事件 | ROI 事件 | 固定接触 observer primary |
|---|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

事件定义冻结于运行前：10 ms 非重叠窗内不同 E 细胞参与率达到全网 1% 或固定 ROI 10%；≤100 ms 的阈下间隔合并。≤500 ms 记 short burst，更长者单列 prolonged；所有事件时间、持续时间与峰值在 `native_de_data/events.csv`。这些是原生群体事件，不是患者 HFO 定义。另复用历史冻结接触 observer，事件层读出独立报告，不能将其计数与原生事件互换。没有 noise-off 试验，因此“噪声下自发活动”不等于证明噪声对每个事件必要。

E/I 电流从未参与模拟反馈的 observer 每 1 ms 记录，分别给出全 E、ROI E、外部 E、全 I 的兴奋/抑制均值、I/E 比、带符号残差和细胞内绝对残差。兴奋电流包含外源与递归分量，另存全 E/I 的递归电流轨迹。单位为 LIF 方程中的电压等效量，不是 nA；稀疏性不替代电流平衡检验。D0 中 E 的兴奋均值 12.865650752、抑制均值 0，放电为零，支持亚阈值静默，不支持 E/I 平衡。

统计单位是一张网络上的一次有限时窗轨迹，事件不算独立网络重复；不声称稳健跨拓扑结果、患者机制、独立临床验证或原生 SNN 分岔。当前状态 `PENDING_AUTHOR_VISUAL_REVIEW`。

关键文件：`native_de_data/design.json` 是主图物理对照；`D0_applied_physics.json` 与 `E_applied_physics.json` 记录具体实施；`primary_validation.json` 核对主图共享拓扑/变化权重；`threshold_pair_validation.json` 仅属于撤阈值诊断的精确随机输入配对。`observables.json` 包含全量读出与明确期望验收值，`firing_rates.csv`/`events.csv` 是可读表，`export_manifest.json` 给出数组与 metadata 校验和。渲染无需外部数据盘。

新实验目录：
'''+str(HERE)+'''

复现实验使用同目录的 `run_native_de.py --output <新的worktree子目录>`，随后 `run_native_d0.py --reference-run <前者目录> --output <另一个新的worktree子目录>`，再 `export_native_de.py <pair目录> <新的export目录> --d0 <D0目录>`。Python 为 `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -B`，BLAS/OMP/MKL/NUMEXPR 均设 1。完整实验仍需 manifest 所列历史冻结图缓存和配置；已保存源代码快照及哈希，源数据/配置路径不伪装成独立自包含模型包。

首个 run01 在完成前为修正 JSON 非有限质心的导出处理而停止，未用于科学结果。正式 pair 为 run02，D0 为 run03_d0；没有改变物理参数或随机 seed。所有修改仅新文件，当前正式 Fig4 入口与既有源码保留。
'''
    (SOURCE/'NATIVE_DE_README.md').write_text(report)
    (SOURCE/'figures/README.md').write_text('''### native_de_preview.png
原生 SNN 真零核 D0 与既有单核 E，各为左侧阈值场、右上 current-LFP proxy、右下局部真实 spike raster。两条件展示同一时间窗、同一 847 E/191 I ROI、同一 LFP 位置与共同增益；D0 的零 spike 明示。E 红、I 蓝，D0 的虚线圈仅是读出 ROI。
**关注点**：D0 为实际静默背景，不能标成 E/I 平衡稀疏态；E 是否出现短事件须同时核对 observables.json 和 events.csv。

### native_de_preview.pdf
与 PNG 使用同一数据与同一 Matplotlib Figure 导出的 PDF。坐标、文字和 core 轮廓保留矢量，密集神经元和波形图层栅格化。
**关注点**：共同尺度、局部人数和 1.5–10 s 时间窗应与 PNG 完全一致。

### native_de_preview.svg
同一状态的 SVG，供父图检查排版或局部复用。正式父图建议调用 de_panel.py，直接创建原生 axes。
**关注点**：这是一份待作者检查的独立候选，不替换正式 Fig4。
''')
    print('DELIVERY_READY',SOURCE,flush=True)


if __name__=='__main__':main()
