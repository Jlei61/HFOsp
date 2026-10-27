#!/usr/bin/env python3
"""On experiment completion, render a separate Fig5 candidate with the new F."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=Path('/data/hfosp/topic4_sef_hfo/cohort_core_count_20261024')
PACKAGE=ROOT/'results/paper-ready-figure/fig5'
FROZEN=PACKAGE/'source/vertical_legend_completion_pipeline_20261027'
DEST=PACKAGE/'candidates/ta_preferred_search_complete_vertical_legend_20261027'

def read(p):return json.loads(p.read_text())
def record(**kw):
    p=RUN/'figure_completion.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(updated_unix=time.time(),pid=os.getpid(),**kw),ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while not (RUN/'completion.json').exists():
        status=read(RUN/'status.json')['status']
        if status in ('EXECUTION_FAILED_PARTIAL','SIMULATIONS_COMPLETE_REPORT_FAILED'):
            record(status='EXPERIMENT_NEEDS_ATTENTION',experiment_status=status);return
        record(status='WAITING_FOR_ALL_TRAINING_AND_CONFIRMATION',experiment_status=status)
        if not args.watch:return
        time.sleep(30)
    summary=read(RUN/'template_recovery_summary.json')
    assert summary['status']=='TRAINING_COMPLETE_REVIEW_PENDING'
    assert summary['training_complete']==summary['training_planned']==1600
    if DEST.exists():
        record(status='DESTINATION_EXISTS_REVIEW_REQUIRED',destination=str(DEST));return
    # Export event ranks and model-only labels at the same nominated workpoints.
    # This preserves the historical F statistic and checks it against cached readout.
    subprocess.run([sys.executable, str(FROZEN/'export_selected_mode_events.py'),
                    '--require-complete'], check=True,
                   env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    shutil.copytree(FROZEN/'candidate',DEST)
    # The new constraint can nominate different core1 workpoints. Old nomination
    # replays must not be presented as confirmation of these reselected points.
    subprocess.run([sys.executable, str(FROZEN/'select_fig5_ta_preferred_workpoints.py'),
                    '--parent-candidate', str(DEST), '--output', str(DEST/'reselection'),
                    '--completed', '--workers', '4'], check=True,
                   env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    shutil.copy2(DEST/'reselection/summary.json', DEST/'source/interim_core_count_summary.json')
    shutil.copy2(DEST/'reselection/selection.json', DEST/'source/interim_selection.json')
    shutil.copy2(DEST/'reselection/selection_audit.json', DEST/'source/ta_preferred_selection_audit.json')
    spec=importlib.util.spec_from_file_location('frozen_fig5_renderer',FROZEN/'build_fig5_aligned_layout.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=ROOT;module.EXP=RUN;module.OUT=DEST
    sys.argv=[str(FROZEN/'build_fig5_aligned_layout.py'),'--output',str(DEST)]
    module.main()
    n=len(read(DEST/'source/cohort_selection.json')['included_subjects'])
    (DEST/'README.md').write_text(f'# 新 Figure 5：TA 偏好约束的单 core 对照\n\n原实验全部训练及原提名点复测结束后生成；F 使用 {n} 位 shared axis 患者各32次搜索，在单 core 的有符号 ρTA>ρTB 候选中选最低 J，双 core 保持原最低 J，真实 TA/TB 身份不交换。新选中的单 core 工作点未因此获得独立噪声验证；原提名点的复测不能代替新点，cohort_comparison_complete 仍为 false。\n\n[完整 PDF](figures/fig5-complete-layout.pdf) · [PNG](figures/fig5-complete-layout.png)。这是非对称选点策略的探索性对照，待作者审阅及新选点复测；未覆盖当前图入口。\n')
    p=DEST/'figures/README.md';text=p.read_text();start=text.index('### fig5-panelf.png');end=text.index('### fig5-complete-layout.png',start)
    text=text[:start]+f'### fig5-panelf.png\n冻结名单内 {n} 位 shared axis 患者的真实 TA–MTA、TB–MTB 比较；单 core 在 ρTA>ρTB 约束下按最低训练 J 选一个工作点，双 core 用无约束最低 J，两项相关来自各自同一工作点。空心/实心为单/双 core，横线及竖段为中位数和四分位范围；无合格单 core 候选时保留患者并标记缺失，不交换模板名。\n**关注点**：这是非对称选点策略的探索性对照，TB 相对较低部分来自入选约束；新选点尚待独立噪声复测，不代表纯 core 数量效应。\n\n'+text[end:]
    text=text.replace('F 保留旧队列数据，单 core 实验状态仅在说明文件中记录','F 已接入同预算单/双 core 队列对照');p.write_text(text)
    stats=read(DEST/'core_count_statistics.json')
    comparison='；'.join(f"{v['mode']}：n={v['n_pairs']}，Holm 校正 P={v['p_holm']:.3f}" if v['p_holm'] is not None else f"{v['mode']}：不可估计" for v in stats['comparisons'])
    note='\nF 横轴刻度标签 TA–MTA、TB–MTB 分别位于两组点的中心，无额外轴标题；右上方图例竖排为空心点 1 core、实心点 2 cores，显著性星号及括号下移并与图例留开间距。图内按 Holm 校正 P 标星号：* P<0.05、** P<0.01、*** P<0.001；不显著时不标文字或括号，完整数值保留于统计文件。患者配对、双侧 Wilcoxon 符号秩检验，跨真实 TA、TB 两项作 Holm 校正；P 值仅描述约束单 core 与无约束双 core 的差异，模板相关同时参与选点与评分。'+comparison+'。\n'
    for path in [DEST/'README.md',p]:
        path.write_text(path.read_text()+note)
    record(status='TA_CONSTRAINED_CANDIDATE_READY_REPLAY_PENDING',destination=str(DEST),pdf=str(DEST/'figures/fig5-complete-layout.pdf'),current_pointer_changed=False,cohort_comparison_complete=False)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        record(status='RENDER_FAILED_REVIEW_REQUIRED',error=repr(exc));raise
