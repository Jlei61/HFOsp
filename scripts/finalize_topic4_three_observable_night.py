"""Assemble completed bounded-search evidence. Never dispatch simulations."""
from pathlib import Path
import sys, json, datetime, subprocess
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT), str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import analyze_topic4_three_observable_bo as a
OUT=a.OUT; A=a.A; D=A/'confirmation_review'; F=D/'figures'; NIGHT=OUT/'overnight_20260914'
REF='bridge_circle_out125_xminus075'; BEST='g2_b01_p01'

def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    from src.topic4_pdf_font_guard import install
    install(); plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':3})
    raw=a.rt.read(D/'confirmation_summary.json'); scores=a.rt.read(D/'score_interpretation.json')
    core=a.rt.read(D/'core_timing_diagnostic.json'); runs=raw['runs']
    assert len(runs)==8 and all(r['score_status']=='SCORABLE' for r in runs)
    fig,axs=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True)
    for i,cid in enumerate([REF,BEST]):
        for j,mode in enumerate(['TA','TB']):
            ax=axs[i,j]
            for r in sorted([r for r in core['runs'] if r['candidate']==cid],key=lambda r:(r['topology'],r['noise'])):
                ev=[e for e in r['events'] if e['mode']==mode and e['B_minus_A_peak_ms'] is not None]
                ax.scatter([e['B_minus_A_peak_ms'] for e in ev],[e['SCL_minus_ICL_ms'] for e in ev],
                    color='#2b79b1' if r['topology']==3711 else '#d78035',marker='o' if r['noise']==849401 else '^',
                    s=16,alpha=.45,label=f"图{r['topology']} / 噪声{r['noise']}")
            target=raw['patient'][mode]['rod_lag_ms']['median']
            ax.axhline(target,color='black',ls='--',label=f'患者{mode}中位数 {target:.2f} ms')
            ax.axvline(0,color='gray',lw=.7);ax.grid(alpha=.15)
            ax.set(title=f"{'固定参考' if i==0 else '联合1-1'} · {mode}",xlabel='右核B−左核A：聚合发放峰差 (ms)',ylabel='SCL−ICL：参与触点质心中位差 (ms)',xlim=(-100,100),ylim=(-50,130))
            ax.legend(fontsize=7,loc='upper left')
    fig.suptitle('核的先后与电极时序有关，但两个标签不等于两条患者路径\n每点为既有确认轨迹中的一个事件；正横轴为A先达峰，负横轴为B先达峰；不是起燃或因果检验',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93))
    for ext in ['png','pdf']:fig.savefig(F/f'confirmation_core_peak_and_rod_timing.{ext}',dpi=145)
    plt.close(fig)
    readme=F/'README.md';text=readme.read_text()
    if '### confirmation_core_peak_and_rod_timing.png' not in text:
        readme.write_text(text+'\n### confirmation_core_peak_and_rod_timing.png\n固定参考与联合1-1的四个新拓扑/噪声单元，逐事件比较两核聚合发放峰差和两杆参与触点质心差，患者TA/TB中位数同图保留。颜色表示拓扑，形状表示噪声；2 ms峰值来自既有1 ms聚合轨迹，不是起始位置或因果方向。\n**关注点**：TB中存在两核不同先后的子群；较短跨杆差不能仅凭标签视为患者路线恢复。\n')
    # A readable result table preserves the run as the statistical unit.
    lines=['|条件|拓扑|噪声|事件 TA / TB|联合误差 J↓|TA 杆间差 ms|TB 杆间差 ms|','|---|---:|---:|---:|---:|---:|---:|']
    for cid in [REF,BEST]:
        for r in sorted([r for r in runs if r['candidate']==cid],key=lambda r:(r['topology'],r['noise'])):
            lines.append(f"|{'固定参考' if cid==REF else '联合1-1'}|{r['topology']}|{r['noise']}|{r['mode_counts']['TA']} / {r['mode_counts']['TB']}|{r['J']:.3f}|{r['raw']['TA']['rod_lag_ms']['median']:.2f}|{r['raw']['TB']['rod_lag_ms']['median']:.2f}|")
    (D/'paired_results.md').write_text('\n'.join(lines)+'\n')
    tbs=[e for r in core['runs'] if r['candidate']==BEST for e in r['events'] if e['mode']=='TB']
    core_counts={k:sum(test(e['B_minus_A_peak_ms']) for e in tbs) for k,test in [('A_first',lambda x:x>0),('B_first',lambda x:x<0),('tied',lambda x:x==0)]}
    a.rt.write(D/'core_timing_counts.json',dict(candidate=BEST,TB_N=len(tbs),**core_counts,TA_N=330,TA_all_A_peak_first=True,interpretation='Pooled descriptive counts only; paired comparisons remain run-level. Peak order is not causal origin.'))
    pages=[
      ('夜间局部优化：完成，但尚未恢复完整患者双模式',[
       '范围：固定病理基底，仅改变两核XY、核向外EE强度、全局EE方向。',
       '完成：32条件×2训练噪声 + 2条件×2新拓扑×2新噪声，共72槽位。',
       '其中64条新仿真、8条严格相容复用；0工程失败、0 runaway。',
       '训练9,586事件；新拓扑确认1,261事件；每条完整轨迹60秒。',
       '',
       '新拓扑确认：联合误差四运行均值 5.199 → 2.178，3/4配对降低。',
       '改善主要来自均值嵌入A；减项B约0.069 → 0.070，基本不变。',
       '模式频率常数项贡献 1.698 → 0.490；不能把总分降低等同路线恢复。',
       '去除模式频率因素后，TB三组条件距离均值改善，TA三组均值变差。',
       '',
       'TB杆间中位差仍34.38–35.50 ms；患者FIT中位数1.19 ms。',
       '候选TA的SCL和ICL杆内顺序概率误差在四个配对中均变差。',
       '所以接受部分分布拟合改善，不接受完整双模式或Fig5基底。',
       '',
       '09:51停止新增派发；已派发确认11:09左右收尾。G4新增0。',
       '冻结数值触发规则通过；TB时差改善仅0.06–0.24 ms（3个配对）。',
       '它是预算分配规则，不是恢复标准；时间窗口独立禁止追加实验。']),
      ('怎样读这些图，以及本轮实际建立了什么',[
       '实验单位是一个固定条件、拓扑与噪声的60秒运行；事件嵌套其中。',
       '患者FIT提供目标，CAL提供开发性尺度和连续事件块参照；非新患者验证。',
       '三组观测：整体rank；杆内触点对顺序/毫秒时差及杆间差；联合参与。',
       'TA/TB标签用于分布损失；神经元输入不读取标签或细路线。',
       '',
       '算法：4批真实反馈的局部贝叶斯优化，每批4条件、每条件2噪声。',
       '第二批中心移到联合1-1，第三批后搜索半宽0.25缩至0.125。',
       '后三批未超过第一批最佳；不是全局最优或优化算法优越性证明。',
       '',
       '单变量配对已有结果：核向外EE从1.25升至1.375，',
       'TA的SCL顺序误差约+0.139，TB约−0.139，两训练噪声同向。',
       '这揭示模式间取舍；未经额外新拓扑干预，不能称全局稳定效应。',
       '位置与方向也改变局部招募/先后；联合候选不能归因于一个旋钮。',
       '',
       '原生审阅：8条确认各模式最早3事件，48事件的实际固定时点帧。',
       '另看8张Fig2C患者STFT—模型发放包络对照；时间不伸缩、固定分杆。',
       '观察到A/B两端扩散，但TB上部ICL折返和局部包络宽度仍有残差。',
       '抽帧不是逐帧观看全部记录；峰差不是起燃/因果证据。',
       'Agent已目视自查，用户人工验图仍待进行。'])]
    with PdfPages(D/'review_summary.pdf') as pdf:
        for i,(title,items) in enumerate(pages):
            fig=plt.figure(figsize=(11.7,8.3));fig.text(.055,.94,title,fontsize=17,weight='bold')
            fig.text(.055,.875,'\n'.join(items),fontsize=11,linespacing=1.6,va='top')
            fig.text(.055,.035,f'2026-09-15 | 固定三组参数 × 三组观测 | 科学审阅 {i+1}/2',fontsize=9,color='gray')
            pdf.savefig(fig);fig.savefig(F/f'final_review_summary_{i+1}.png',dpi=130);plt.close(fig)
    with PdfPages(D/'native_review_appendix.pdf') as pdf:
        for cid in [REF,BEST]:
            for topo in [3711,3712]:
                for noise in [849401,849402]:
                    folder=A/'native_review'/cid/f'{topo}_{noise}'
                    paths=[folder/f'{cid}_patient_spectra_model_envelopes.png',D/'native_frames'/f'{cid}_{topo}_{noise}.png']
                    for p in paths:
                        with Image.open(p) as im:
                            fig=plt.figure(figsize=(11.7,11.7*im.height/im.width));ax=fig.add_axes([0,0,1,1]);ax.imshow(im);ax.axis('off');pdf.savefig(fig);plt.close(fig)
    parts=[D/'review_summary.pdf',F/'confirmation_report.pdf',A/'patient_count_block_reference/figures/patient_actual_N_confirmation_reference.pdf',F/'confirmation_core_peak_and_rod_timing.pdf',NIGHT/'figures/night_progress_report.pdf']
    subprocess.run(['pdfunite',*[str(p) for p in parts],str(NIGHT/'final_optimization_report.pdf')],check=True)
    a.rt.write(D/'report_assembly.json',dict(time=datetime.datetime.now().astimezone().isoformat(),parts=[dict(path=str(p),sha256=a.rt.sha(p)) for p in parts],output=str(NIGHT/'final_optimization_report.pdf'),native_appendix=str(D/'native_review_appendix.pdf'),physics_changed=False))
    text=readme.read_text()
    for i in [1,2]:
        if f'### final_review_summary_{i}.png' not in text:text+=f'\n### final_review_summary_{i}.png\n本轮完成量、统计单位、拟合改善与传播残差的中文审阅页，对应最终报告第{i}页。它汇总既有轨迹，不新增训练条件或改变评分。\n**关注点**：窗口结束与科学恢复是不同判断；条件性G4未启动。\n'
    readme.write_text(text)
    print('FINAL_REPORT_ASSEMBLED',NIGHT/'final_optimization_report.pdf',flush=True)

if __name__=='__main__':main()
