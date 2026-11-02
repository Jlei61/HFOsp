#!/usr/bin/env python3
"""Check complete recovery/re-entry and recorded logistic gates against Fig6."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit

MAIN=Path('/home/honglab/leijiaxin/HFOsp')
ROOT=Path('/data/hfosp/topic4_sef_hfo/logistic_feedback_20261102')
sys.path.insert(0,str(MAIN/'scripts'))
spec=importlib.util.spec_from_file_location('frozen_logistic_observer',
    ROOT/'source/analyze_topic4_unified_current_adaptation.py')
assay=importlib.util.module_from_spec(spec);spec.loader.exec_module(assay)


def write(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def summarize(root,name):
    assay.ROOT=root
    row=assay.analyze(name)
    job=json.loads((root/'jobs'/f'{name}.json').read_text())
    stop=root/'runs'/name/'pilot_stop.json'
    if stop.exists():
        pilot=json.loads(stop.read_text())
        assert row['observed_s']==pilot['observed_s']
        row['run_status']=pilot['status']
        row['pilot_stop']=pilot
    row.update(model=job.get('model'),feedback_kind=job.get('feedback_kind'),
               source_run=str(root/'runs'/name))
    write(root/'analysis'/name/'result.json',row)
    qualified=[]
    for ep in row['episodes']:
        later=[entry for entry in row['temporal']['entries']
               if entry['onset_s']>=ep['observation_end_s'] and
               ep['both_core_Z_reference_s'] is not None and
               entry['onset_s']>ep['both_core_Z_reference_s']]
        if ep['sustained_return'] and later:
            earlier=[e for e in row['temporal']['entries'] if e['onset_s']<ep['exit_start_s']]
            qualified.append(dict(first_entry_s=earlier[-1]['onset_s'],
                exit_s=ep['exit_start_s'],Z_recovered_s=ep['both_core_Z_reference_s'],
                returned_brief_count=ep['brief_count'],returned_brief_span_s=ep['brief_span_s'],
                first_returned_brief_s=ep['returned_brief_events'][0]['start_s'],
                reentry_s=later[0]['onset_s']))
    result=dict(name=name,model=job.get('model'),status=row['run_status'],observed_s=row['observed_s'],
        initial_brief_count=len(row['initial_brief_events']),
        initial_8s_brief_count=int(sum(.5<=e['start_s'] and e['end_s']<=8 and
            .02-1e-9<=e['duration_s']<=.2+1e-9 and not e.get('right_censored',False)
            for e in row['temporal']['events'])),
        has_entry=row['has_high_entry'],has_exit=row['has_autonomous_exit'],
        has_interictal_return=row['has_sustained_return'],qualified_cycles=qualified,
        complete_cycle=bool(row['run_status']=='COMPLETE' and row['observed_s']>=120 and qualified
                            and len(row['initial_brief_events'])>=10),
        first_cycle_complete=bool(qualified and qualified[0]['first_entry_s']==row['temporal']['entries'][0]['onset_s']),
        entries=row['temporal']['entries'],criteria=row['criteria'],
        source_run=str(root/'runs'/name),paired_noise_seed=job['seed'])
    if job.get('feedback_kind')=='logistic_shared':
        inputs=assay.load(root/'runs'/name,'chunks',['inputs'])['inputs']
        reference=Path('/data/hfosp/topic4_sef_hfo/adaptation_history_20261031/runs/seed9108404')
        reference_inputs=assay.load(reference,'chunks',['inputs'])['inputs']
        assert len(inputs)>0
        np.testing.assert_array_equal(inputs,reference_inputs[:len(inputs)])
        a=assay.load(root/'runs'/name,'adaptation_chunks',['time_ms','q','global_s'])
        g=assay.load(root/'runs'/name,'mechanism_chunks',
            ['time_ms','global_E_rate_Hz','global_raw_conductance_ratio'])
        ix=np.searchsorted(g['time_ms'],a['time_ms'])
        np.testing.assert_array_equal(g['time_ms'][ix],a['time_ms'])
        pred=expit((g['global_E_rate_Hz'][ix]-job['logistic_midpoint_Hz'])/job['logistic_width_Hz'])
        error=float(np.max(abs(pred-a['q'])))
        np.testing.assert_array_equal(pred,a['q'])
        np.testing.assert_array_equal(g['global_raw_conductance_ratio'][ix],job['global_gain']*a['global_s'])
        result.update(logistic_midpoint_Hz=job['logistic_midpoint_Hz'],
            logistic_width_Hz=job['logistic_width_Hz'],recorded_gate_max_error=error,
            gate_samples=len(pred),G_equals_Gmax_s=True,input_samples=len(inputs),
            external_input_bitwise_equal=True)
    return result


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig,axes=plt.subplots(4,len(rows),figsize=(4.4*len(rows),9),squeeze=False,
                          gridspec_kw={'height_ratios':[1,1,1,1]})
    maximum_M=maximum_G=0.
    for col,row in enumerate(rows):
        run=Path(row['source_run']);root=run.parent.parent
        r=np.load(root/'analysis'/row['name']/'readouts.npz')
        g=assay.load(run,'mechanism_chunks',['time_ms','global_raw_conductance_ratio'])
        title='Previous Fig6: clip' if col==0 else f"Logistic: k = {row['logistic_width_Hz']:g} Hz"
        if row['observed_s']<120:title+=f"\n{row['observed_s']:g}-s pilot"
        axes[0,col].set_title(title,fontsize=12)
        axes[0,col].plot(r['time10_s'],r['rates10_Hz'][:,0],color='#333333',lw=.45)
        axes[0,col].axhline(200,color='#888888',ls=':',lw=.7)
        st=r['slow_time_s']
        for j,color in [(1,'#b64349'),(2,'#267aad')]:
            axes[1,col].plot(st,r['Z'][:,j],c=color,lw=.8,label=f'Core {"AB"[j-1]}')
        axes[1,col].set_ylim(0,1.04)
        axes[2,col].plot(st,r['adaptation_mV'][:,0],c='#C35A00',lw=.7)
        axes[3,col].plot(g['time_ms']/1000,g['global_raw_conductance_ratio'],c='#0072B2',lw=.7)
        maximum_M=max(maximum_M,float(r['adaptation_mV'][:,0].max()))
        maximum_G=max(maximum_G,float(g['global_raw_conductance_ratio'].max()))
        for ax in axes[:,col]:
            ax.set_xlim(0,120);ax.spines[['top','right']].set_visible(False)
            if row['observed_s']<120:ax.axvspan(row['observed_s'],120,fc='#eeeeee',zorder=-5)
            for e in row['entries']:
                ax.axvline(e['onset_s'],color='#a72c40',lw=.6,ls='--',alpha=.6)
        axes[3,col].set_xlabel('Time (s)')
        axes[1,col].legend(frameon=False,fontsize=9)
    for ax,label in zip(axes[:,0],['All-E rate (Hz)','$Z$',r'$\eta_M M$ (mV)','$G$']):
        ax.set_ylabel(label)
    for col in range(len(rows)):
        axes[0,col].set_ylim(0,500)
        axes[2,col].set_ylim(0,np.ceil(maximum_M/100)*100)
        axes[3,col].set_ylim(0,np.ceil(maximum_G/5)*5)
    fig.tight_layout()
    dest=ROOT/'figures';dest.mkdir(exist_ok=True)
    pdf=dest/'paired_clip_logistic_validation.pdf'
    fig.savefig(pdf);plt.close(fig)
    try:import pymupdf as fitz
    except ImportError:
        sys.path.insert(0,'/tmp/hfosp_fig5_pdf_tools');import pymupdf as fitz
    with fitz.open(pdf) as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(180/72,180/72),alpha=False).save(pdf.with_suffix('.png'))
    rates=np.linspace(0,700,1401)
    fig,ax=plt.subplots(figsize=(6.3,3.4))
    ax.plot(rates,np.clip((rates-200)/300,0,1),'--',c='#333333',label='Previous clip')
    for row,color in zip(rows[1:],['#0072B2','#009E73','#C35A00','#7b61a8']):
        width=row['logistic_width_Hz']
        ax.plot(rates,expit((rates-row['logistic_midpoint_Hz'])/width),c=color,label=f'Logistic: k = {width:g} Hz')
    ax.set(xlabel='All-E rate R (Hz)',ylabel='Feedback fraction S',xlim=(0,700),ylim=(-.02,1.02))
    ax.legend(frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();gate_pdf=dest/'logistic_gate_comparison.pdf';fig.savefig(gate_pdf);plt.close(fig)
    with fitz.open(gate_pdf) as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(180/72,180/72),alpha=False).save(gate_pdf.with_suffix('.png'))
    readme=dest/'README.md'
    previous=readme.read_text() if readme.exists() else ''
    previous='### logistic_gate_comparison.png / .pdf\n\n旧 clip 与四个平滑宽度的 logistic。半最大响应发放率固定，宽度调节过渡区及低活动尾部。\n**关注点**：函数曲线仅说明门控差异，完整状态转变由原生仿真判定。'
    readme.write_text(previous+'\n\n### paired_clip_logistic_validation.png / .pdf\n\n'
        '同一网络、同一噪声种子的旧 clip 与不同宽度的共享 logistic 连续仿真。'
        '从上到下为全体 E 发放率、两个 core 的抑制资源、平均适应电流和全局抑制；每行共用纵轴范围。虚线为持续高活动的进入时刻，灰区为60秒筛查停止后未继续计算的区间。\n'
        '**关注点**：比较完整的发作、终止、间期恢复、再次进入，结果为单网络单种子机制检查。\n')


def main():
    # Read-only link: compute the reference assay into the NEW analysis folder.
    reference=Path('/data/hfosp/topic4_sef_hfo/adaptation_history_20261031/runs/seed9108404')
    alias='clip_reference_s9108404'
    link=ROOT/'runs'/alias
    if link.exists():assert link.resolve()==reference
    else:link.symlink_to(reference,target_is_directory=True)
    job=json.loads((reference/'result.json').read_text())['job']
    write(ROOT/'jobs'/f'{alias}.json',dict(job,name=alias,feedback_kind='clip_reference'))
    roots=[(ROOT,alias)]
    roots.extend((ROOT,p.stem) for p in sorted((ROOT/'jobs').glob('logistic_*.json')))
    rows=[]
    for root,name in roots:
        if not any((root/'runs'/name/file).exists() for file in ['result.json','pilot_stop.json']):
            print(f'Not complete: {name}');return
        rows.append(summarize(root,name))
    write(ROOT/'comparison.json',dict(rows=rows,statistical_unit='one fixed network, paired noise seed9108404',
        any_logistic_complete_cycle=any(r['complete_cycle'] for r in rows[1:]),
        all_parameter_settings_reported=True,population_equivalence_tested=False))
    plot(rows)
    print(json.dumps([{k:r[k] for k in ['name','complete_cycle','qualified_cycles']} for r in rows],indent=2))


if __name__=='__main__':main()
