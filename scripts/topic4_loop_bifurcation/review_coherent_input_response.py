#!/usr/bin/env python3
"""Review the completed paired input diagnostics, without bifurcation claims."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil,time
import numpy as np
from scipy import fft,sparse
from campaign import ROOT,read,write,sha
from compare_local_correlated_inputs import OUT as LOCAL,SOURCE,N
from measure_coherent_phase_component import OUT as PHASE,L

OUT=ROOT/'coherent_input_response_review'


def main():
    OUT.mkdir(exist_ok=True)
    original=read(LOCAL/'result.json');randomphase=read(LOCAL/'phase_only_followup_v2/result.json')
    coherent=read(ROOT/'high_history_local_coherent_phase_mean/result.json')
    r=[next(x for x in a['rows'] if x['region']=='largest20_errors') for a in [original,randomphase,coherent]]
    values=np.r_[r[0]['native_development_rate_RMS_error_Hz'][:3],r[1]['native_development_rate_RMS_error_Hz'],
        r[0]['native_development_rate_RMS_error_Hz'][3],r[2]['native_development_rate_RMS_error_Hz']]
    labels=['Independent source spectra','Full marginal Gaussian','Joint E/I Gaussian',
        'Fixed power, randomized phase','Full recurrent waveform','Phase mean + Gaussian residual']
    inputs=dict(np.load(LOCAL/'inputs.npz'));cells=inputs['cells'];selected=np.isin(cells,inputs['largest_discrepancy_cells'])
    full=abs(inputs['full_complex'])**2
    diagonal=inputs['diagonal_PSD'];w=np.full(N//2+1,2.);w[[0,-1]]=1
    contributions=np.mean((full[0,selected]-diagonal[0,selected])*w/N**2,axis=0)
    frequencies=np.fft.rfftfreq(N,.0001);top=np.argsort(contributions)[-10:][::-1]
    np.savez_compressed(OUT/'frequency_budget.npz',frequency_Hz=frequencies,
        selected20_cross_source_IE_variance_contribution_mV2=contributions)
    result=dict(status='COMPLETE_LOCAL_INPUT_DIAGNOSTIC_REVIEW',labels=labels,selected20_rate_RMS_error_Hz=values.tolist(),
        largest_positive_frequency_contributions=[dict(frequency_Hz=float(frequencies[i]),variance_mV2=float(contributions[i])) for i in top],
        interpretation='Source correlations explain a large local rate bias. Gaussian random Fourier amplitudes also matter; fixed-power random phases retain further error. Explicit phase mean plus Gaussian residual gives a useful local representation but is still native-data-conditioned.',
        unit='20largest firstresponse E-rate-error cells, selected within59diagnostic targets from one conditional native2s developmentrecord.512 numericalreplicas per condition are not native seeds.',
        limits='Localinputs are compared with a variable-background native record while the local external mean is fixed. Selectedcells are not an independent validation set; this figure certifies neither autonomous closure nor a bifurcation.',
        formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(OUT/'result.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(12.3,5.1),layout='constrained',gridspec_kw={'width_ratios':[1.5,1]})
    colors=['#777777','#bc8651','#786fa6','#53a0ae','#2b5877','#19845c']
    axs[0].barh(np.arange(6),values,color=colors,height=.65);axs[0].invert_yaxis()
    axs[0].set(yticks=np.arange(6),yticklabels=labels,xlabel='Rate RMS error (Hz)',xlim=(0,24),title='A   Local response: 20 selected targets')
    for i,v in enumerate(values):axs[0].text(v+.3,i,f'{v:.2f}',va='center')
    # Illustration is the first prespecified diagnostic target, not a new
    # waveform selected to maximize an apparent phase fit.
    target=int(inputs['largest_discrepancy_cells'][0]);idx=int(np.flatnonzero(cells==target)[0])
    projection=np.load(SOURCE/'cross_spectral_diagnostic_v2/projection.npz')
    actual=projection['reconstructed_recurrent_currents'][0,idx]
    phase=np.load(PHASE/'target_phase_inputs.npz')
    cycle=phase['phase_wave'][0,idx]+phase['mean_recurrent'][0,idx]+phase['cycle_mean_offset'][0,idx]
    lo=10010;tt=np.arange(lo,lo+88);times=(tt-lo)*.1
    axs[1].plot(times,actual[tt],color='#777777',lw=1.2,label='Reconstructed recurrent input')
    axs[1].plot(times,cycle[tt%L],color=colors[-1],lw=1.3,label='22-step phase mean')
    axs[1].set(xlabel='Relative time (ms)',ylabel='Recurrent E input (mV equiv.)',
        title=f'B   Coherent input: cell {target}')
    axs[1].legend(loc='upper center',bbox_to_anchor=(.5,-.20),fontsize=8,frameon=False)
    fig.suptitle('Conditional input diagnostics; not an autonomous branch',fontsize=12)
    figures=ROOT/'figures'
    for ext in ['png','svg']:fig.savefig(figures/f'coherent_input_response.{ext}',dpi=180)
    plt.close(fig);shutil.copy2(__file__,OUT/'producer.py')
    write(OUT/'figure_qa.json',dict(status='RENDERED_NOT_YET_INSPECTED',human_visual_review='PENDING'))
    readme=figures/'README.md';entry='### coherent_input_response.png / coherent_input_response.svg'
    if entry not in readme.read_text():
        with readme.open('a') as f:f.write('\n\n'+entry+'\n同一原生条件轨迹中，59个诊断细胞里的20个最大率误差目标用于比较不同输入近似；512个数值复制不是独立原生种子。左图显示源相关、随机振幅与时间相位结构怎样改变局部预测，右图显示一个预先列出的误差目标的原递归输入和22步相位均值。此图只验证局部输入表示，不是自主闭合或正式分岔。\n**关注点**：不能把用实测输入获得的局部精度写成模型已自行产生该结构；22步周期还受原0.1毫秒离散模型的约束。\n')
    print(dict(status=result['status'],rate_RMS_error_Hz=values.tolist()),flush=True)


if __name__=='__main__':main()
