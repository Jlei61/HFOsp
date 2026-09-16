#!/usr/bin/env python3
"""Exact frozen-input LIF release threshold, separate from a network stability claim."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/autonomous_recovery_exploration_20260914'

def main():
    params=json.loads((ROOT/'results/topic4_sef_hfo/rate_model_dynamics_validation_v1/reconstruction.json').read_text())['params']
    p=ROOT/'results/topic4_sef_hfo/reset_state_diagnosis_20260911/parents/high_parent.pkl'
    with p.open('rb') as fh:raw=pickle.load(fh)
    print('checkpoint keys',list(raw),flush=True)
    state=raw.get('engine',raw.get('state',raw))
    if 'V' not in state:
        for v in raw.values():
            if isinstance(v,dict) and 'V' in v:state=v;break
    ne=32000;dt=params['dt'];tau=params['tau_m_E'];reset=params['V_reset'];decay=np.exp(-dt/tau)
    geometry=np.load(ROOT/'results/topic4_sef_hfo/historical_manual_hard_native_z_v1/substrate.npz')
    vth=geometry['vtheta'][:ne]
    z=state['slow']['z'][:ne];m=state['slow']['m'][:ne]
    drive=state['I_E'][:ne]-z*state['I_I'][:ne]
    # One free integration step after absolute refractoriness, held inputs.
    threshold_drive=(vth-reset*decay)/(1-decay)
    eta_break=np.maximum(0,drive-threshold_drive)/np.maximum(m,1e-12)
    eta_stop=np.maximum(0,drive-vth)/np.maximum(m,1e-12)
    etas=[.0005,.005,.02,.2,2.]
    records=[]
    for eta in etas:
        current=drive-eta*m
        nextV=current+(reset-current)*decay
        records.append(dict(eta=eta,M_current_mean=float((eta*m).mean()),
            first_free_step_crossing_fraction=float((nextV>=vth).mean()),
            held_current_above_threshold_fraction=float((current>=vth).mean())))
    res=dict(source=str(p),time_s=float(state.get('absolute_time_ms',75500))/1000,
        dt_ms=dt,tau_m_E_ms=tau,reset_mV=reset,background_Vth_mV=params['V_th'],
        background_one_step_current_threshold=float((18-reset*decay)/(1-decay)),
        records=records,eta_break_quantiles=np.quantile(eta_break,[.1,.5,.9]).tolist(),
        eta_stop_quantiles=np.quantile(eta_stop,[.1,.5,.9]).tolist(),
        meaning='The exact first free LIF update under frozen actual synaptic currents and Z/M; not the whole-network Jacobian or a sufficiency proof of termination.',
        caveat='Preventing one immediate spike is weaker than stopping all future firing. Recurrent drive changes after spike suppression. Cells are components of one realization, not independent repetitions.')
    (OUT/'release_margin_audit.json').write_text(json.dumps(res,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(12,5));xx=np.arange(len(etas));width=.35
    axs[0].bar(xx-width/2,[r['first_free_step_crossing_fraction'] for r in records],width,label='Immediate re-spike')
    axs[0].bar(xx+width/2,[r['held_current_above_threshold_fraction'] for r in records],width,label='Tonic drive possible')
    axs[0].set(xticks=xx,xticklabels=[str(v) for v in etas],xlabel=r'$\eta_M$',ylabel='E-cell fraction',ylim=(0,1.03));axs[0].legend(frameon=False)
    for vals,label in [(eta_break,'Break first-step re-spike'),(eta_stop,'Remove tonic drive')]:
        v=np.sort(vals);axs[1].plot(v,(np.arange(ne)+1)/ne,label=label)
    axs[1].set(xscale='symlog',xlim=(0,10),xlabel=r'Required $\eta_M$ at saved M',ylabel='E-cell cumulative fraction');axs[1].legend(frameon=False)
    for ax in axs:ax.tick_params(labelsize=12);ax.xaxis.label.set_fontsize(15);ax.yaxis.label.set_fontsize(15);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();dest=OUT/'release_margin/figures';dest.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(dest/f'release_margin.{ext}',dpi=180)
    plt.close(fig)
    (dest/'README.md').write_text('### release_margin.png\n在实际75.5秒高态快照上，按原LIF离散积分公式计算：不应期结束后的第一步能否再次越阈，以及固定电流是否仍支持未来放电。右图区分打断最高频放电和完全消除固定驱动所需的M增益。**关注点**：这是冻结实际电流的代数诊断，不能代替SNN续跑、终止机制或分岔证明。\n\n### release_margin.pdf\n对应的矢量图。**关注点**：细胞不是独立实验重复。\n')
    print(json.dumps(res,indent=2))

if __name__=='__main__':main()
