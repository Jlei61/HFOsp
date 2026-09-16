#!/usr/bin/env python3
"""Plot the completed paired p030 short regression without assigning event classes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=ROOT/'results/topic4_sef_hfo/substrate_autapse_correction/dynamic_check'
    figures=out/'figures';figures.mkdir(exist_ok=True)
    summaries={arm:json.loads((out/f'{arm}_summary.json').read_text()) for arm in ('legacy','corrected')}
    for key in ('candidate_id','duration_ms','topology_seed','dynamics_seed','Z_M','source_hashes'):
        if summaries['legacy'][key]!=summaries['corrected'][key]:
            raise RuntimeError(f'paired run identity mismatch: {key}')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(11,5.8),sharex=True)
    fig.subplots_adjust(left=.09,right=.97,bottom=.18,top=.81,hspace=.18)
    fig.suptitle('p030 short regression: original versus autapse-corrected graph',fontsize=16,weight='bold',y=.97)
    fig.text(.09,.88,'Same topology seed, core placement, learned parameters and drive seed; Z/M off.',fontsize=11)
    for arm,color,label in [('legacy','#999999','Original'),('corrected','#245C3F','Autapse corrected')]:
        summary=summaries[arm]
        arr=summary['arrays']
        if sha(arr['path'])!=arr['sha256']:
            raise RuntimeError('array source changed')
        with np.load(arr['path']) as data:
            dt=float(data['dt_ms']);n=int(round(20/dt))
            for ax,pop in zip(axes,('E','I')):
                rate=data[f'rate_{pop}_hz'];binned=rate[:len(rate)//n*n].reshape(-1,n).mean(1)
                times=(np.arange(len(binned))+.5)*n*dt/1000
                ax.plot(times,binned,color=color,lw=1.0,label=label,alpha=.85)
                ax.set_ylabel(f'{pop} rate (Hz)')
    axes[0].legend(frameon=False,loc='upper right')
    axes[-1].set(xlabel='Time (s)',xlim=(0,8))
    fig.text(.09,.055,'Rates averaged over 20 ms. One 8-second realization; no event-class or Z/M-transition validation.',fontsize=10)
    stem=figures/'autapse_correction_paired_rates'
    for ext in ('png','pdf','svg'):
        fig.savefig(stem.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    a,b=summaries['legacy'],summaries['corrected']
    comparison={'status':'PAIRED_SHORT_REGRESSION_COMPLETE','arms':summaries,
                'mean_E_rate_relative_change':b['mean_E_rate_hz']/a['mean_E_rate_hz']-1,
                'peak_20ms_E_rate_relative_change':b['peak_20ms_E_rate_hz']/a['peak_20ms_E_rate_hz']-1,
                'source_hashes':{str(out/f'{arm}_summary.json'):sha(out/f'{arm}_summary.json') for arm in summaries},
                'producer_sha256':sha(__file__),
                'output_hashes':{str(stem.with_suffix('.'+ext)):sha(stem.with_suffix('.'+ext)) for ext in ('png','pdf','svg')},
                'claim_boundary':'Finite trajectories under the corrected graph. Similar mean rate is not evidence of equivalent event distributions, patient fit, or transition behavior.'}
    (out/'paired_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    (figures/'README.md').write_text('### autapse_correction_paired_rates.png / .pdf / .svg\np030、topology/dynamics 2511、相同原双核与OU种子的8秒动态对照，Z/M关闭；上下分别为E、I群体20 ms平均放电率。灰色为旧图，绿色为排除自连接后的图，完整数组与图哈希记录在paired_comparison.json。\n**关注点**：平均率接近不能说明瞬时轨迹、事件分布或转变行为等价，本图用于修复回归，不作患者拟合或发作结论。\n')
    print(json.dumps({k:comparison[k] for k in ('status','mean_E_rate_relative_change','peak_20ms_E_rate_relative_change')},indent=2))


if __name__=='__main__':
    main()
