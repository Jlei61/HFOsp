#!/usr/bin/env python3
"""Shared-resolution observables for the native/rate paired-history experiment."""
from topic4_spatial_boundary_common import OUT, OLD, read, write, observables
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze():
    source=np.load(OLD/'external_input.npz');ce10=source['cell_e'];weights=source['count_e']
    rows=[];arrays={};profiles=[8000,8400,8800,9200,9400]
    for branch in ('native','rate'):
        allowed={j['name'] for j in read(OUT/f'{branch}_jobs.json')}
        for path in sorted((OUT/branch).glob('*.json')):
            if path.stem not in allowed:continue
            meta=read(path);a=np.load(path.with_suffix('.npz'));name=path.stem
            if branch=='native':
                ce20=a['cell_e'];aggregation=np.zeros((100,400));aggregation[ce10,ce20]=1
                counts=a['field_e_count_1ms']@aggregation.T
            else:
                counts=a['fields_hz'][:,0]*weights[None,:]/1000
            m=observables(counts,weights)
            rows.append({'model':branch,'name':name,'z_profile_ms':meta['z_profile_ms'],
                'history_ms':meta['history_ms'],'metrics':m})
            rate1=counts/weights[None,:]*1000
            arrays[(branch,name)]=(a,rate1,np.average(rate1,axis=1,weights=weights))
    write(OUT/'factorial_metrics.json',{'rows':rows,'completed_native':sum(r['model']=='native' for r in rows),
        'completed_rate':sum(r['model']=='rate' for r in rows),
        'resolution':'Both compared on the same 10x10 E-cell partition and neuron weights. Native 20x20 count maps re-aggregated exactly.',
        'scope':'Second second of each 2-s continuation; repeated cells/time bins are not independent samples; finite-time state separation is not an asymptotic bifurcation.'})
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    colors={'native':'#222222','rate':'#ae3a6b'}
    for row_index,history in enumerate((8000,9400)):
        for branch in ('native','rate'):
            selected={r['z_profile_ms']:r['metrics'] for r in rows if r['model']==branch and r['history_ms']==history}
            x=[t/1000 for t in profiles if t in selected]; ms=[selected[t] for t in profiles if t in selected]
            values=[[m['E_mean_hz'] for m in ms],
                    [m['quiet']['1Hz_5ms']['fraction'] for m in ms],
                    [m['spatial']['50Hz']['persistent_fraction_duty80'] for m in ms]]
            for col,y in enumerate(values):axs[row_index,col].plot(x,y,'o-',c=colors[branch],label=branch.title())
        for col,ax in enumerate(axs[row_index]):
            ax.set(xlabel='Time supplying the frozen Z profile (s)',
                   title=f'{chr(65+row_index*3+col)}  History from {history/1000:g} s')
            ax.legend(fontsize=10)
        axs[row_index,0].set_ylabel('Mean E rate (Hz)')
        axs[row_index,1].set(ylabel='Quiet-bin fraction',ylim=(-.025,1.025))
        axs[row_index,2].set(ylabel='Persistent spatial fraction\n(>50 Hz in >=80% of 10-ms bins)',ylim=(-.025,1.025))
    fig.suptitle('Fixed Z and paired activity histories | identical future-input protocol',fontsize=15)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    name='paired_history_spatial_boundary'
    fig.savefig(folder/f'{name}.png',bbox_inches='tight');fig.savefig(folder/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    readme=folder/'README.md';text=readme.read_text() if readme.exists() else ''
    marker='### paired_history_spatial_boundary'
    sections=['### '+part for part in text.split('### ') if part.strip() and not part.startswith('paired_history_spatial_boundary\n')]
    text=''.join(sections)
    readme.write_text(text+'\n'+marker+'\n\n'
        '固定五个实际逐神经元 Z 场，分别从 8.0 秒及 9.4 秒的完整快速状态开始，在共同未来输入下比较原 SNN 和 rate model。图中使用最后一秒、同一 10×10 空间分区的群体活动、安静间隔和持续空间占据；原 SNN 的 20×20 计数先精确合并。\n\n'
        '**关注点**：横轴是提供 Z 场的原轨迹时间，平均 Z 并不严格单调；连线是离散条件的比较，不是经过验证的连续分岔支。PNG/PDF 待用户目视审阅。\n')
    return rows


if __name__=='__main__':
    rows=analyze()
    for r in rows:
        m=r['metrics'];print(r['model'],r['name'],round(m['E_mean_hz'],2),
            m['quiet']['1Hz_5ms']['gaps_at_least_20ms'],round(m['spatial']['50Hz']['persistent_fraction_duty80'],3))
