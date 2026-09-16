#!/usr/bin/env python3
"""All six actual trajectories, with shared rate and Z scales."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import analyze_topic4_interictal_recurrence as audit


def main():
    root=audit.OUT
    protocol=json.loads((root/'protocol.json').read_text())
    plt.rcParams.update({'font.size':14,'axes.labelsize':17,'xtick.labelsize':14,'ytick.labelsize':14,
                         'axes.spines.top':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,2,figsize=(17,11),sharex=True,sharey=True)
    fig.subplots_adjust(left=.075,right=.925,top=.93,bottom=.11,wspace=.24,hspace=.43)
    for index,job in enumerate(protocol['initial_jobs']):
        ax=axes.flat[index]
        row=json.loads((root/'analysis'/f"{job['name']}.json").read_text());p=row['primary']
        d=audit.old.load(root/'runs'/job['name'],keys=['spikes_1ms','slow_time_ms','Z'])
        n=len(d['spikes_1ms'])//10;t=(np.arange(n)+.5)*.01
        rate=d['spikes_1ms'][:n*10,0].reshape(n,10).sum(1)/320
        ax.plot(t,rate,c='#176b9c',lw=.9)
        zax=ax.twinx();zax.plot(d['slow_time_ms']/1000,d['Z'][:,0],c='#8b3d8d',lw=1.1)
        zax.set(ylim=(0,1.05),yticks=[0,.5,1]);zax.tick_params(axis='y',colors='#8b3d8d')
        if index%2==1:zax.set_ylabel('Mean Z',color='#8b3d8d')
        else:zax.tick_params(labelright=False)
        starts=[e['start_s'] for e in p['preentry']['brief_events']]
        ax.scatter(starts,np.full(len(starts),-15),s=25,marker='|',c='#209679')
        for entry in p['entries']:
            on=entry['onset_s'];ax.axvline(on,c='#b82d46',ls='--',lw=1)
            ax.text(on+1,360,f'{on:.2f} s',color='#b82d46',fontsize=13)
        if p['observed_s']<60:
            ax.axvspan(p['observed_s'],60,facecolor='#eeeeee',edgecolor='#aaaaaa',hatch='///',lw=0)
        gamma='1/6' if np.isclose(job['gamma'],1/6) else f"{job['gamma']:g}"
        ax.set_title(f"γ = {gamma}, K = {job['sahp_gain']:g}, τK = {job['sahp_tau_s']:g} s",loc='left',fontsize=17)
        ax.text(.98,.91,f"{p['preentry']['brief_count']} brief events",transform=ax.transAxes,ha='right',fontsize=14)
        ax.set(xlim=(0,60),ylim=(-30,520),yticks=[0,250,500],xticks=[0,20,40,60])
        if index%2==0:ax.set_ylabel('Mean E rate (Hz)')
        if index>=4:ax.set_xlabel('Time (s)')
    fig.legend(handles=[Line2D([],[],c='#176b9c',label='E rate'),Line2D([],[],c='#8b3d8d',label='Z'),
              Line2D([],[],c='#209679',marker='|',ls='none',label='Brief event before first high'),
              Patch(facecolor='#eee',edgecolor='#aaa',hatch='///',label='Unobserved')],
              loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.01),fontsize=13)
    dest=root/'figures';dest.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(dest/f'six_condition_overview.{ext}',dpi=150)
    plt.close(fig)
    audit.old.write(dest/'six_condition_overview_metadata.json',dict(source=str(root),
            common_axes=True,rate_bin_ms=10,Z_sample_ms=20,
            brief_event_definition=audit.RULE,short_records_hatched_not_imputed=True,
            counts_before_first_high_or_full_record_if_none=True,human_review='PENDING'))
    readme=dest/'README.md';text=readme.read_text() if readme.exists() else ''
    text+='\n### six_condition_overview.png / .pdf\n\n六组实际全E放电率与Z共用时间轴和纵轴，绿色短线标出首次高活动前的合格短事件；无高活动时统计整段记录。两条56s记录后四秒用斜线标记未观测，不填充数据。\n\n**关注点**：同时检查短事件是否保留、是否进入与退出；高率平台不等同持续振荡发作，当前六组均未通过完整闭环。\n'
    readme.write_text(text)


if __name__=='__main__':main()
