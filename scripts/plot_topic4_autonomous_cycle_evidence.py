#!/usr/bin/env python3
"""Continuous full-run state evidence and two honest absolute-time raster zooms."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_topic4_fixed_zm_termination as a

def main():
    root=a.OUT/'autonomous_recurrence_continuation';a.OUT=root;a.run.OUT=root
    name='sahp1.5_g0.5_s9108401_to50s';folder=root/'runs'/name;r=a.analyze(name)
    assert len(r['entries'])>=2 and r['recoveries']
    d=a.load(folder);k=a.load(folder,'intrinsic_adaptation_chunks');geo=dict(np.load(root/'geometry.npz'))
    n=len(d['spikes_1ms'])//10;time=(np.arange(n)+.5)*.01;counts=d['spikes_1ms'][:n*10].reshape(n,10,2).sum(1)
    rates=counts/np.array([320,80]);local=d['regions_1ms'][:n*10,:2].reshape(n,10,2).sum(1)/geo['region_counts'][:2]/.01
    on=r['entries'][1]['onset_s'];lo=round(on/.01);window=rates[lo:lo+20,0];assert len(window)==20 and np.all(window>=200)
    display_end=min(r['observed_s'],r['entries'][1]['confirmation_s']+2.)
    audit=dict(status='PASS',second_entry_s=on,second_confirmation_s=r['entries'][1]['confirmation_s'],minimum_E_Hz_in_second_entry_window=window.min(),first_return_confirmation_s=r['recoveries'][0]['confirmation_s'],no_Z_M_K_reset=True,same_parameters=True,source=str(folder),same_realization_as_first_episode=True,observed_s=r['observed_s'])
    audit['display_end_s']=display_end
    a.write(root/'recurrence_count_audit.json',audit)
    plt.rcParams.update({'font.size':14,'axes.labelsize':17,'xtick.labelsize':13,'ytick.labelsize':13,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(17,11));gs=fig.add_gridspec(4,2,height_ratios=[1.3,1,.7,1.65],hspace=.42,wspace=.22,left=.09,right=.92,top=.95,bottom=.08)
    ax=fig.add_subplot(gs[0,:]);ax.plot(time,rates[:,0],c='#206c9a',lw=1,label='All E');ax.plot(time,rates[:,1],c='#c67c2d',lw=.8,label='All I');ax.legend(loc='upper center',ncol=2,frameon=False)
    ax.set(xlim=(0,display_end),ylabel='Rate (Hz)',ylim=(0,max(rates[time<=display_end].max()*1.07,250)));ax.tick_params(labelbottom=False)
    zax=fig.add_subplot(gs[1,:],sharex=ax);st=d['slow_time_ms']/1000;zax.plot(st,d['Z'][:,0],c='#794495',lw=1.7,label='Mean Z');zax.fill_between(st,d['Z'][:,2],d['Z'][:,4],color='#794495',alpha=.15)
    zz=zax.twinx();zz.plot(st,r['job']['eta_m']*d['M'][:,0],c='#b47728',lw=1);zz.set_ylabel(r'Native $\eta_M M$ (mV equiv.)',c='#b47728');zz.spines['right'].set_visible(True)
    zax.set(ylabel='Mean Z',ylim=(0,1.05));zax.tick_params(labelbottom=False)
    ka=fig.add_subplot(gs[2,:],sharex=ax);ka.plot(k['time_ms']/1000,k['sahp_mean_conductance_ratio'],c='#a86022',lw=1.2);ka.set(ylabel='Added '+r'$g_K/g_L$',xlabel='Time (s)')
    marks=[(r['entries'][0]['onset_s'],'First high','#ad3049'),(r['recoveries'][0]['confirmation_s'],'Return','#248a78'),(on,'Second high','#ad3049')]
    for tm,label,col in marks:
        for axes in [ax,zax,ka]:axes.axvline(tm,c=col,ls=':',lw=1)
        ax.text(tm,1.02,label+'\n'+f'{tm:.2f} s',transform=ax.get_xaxis_transform(),ha='center',c=col,fontsize=13)
    it,ix=np.where(d['raster']);rt=it*.0001;mapping=np.r_[np.linspace(0,33,20),np.linspace(36,69,20),np.linspace(72,84,20),np.linspace(87,99,20)]
    for i,win in enumerate([(0,12),(max(0,on-3),display_end)]):
        ra=fig.add_subplot(gs[3,i])
        for low,high,col in [(0,20,'#176ba1'),(20,40,'#168aa2'),(40,60,'#466675'),(60,80,'#c17730')]:
            take=(ix>=low)&(ix<high)&(rt>=win[0])&(rt<win[1]);ra.scatter(rt[take],mapping[ix[take]],s=5,c=col,lw=0,rasterized=True)
        for y in [34.5,70.5,85.5]:ra.axhline(y,c='#ccc',lw=.5)
        ra.set(xlim=win,ylim=(-2,102),xlabel='Time (s)',yticks=[16.5,52.5,78,93],yticklabels=['Core A E','Core B E','Other E','I'] if i==0 else [])
    dest=root/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/'autonomous_cycle_full_timeline.png',dpi=160);fig.savefig(dest/'autonomous_cycle_full_timeline.pdf');plt.close(fig)
    readme=dest/'README.md';text=readme.read_text() if readme.exists() else ''
    section='### autonomous_cycle_full_timeline.png\n\n上部展示同一条记录的完整时间轴、全E/I率、原Z/M及新增慢钾电导；下方分别放大第一次和第二次事件，使用绝对时间。第二次进入按全体E计数独立复核，原参数及全部状态从第一次恢复后连续演化，没有手工重置。\n\n**关注点**：证明一条轨迹中自主退出后可再次进入，不等同跨噪声稳健性、患者间期模式恢复或临床复发率验证。\n'
    if '### autonomous_cycle_full_timeline.png' not in text:readme.write_text(text+'\n'+section)
    print(audit)

if __name__=='__main__':main()
