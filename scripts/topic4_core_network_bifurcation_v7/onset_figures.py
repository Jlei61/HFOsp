from common import *
from figures import save,J,COL,FIG
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from scipy.optimize import brentq

def main():
    s=System();h=read(OUT/'homoclinic_audit.json');fold=read(OUT/'folds/low_global_fold_014_N4096.json');jhc=h['fits'][-1]['J_infinite_period']
    rows=read(OUT/'arcs/low_fast/progress.json');split=next(i for i in range(len(rows)-1) if rows[i]['tangent_g']*rows[i+1]['tangent_g']<0)
    tail_ok=(OUT/'long_tail_stability_validation.json').exists() and read(OUT/'long_tail_stability_validation.json').get('status')=='STABLE_BOUND'
    stable=[fold]+rows[split+1:]+[dict(g=x['g'],mean_hz=x['mean_hz'],T_ms=x['T_ms']) for x in h['orbits'] if tail_ok or x['T_ms']<1001]
    unstable=rows[:split+1]+[fold]
    native=read(OUT/'native_coordinates.json') if (OUT/'native_coordinates.json').exists() else []
    fig,axes=plt.subplots(2,2,figsize=(12,10));fig.subplots_adjust(left=.09,right=.97,bottom=.08,top=.94,wspace=.27,hspace=.3)
    for group in (0,1):
        ax=axes[0,group]
        for seq,ls in [(unstable,'--'),(stable,'-')]:ax.plot([(r['g']-1.1218)*1e6 for r in seq],[r['mean_hz'][group] for r in seq],color=COL[group],ls=ls,lw=2,label='Unstable periodic' if ls=='--' else 'Stable periodic')
        if not tail_ok:
            tail=[x for x in h['orbits'] if x['T_ms']>999];ax.plot([(r['g']-1.1218)*1e6 for r in tail],[r['mean_hz'][group] for r in tail],color=COL[group],ls=':',lw=2,label='Long-period continuation')
        for init,color,ls,name in [([.415,.304,0,0,0,0],'#222222','-','Stable low rate'),([.498,.304,0,0,0,0],'#a05038','--','Saddle')]:
            xx=np.linspace(1.12180,1.12185,100);rr=[]
            for g in xx:
                r,err,ok=s.solve(g,np.array(init)/1000);assert ok;rr.append(r[group]*1000)
            ax.plot((xx-1.1218)*1e6,rr,color=color,ls=ls,lw=1.4,label=name)
        ax.plot((fold['g']-1.1218)*1e6,fold['mean_hz'][group],'o',mfc='white',mec='#222222',ms=8)
        ax.annotate('Cycle fold',((fold['g']-1.1218)*1e6,fold['mean_hz'][group]),xytext=(6,20),textcoords='offset points',arrowprops=dict(arrowstyle='-',lw=.7))
        ax.axvline((jhc-1.1218)*1e6,color='#a05038',ls=':',lw=1.2)
        ax.plot((jhc-1.1218)*1e6,h['saddle_hz'][group],marker='*',ms=10,mfc='white',mec='#a05038')
        ax.text((jhc-1.1218)*1e6+1,1.5,r'HC: $T\rightarrow\infty$',fontsize=11)
        ax.set(xlim=(5,50),ylim=(-.3,17),title=f'Core {"AB"[group]}',xlabel=r'$(J_{\mathrm{EE,core}}-1.1218)\times10^6$',ylabel='Mean E rate (Hz / cell)')
        if group==0:ax.legend(frameon=False,fontsize=9,loc='upper left')
    data=h['orbits'];tt=np.array([r['T_ms'] for r in data])/1000;gg=np.array([r['g'] for r in data]);ax=axes[1,0]
    ax.semilogy(tt,jhc-gg,'o',color=COL[0],label='Periodic BVP solutions');x=np.linspace(tt.min(),tt.max(),100);f=h['fits'][-1]
    ax.semilogy(x,f['amplitude']*np.exp(-f['rate_per_s']*x),color=COL[0],label=f"Tail fit: {f['rate_per_s']:.3f} / s")
    ax.set(xlabel='Full period (s)',ylabel=r'$J_{HC}-J_{\mathrm{EE,core}}$',title='Convergence to the homoclinic limit');ax.legend(frameon=False)
    ax=axes[1,1];d=np.array([r['distance_to_saddle_hz'] for r in data]);ax.semilogy(tt,d,'o-',color=COL[2]);ax.set(xlabel='Full period (s)',ylabel='Closest distance to saddle (Hz)',title='Orbit approaches the same saddle')
    save(fig,'onset_cycle_fold_homoclinic','新增的周期折点和同宿型终止构成一个窄的稳定低率/周期burst共存区。实/虚线周期段由折点两侧返回乘子核查支持；HC来自有限长周期轨道对同一鞍点的逼近及参数指数收敛，而非已经求得精确无限时长同宿边值解。')
    # Long-period waveform retains true time; quiet plateau is not cropped out.
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True);fig.subplots_adjust(left=.12,right=.96,bottom=.09,top=.93,hspace=.18)
    for i,path in enumerate([OUT/'periodic/long_period_tail/T1000_N8192.npz',OUT/'periodic/long_period_tail/T1500_N8192.npz',OUT/'periodic/long_period_tail/T2200_N8192.npz']):
        z=np.load(path);r=z['r']*1000;T=float(z['T']);peak=np.argmax(r[:,0]);r=np.roll(r,-peak+round(.15/(T/1000)*len(r)),axis=0);t=np.arange(len(r))*T/len(r)/1000
        for k in (0,1,2):axes[i].plot(t,r[:,k],color=COL[k],label=['A E','B E','Sur E'][k],lw=1.7)
        axes[i].set(ylabel='Rate (Hz / cell)',title=f'T = {T:.0f} ms',ylim=(-2,260));axes[i].legend(frameon=False,loc='upper right',ncol=3)
    axes[-1].set_xlabel('Time within orbit (s)');save(fig,'onset_long_period_waveforms','相同确定性周期族的1000、1500、2200ms解。峰值仍有限，周期增长主要来自鞍点附近的低活动停留时间增加；各轨道以A峰对齐，时间轴仍为真实秒。')
if __name__=='__main__':main()
