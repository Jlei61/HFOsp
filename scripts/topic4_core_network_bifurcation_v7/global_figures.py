from common import *
from figures import save,FIG,J,COL
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample

def main():
    fig,axes=plt.subplots(2,2,figsize=(12,10));fig.subplots_adjust(left=.09,right=.97,bottom=.09,top=.93,wspace=.24,hspace=.3)
    sets=[('Low-surround family',COL[0],[V4/'arcs/recruitment_turn',V5/'arcs/surround_unstable',OUT/'arcs/low_global',OUT/'arcs/low_fast']),('Recruited-surround family',COL[2],[V5/'arcs/surround_recruited_back',OUT/'arcs/recruited_global',OUT/'arcs/recruited_fast'])]
    for label,color,paths in sets:
        rows=sum([read(p/'progress.json') for p in paths if (p/'progress.json').exists()],[])
        if label.startswith('Low-'):
            rows+=read(OUT/'homoclinic_audit.json')['orbits']
        g=[r['g'] for r in rows];T=[r['T_ms'] for r in rows]
        for ax,key,index in [(axes[0,0],'mean_hz',0),(axes[0,1],'mean_hz',1),(axes[1,0],'mean_hz',2)]:ax.plot(g,[r[key][index] for r in rows],color=color,lw=1.8,label=label)
        axes[1,1].plot(g,T,color=color,lw=1.8,label=label)
        for ax,key,index in [(axes[0,0],'mean_hz',0),(axes[0,1],'mean_hz',1),(axes[1,0],'mean_hz',2),(axes[1,1],'T_ms',None)]:
            yy=rows[-1][key] if index is None else rows[-1][key][index];ax.plot(g[-1],yy,'s',mfc='white',mec=color,ms=7)
    for i,ax in enumerate(axes.flat):ax.set(xlabel=J,ylabel=['Core A E mean (Hz)','Core B E mean (Hz)','Surround E mean (Hz)','Full period (ms)'][i]);ax.legend(frameon=False)
    fig.suptitle('Extended periodic families — continuation order',fontsize=17)
    save(fig,'global_periodic_families','两条周期族按实际弧长顺序绘制，颜色只表示族身份，此图不把未测谱的位置认作稳定或不稳定。空方块为本次数值延拓终点，不是已证明的分岔点；均值投影相交不构成轨道连接。')
    rows=read(OUT/'arcs/recruited_global/progress.json')+read(OUT/'arcs/recruited_fast/progress.json')
    fig,axes=plt.subplots(2,2,figsize=(12,10));fig.subplots_adjust(left=.1,right=.97,bottom=.09,top=.93,wspace=.27,hspace=.3)
    specs=[('mean_hz',0,'Core A E mean (Hz)'),('mean_hz',1,'Core B E mean (Hz)'),('mean_hz',2,'Surround E mean (Hz)'),('T_ms',None,'Full period (ms)')]
    for ax,(key,index,ylabel) in zip(axes.flat,specs):
        y=[r[key] if index is None else r[key][index] for r in rows]
        ax.plot([(r['g']-1.17625)*1e6 for r in rows],y,color=COL[2],lw=1.5)
        ax.plot((rows[-1]['g']-1.17625)*1e6,y[-1],'s',mfc='white',mec=COL[2],ms=7)
        for k,p in enumerate(sorted((OUT/'folds').glob('recruited*_N4096.json')),1):
            r=read(p);value=r[key] if index is None else r[key][index];ax.plot((r['g']-1.17625)*1e6,value,'o',mfc='white',mec='#222222',ms=7)
            ax.annotate(f'Cycle fold {k}',((r['g']-1.17625)*1e6,value),xytext=(-80 if k==1 else 10,15),textcoords='offset points',fontsize=11,arrowprops=dict(arrowstyle='-',lw=.7))
        ax.set(xlabel=r'$(J_{\mathrm{EE,core}}-1.17625)\times10^6$',ylabel=ylabel)
    fig.suptitle('Recruited periodic family — resolved continuation',fontsize=17)
    save(fig,'recruited_family_resolved','招募周边周期族的局部参数轴放大，保留全部弧长折返和多burst周期延长。圆圈为两个人工精化并检查零模的新增周期折点，方块为有限延拓终点；此图不为每个细小折返统一分配稳定性，也不把所有转弯都计作已精化临界点。')
    paths=sorted((OUT/'folds').glob('*_N4096.npz'))
    if not paths:return
    fig,axes=plt.subplots(len(paths),3,figsize=(14,3.8*len(paths)),squeeze=False);fig.subplots_adjust(left=.07,right=.98,bottom=.08,top=.92,wspace=.3,hspace=.4)
    for k,p in enumerate(paths):
        z=np.load(p);r=z['r']*1000;t=np.arange(len(r))*float(z['T'])/len(r)/1000;row=read(p.with_suffix('.json'))
        for j in (0,1,2):axes[k,0].plot(t,r[:,j],label=['A E','B E','Sur E'][j],color=COL[j])
        axes[k,0].set(title=f"Cycle fold: {J}={float(z['g']):.10f}",xlabel='Time (s)',ylabel='Rate (Hz / cell)');axes[k,0].legend(frameon=False)
        for j,key in enumerate(['right_null','left_null'],1):
            v=z[key][:-1].reshape(len(r),6);frac=(v*v).sum(0);frac/=frac.sum()
            axes[k,j].bar(np.arange(6),100*frac,color=COL);axes[k,j].set(xticks=np.arange(6),xticklabels=['A E','B E','S E','A I','B I','S I'],ylabel='Rate-mode squared norm (%)',title=('Right' if j==1 else 'Adjoint')+' critical mode')
    fig.suptitle('Additional cycle folds on continued families',fontsize=17)
    save(fig,'global_fold_waveforms_modes','新增周期折点的联合E波形及六群体左右临界率模。比例不按细胞数加权，不是因果贡献；左右零模来自完整周期边值Jacobian，相位中性方向已单独约束。')
if __name__=='__main__':main()
