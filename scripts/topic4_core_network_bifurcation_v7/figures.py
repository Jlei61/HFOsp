"""Bifurcation atlas with explicit native-versus-reduced numbered correspondences."""
from common import *
import numpy as np,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import resample
from PIL import Image
J=r'$J_{\mathrm{EE,core}}$';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
COL=['#286aa4','#9553a3','#28865c','#df8243','#c05f88','#80804a']
plt.rcParams.update({'font.size':12,'axes.labelsize':13,'axes.titlesize':15,'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':10})
MAN=[];CAP=[]
def save(fig,name,caption):
    assert all(not ax.child_axes for ax in fig.axes)
    for ext in ('png','pdf'):fig.savefig(FIG/f'{name}.{ext}',dpi=190)
    with Image.open(FIG/f'{name}.png') as z:z.load();size=list(z.size)
    dpath=OUT/'figure_descriptions.json';descriptions=read(dpath) if dpath.exists() else {};descriptions[name]=caption;write('figure_descriptions.json',descriptions)
    MAN.append(dict(name=name,pixels=size,axes=len(fig.axes)));CAP.append(f'### {name}.png / .pdf\n{caption}\n**关注点**：核对同编号、参数、模型层与两核读出。\n');plt.close(fig)
def orb(p):
    z=np.load(p);r=z['r'];rr=resample(r,max(8192,len(r)),axis=0)
    return dict(g=float(z['g']),T=float(z['T']),mean=r.mean(0)*1000,lo=rr.min(0)*1000,hi=rr.max(0)*1000,path=str(p),r=r)
def critical():
    out={}
    for row in csv.DictReader((V6/'critical_points.csv').open()):out[row['label'] if 'label' in row else row['name']]=row
    return out

def reduced_cases():
    rows=[];s=System();eq=read(V2/'equilibrium_spectrum.json')
    for number,g in [(1,.5),(2,.7),(3,.85),(4,1.),(5,1.1),(19,1.12181),('20a',1.12183)]:
        seed=min([a for a in eq if a['direction']==-1],key=lambda a:abs(a['g']-g));r,err,ok=s.solve(g,np.array(seed['r_hz'])/1000)
        assert err<1e-9
        rows.append(dict(number=str(number),g=g,r=np.tile(r,(1024,1)),T=500.,mean=r*1000,kind='Stable equilibrium',path=str(V2/'equilibrium_spectrum.json')))
    for p in (OUT/'periodic').glob('condition_*/*.npz'):
        row=orb(p);row.update(number=p.parent.name.removeprefix('condition_'),kind='Stable periodic orbit');rows.append(row)
    rows.sort(key=lambda r:(int(''.join(c for c in r['number'] if c.isdigit())),r['number']))
    write('reduced_condition_coordinates.json',[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items() if k!='r'} for r in rows]);return rows

def main_axis(ax,seq,group,native,window=None):
    fold=read(V2/'fold.json');eq=read(V2/'equilibrium_spectrum.json')
    for direction,ls in [(-1,'-'),(1,'--')]:
        a=[r for r in eq if r['direction']==direction];a=sorted(a,key=lambda x:x['g']) if direction==-1 else a
        ax.plot([r['g'] for r in a],[r['r_hz'][group] for r in a],ls=ls,color=COL[0],lw=2)
    for idx,rows in enumerate(seq):
        ax.plot([r['g'] for r in rows],[r['mean'][group] for r in rows],ls='-' if rows[0].get('stable',idx<4) else '--',color='#d18a26',lw=2)
        if rows[0].get('stable',idx<4):
            for key,color in [('hi','#28865c'),('lo','#36a4b6')]:ax.plot([r['g'] for r in rows],[r[key][group] for r in rows],color=color,lw=1.5)
    cp=[('Fold',fold['g'],fold['r_hz'][group])]
    fpath=OUT/'folds/low_global_fold_014_N4096.json'
    if fpath.exists():
        fc=read(fpath);ax.plot(fc['g'],fc['mean_hz'][group],marker='o',mfc='white',mec='#222222',ms=6)
        ax.annotate('Cycle fold',(fc['g'],fc['mean_hz'][group]),xytext=(-90,95),textcoords='offset points',arrowprops=dict(arrowstyle='-',lw=.7),fontsize=11)

    for label,folder in [('LP1','folds/burst_end_fold_N4096'),('PD1','flips/mixed_lower_flip_N2048'),('PD2','flips/mixed_flip_N2048'),('PD3','flips/tonic_lower_flip_N2048')]:
        z=read(V5/(folder+'.json'));cp.append((label,z['g'],z['mean_hz'][group]))
    offsetA={'Fold':(-75,5),'LP1':(-48,45),'PD1':(-70,40),'PD2':(40,-20),'PD3':(30,-35)}
    offsetB={'Fold':(-75,5),'LP1':(-60,45),'PD1':(-65,-40),'PD2':(25,20),'PD3':(-65,-10)}
    for label,g,y in cp:
        ax.scatter(g,y,s=40,facecolors='white',edgecolors='#202020',zorder=7)
        ax.annotate(label,(g,y),xytext=(offsetA if group==0 else offsetB)[label],textcoords='offset points',arrowprops=dict(arrowstyle='-',lw=.7),fontsize=11)
    for r in native:
        g=r['g'];y=r.get('main_means',r['means'])[group];n=r['number']
        if y is None:continue
        ax.scatter(g,y,s=38,c='#171717',marker='D',zorder=9)
    # All markers retain measured coordinates. Close parameter pairs share a
    # label in the full overview; enlarged axes label each member separately.
    clusters=[[1],[2],[3],[4],[5],[19,20],[6],[7,8],[9,10],[11,12],[13,14,15,16,17],[18]] if window is None else [[r['number']] for r in native]
    for ids in clusters:
        rr=[r for r in native if r['number'] in ids and r.get('main_means',r['means'])[group] is not None]
        if not rr:continue
        g=np.mean([r['g'] for r in rr]);y=np.mean([r.get('main_means',r['means'])[group] for r in rr]);n=ids[0]
        if window is not None and not window[0]<=g<=window[1]:continue
        label=','.join(str(r['number']) for r in rr) if len(rr)<=2 else f"{rr[0]['number']}–{rr[-1]['number']}"
        if window is None:
            dx,dy={5:(-15,58),19:(-10,30),6:(8,50),7:(8,85),9:(0,20),11:(-12,65),13:(20,25),18:(25,40)}.get(n,(0,20+(n%3)*10))
        else:dx,dy=(-8 if n%2 else 8),18+(n%3)*20
        ax.annotate(label,(g,y),xytext=(dx,dy),textcoords='offset points',ha='center',fontsize=10,fontweight='bold',arrowprops=dict(arrowstyle='-',lw=.5),bbox=dict(facecolor='white',edgecolor='none',pad=.5),zorder=10)
    ax.set(xlabel=J,ylabel=f'Core {"AB"[group]} E rate (Hz / cell)',xlim=window or (.45,1.62),ylim=(-5,450 if window is None else 405))
    ax.legend(handles=[Line2D([],[],color=COL[0],label='Equilibrium'),Line2D([],[],color='#d18a26',label='Periodic mean'),Line2D([],[],color='#28865c',label='Periodic maximum'),Line2D([],[],color='#36a4b6',label='Periodic minimum'),Line2D([],[],color='#222222',ls='--',label='Unstable'),Line2D([],[],color='#171717',marker='D',ls='',label='Numbered native SNN')],loc='upper left',frameon=False,fontsize=10)

def main():
    seq=read(V6/'displayed_curve_sequences.json')
    if (OUT/'folds/low_global_fold_014_N4096.npz').exists():
        paths=read(OUT/'arcs/low_fast/progress.json');split=next(i for i in range(len(paths)-1) if paths[i]['tangent_g']*paths[i+1]['tangent_g']<0)
        foldpath=OUT/'folds/low_global_fold_014_N4096.npz'
        older=read(OUT/'arcs/low_global/progress.json')
        tail_ok=(OUT/'long_tail_stability_validation.json').exists() and read(OUT/'long_tail_stability_validation.json').get('status')=='STABLE_BOUND'
        tailpaths=[p for p in (OUT/'periodic/long_period_tail').glob('T*_N8192.npz') if tail_ok or float(np.load(p)['T'])<1001]
        for stable,pp in [(False,[x['source'] for x in older+paths[:split+1]]+[foldpath]),(True,[foldpath]+[x['source'] for x in paths[split+1:]]+sorted(tailpaths,key=lambda p:float(np.load(p)['T'])))]:
            data=[]
            for path in pp:
                row=orb(path);row.pop('r');row.update(stable=stable)
                for key in ('mean','lo','hi'):row[key]=row[key].tolist()
                data.append(row)
            seq.append(data)
    write('displayed_curve_sequences.json',seq)
    native=read(OUT/'native_coordinates.json') if (OUT/'native_coordinates.json').exists() else []
    orig=read(V6/'native_AB_coordinates.json')
    # Preserve historical 1-4 locations based on their original 2-20s means.
    for r in native:
        if r['number']<=4:
            old=next(x for x in orig if x['number']==r['number']);r['main_means']=old['means']
    write('plotted_native_coordinates.json',native)
    for group in (() if '--waveforms-only' in sys.argv else (0,1)):
        for zoom in ('full','right','middle','onset'):
            fig,ax=plt.subplots(figsize=(8.4,8.4));fig.subplots_adjust(left=.13,right=.97,bottom=.11,top=.93)
            window={'full':None,'right':(1.32,1.46),'middle':(1.08,1.28),'onset':(1.12178,1.12186)}[zoom]
            main_axis(ax,seq,group,native,window);ax.set_title(f'Core {"AB"[group]} — '+('bifurcation and native conditions' if zoom=='full' else zoom+' correspondence'))
            if zoom!='full':ax.get_legend().remove()
            if zoom in ('middle','onset'):ax.set_ylim(-1,85 if zoom=='middle' else 40)
            # 19/20 resolve only in their own very narrow parameter panel.
            if zoom=='middle':
                for text in list(ax.texts):
                    if text.get_text() in ('19','20'):text.remove()
                pair=[r for r in native if r['number'] in (19,20)]
                if pair:ax.annotate(','.join(str(r['number']) for r in pair),(np.mean([r['g'] for r in pair]),np.mean([r['means'][group] for r in pair])),xytext=(-10,40),textcoords='offset points',fontsize=10,fontweight='bold',arrowprops=dict(arrowstyle='-',lw=.5))
            save(fig,f'00_core_{"AB"[group]}_'+('bifurcation' if zoom=='full' else zoom+'_numbers'), '确定性平衡点与周期支来自v2–v7边值解；黑菱形为真实SNN统计均值与图库编号，不能与橙色周期均值当成同一条曲线。原编号1–4保留2–20秒原坐标，新增5–20取2–12秒；总图合并过近的编号，放大图逐一对应。虚线为已确认不稳定周期支，没有跨未知连接补线。')
    if '--main-only' in sys.argv:
        write('bifurcation_main_figure_manifest.json',MAN)
        return
    cases=reduced_cases()
    with PdfPages(FIG/'reduced_numbered_waveforms.pdf') as book:
        for r in cases:
            rate=r['r']*1000;count=np.array([720,742,30538,197,200,7603]);t=np.arange(len(rate))*r['T']/len(rate)/1000
            fig,ax=plt.subplots(1,3,figsize=(14,4.4));fig.subplots_adjust(left=.09,right=.99,bottom=.17,top=.72,wspace=.30)
            fig.suptitle(f"{r['number']}    Reduced model    {J} = {r['g']:g}",fontsize=18,y=.98)
            fig.text(.5,.845,r['kind']+(f"    T = {r['T']:.3f} ms" if r['kind'].startswith('Stable periodic') else ''),ha='center',fontsize=13)
            traces=[[(rate@count/40000,'All cells','#222222'),(rate[:,:3]@count[:3]/32000,'E',COL[0]),(rate[:,3:]@count[3:]/8000,'I',COL[3])],[(rate[:,0],'E',COL[0]),(rate[:,3],'I',COL[3])],[(rate[:,1],'E',COL[1]),(rate[:,4],'I',COL[4])]]
            for j,aa in enumerate(ax):
                for y,label,color in traces[j]:aa.plot(np.r_[t,t+r['T']/1000],np.r_[y,y],label=label,color=color,lw=1.6)
                aa.set(title=['Whole network (E + I)','Core A','Core B'][j],xlabel='Time within orbit (s)',ylabel='Rate (Hz / cell)');aa.legend(frameon=False);aa.set_ylim(bottom=0)
            book.savefig(fig);save(fig,f'reduced_{r["number"]}_waveform','与同编号原生SNN条件使用相同J。这里的线是确定性模型周期边值解，a/b表示同J的不同周期族；不生成或伪造raster。')
    write('bifurcation_figure_manifest.json',MAN);(FIG/'REDUCED_README.md').write_text('\n'.join(CAP))
    print('REDUCED_FIGURES',len(MAN),flush=True)
if __name__=='__main__':main()
