"""Joint-network critical orbits, per-core observables and critical modes."""
from common import *
from figures import save,FIG,J,COL
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

def main():
    cases=[('Cycle fold',OUT/'folds/low_global_fold_014_N4096.npz','19 / 20')]
    # Names are recovered from the accepted critical table, never guessed.
    import csv
    old=list(csv.DictReader((V6/'critical_points.csv').open()))
    inventory=[(p,read(p)) for root in (V5,V6) for folder in ('folds','flips') for p in (root/folder).glob('*.json') if p.with_suffix('.npz').exists()]
    for label in ['LP0a','LP0b','LP0c','PD0','LP1','PD1','PD2','PD3']:
        row=next(r for r in old if r.get('label',r.get('name'))==label)
        candidates=[(p,r) for p,r in inventory if abs(r.get('g',0)-float(row['JEE_core']))<2e-14]
        assert candidates,(label,row)
        source=max(candidates,key=lambda x:x[1].get('N',0))[0].with_suffix('.npz')
        cases.append((label,source,{'LP0a':'7 / 8','LP0b':'7 / 8','LP0c':'7 / 8','PD0':'7 / 8','LP1':'12 / 13','PD1':'11 / 12','PD2':'16 / 17','PD3':'14 / 15'}[label]))
    cases.append(('PD0 (2T → 4T)',OUT/'flips/surround_2T_flip_N8192.npz','7 / 8'))
    for path in sorted((OUT/'folds').glob('recruited*_N4096.npz')):cases.append(('Additional cycle fold',path,'7 / 8'))
    manifest=[]
    with PdfPages(FIG/'joint_critical_waveforms_modes.pdf') as book:
        for index,(label,path,numbers) in enumerate(cases,1):
            z=np.load(path);r=z['r']*1000;T=float(z['T']);g=float(z['g']);t=np.arange(len(r))*T/len(r)/1000
            modes=[]
            for key in ('mode','left_mode') if 'mode' in z else ('right_null','left_null'):
                v=z[key];v=v[:-1].reshape(len(r),6) if v.ndim==1 else v
                f=(abs(v)**2).sum(0);modes.append(f/f.sum())
            fig,axes=plt.subplots(2,3,figsize=(15,8.5));fig.subplots_adjust(left=.065,right=.98,bottom=.09,top=.83,hspace=.38,wspace=.31)
            fig.suptitle(f'{label}    {J} = {g:.13f}',fontsize=18,y=.97)
            fig.text(.5,.905,f'Reduced critical orbit    T = {T:.3f} ms    Nearby native conditions: {numbers}',ha='center',fontsize=13)
            counts=np.array([720,742,30538,197,200,7603]);traces=[[(r@counts/40000,'All cells','#222222'),(r[:,:3]@counts[:3]/32000,'E',COL[0]),(r[:,3:]@counts[3:]/8000,'I',COL[3])],[(r[:,0],'E',COL[0]),(r[:,3],'I',COL[3])],[(r[:,1],'E',COL[1]),(r[:,4],'I',COL[4])]]
            for k,ax in enumerate(axes[0]):
                for y,name,color in traces[k]:ax.plot(t,y,label=name,color=color,lw=1.6)
                ax.set(title=['Whole network (E + I)','Core A','Core B'][k],xlabel='Time within orbit (s)',ylabel='Rate (Hz / cell)',ylim=(0,None));ax.legend(frameon=False,ncol=len(traces[k]))
            axes[1,0].plot(r[:,0],r[:,1],color='#333333',lw=1.1);axes[1,0].plot(r[0,0],r[0,1],'o',ms=4,color='#333333')
            axes[1,0].set(xlabel='Core A E rate (Hz)',ylabel='Core B E rate (Hz)',title='Joint A/B orbit projection')
            for k,f in enumerate(modes,1):
                axes[1,k].bar(np.arange(6),100*f,color=COL);axes[1,k].set(xticks=np.arange(6),xticklabels=['A E','B E','S E','A I','B I','S I'],ylabel='Rate-mode squared norm (%)',title=['','Right critical mode','Adjoint critical mode'][k])
            book.savefig(fig);name=f'critical_{index:02d}_joint_orbit_modes'
            save(fig,name,f'{label}的同一条六群体临界周期轨道：上排为全网络与A/B的E/I率，下排为A–B投影和左右临界率模。相邻SNN编号{numbers}使用有限参数间隔，不表示原生SNN在该精确临界J存在同一轨道；图中不生成raster。')
            manifest.append(dict(number=index,label=label,g=g,T_ms=T,source=str(path),nearby_native_ids=numbers,figure=name))
    write('joint_critical_gallery.json',manifest)

if __name__=='__main__':main()
