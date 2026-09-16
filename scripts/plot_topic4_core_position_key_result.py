"""Compact scientific response figure from the complete position pilot."""
import sys,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scripts import run_topic4_core_position_response as run
run.configure()
OUT=run.OUT/'analysis';F=OUT/'key_result/figures'

def main():
    F.mkdir(parents=True,exist_ok=True)
    p=run.rt.read(run.OUT/'plan.json');summary=run.rt.read(OUT/'position_response_summary.json')
    obs=summary['mode_rows'];contacts=list(csv.DictReader((OUT/'contact_participation.csv').open()))
    candidates=[c for c in p['candidates'] if not c['retain_I_state'] and c['y_shift_mm'] is not None]
    candidates.sort(key=lambda c:c['y_shift_mm']);xx=[c['y_shift_mm'] for c in candidates]
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,4,figsize=(15,4.1),layout='constrained')
    with np.load(run.core.result_path('base_off',p['seeds'][0]).with_suffix('.npz')) as a:
        names=a['contact_names'].astype(str);xy=a['contact_xy_mm']
    ax=axes[0]
    for prefix,color in [('SCL','#008a94'),('ICL','#c58727')]:
        ids=sorted([i for i,n in enumerate(names) if n.startswith(prefix)],key=lambda i:int(names[i][3:]))
        ax.plot(*xy[ids].T,'-o',color=color,ms=3,lw=1)
        for i in ids:ax.annotate(names[i],xy[i],xytext=(1,3),textcoords='offset points',fontsize=5)
    for c,color in [(candidates[0],'#777777'),(candidates[-1],'#c24f51')]:
        ax.add_patch(Circle(c['centers_mm'][0],c['radii_mm'][0],fill=False,ec=color,lw=1.5))
    ax.add_patch(Circle(candidates[0]['centers_mm'][1],candidates[0]['radii_mm'][1],fill=False,ec='#777777',lw=1))
    a0=np.asarray(candidates[0]['centers_mm'][0]);a1=np.asarray(candidates[-1]['centers_mm'][0])
    ax.annotate('',xy=a1,xytext=a0,arrowprops=dict(arrowstyle='->',color='#c24f51',lw=1.5))
    ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',title='位置干预：原位 → 上移4.5 mm',xlabel='x (mm)',ylabel='y (mm)')
    ax.text(.03,.97,'灰圈：原位；红圈：上移后',transform=ax.transAxes,va='top',fontsize=7)
    titles=['两杆共同参与的事件比例','最上方 SCL9 参与比例','被归为 TA 的事件比例']
    for k,ax in enumerate(axes[1:]):
        for si,seed in enumerate(p['seeds']):
            values=[];counts=[]
            for c in candidates:
                allrow=next(r for r in obs if r['candidate']==c['id'] and r['seed']==seed and r['mode']=='ALL')
                tarow=next(r for r in obs if r['candidate']==c['id'] and r['seed']==seed and r['mode']=='TA')
                value=allrow['both_rods'] if k==0 else float(next(r['participation'] for r in contacts if r['candidate']==c['id'] and int(r['seed'])==seed and r['mode']=='ALL' and r['contact']=='SCL9')) if k==1 else tarow['n']/allrow['n']
                values.append(value);counts.append(allrow['n'])
            ax.plot(xx,values,marker='o',linestyle='-' if si==0 else '--',color=f'C{si}',label=f'噪声 {seed}')
            if k==2:
                for x,y,n in zip(xx,values,counts):ax.annotate(f'n={n}',(x,y),xytext=(1,7 if si==0 else -13),textcoords='offset points',fontsize=7,color=f'C{si}')
        reference=summary['patient_reference']['ALL']['both_rods'] if k==0 else float(next(r['patient_participation'] for r in contacts if r['mode']=='ALL' and r['contact']=='SCL9')) if k==1 else summary['patient_reference']['TA']['n']/summary['patient_reference']['ALL']['n']
        ax.axhline(reference,color='black',ls=':',lw=1,label='患者 FIT')
        ax.set(title=titles[k],xlabel='左核上移量 (mm)',xticks=xx,xlim=(-.2,4.9),ylim=(-.08,1.06),yticks=[0,.25,.5,.75,1],yticklabels=['0%','25%','50%','75%','100%'])
    axes[1].legend(fontsize=7,loc='upper left')
    fig.suptitle('左核上移带来明确取舍：两杆共同参与增加，SCL9仍缺失，TA样式变少\n同一拓扑、两条配对噪声，各20秒；固定半径、连接与输入规律，主系列关闭外加慢I状态',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(F/f'position_key_response.{ext}',dpi=180)
    plt.close(fig)
    (F/'README.md').write_text('\n\n'.join(f'### position_key_response.{ext}\n\n左图为实际SEEG几何和固定半径的位置干预，灰圈原位、红圈上移4.5mm；后三图为不分模式的两杆参与、SCL9参与和TA标签比例。蓝/橙线分别是同一网络的两条噪声，患者FIT为虚线；n是每运行合格事件数。\n\n**关注点**：SCL某处参与不等于上杆恢复，TA标签缺失也不等于左核静默；本图没有证明新拓扑上的稳定性。' for ext in ['png','pdf'])+'\n')

if __name__=='__main__':main()
