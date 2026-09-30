"""Spatial consequences of two original-clock native Z clamps."""
from native_same_history_audit import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    q=N.read(PAIR/'late_clamp_result.json');assert q['status']=='COMPLETE'
    z=np.load(PAIR/'late_clamp_fields.npz');centers=[10070,10390,12000]
    conditions=[('held_9420',9420,'Z held at 9.420 s'),
                ('held_9870',9870,'Z held at 9.870 s'),
                ('dynamic_9000',9000,'Z dynamic')]
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,3,figsize=(9.0,8.1),sharex=True,sharey=True)
    fig.subplots_adjust(left=.18,right=.86,bottom=.08,top=.94,wspace=.12,hspace=.16)
    records=[]
    for i,(key,start,label) in enumerate(conditions):
        fields=z[f'field_{key}_Hz']
        for j,t in enumerate(centers):
            window=fields[t-start-25:t-start+25]
            assert len(window)==50
            rate=window.mean(0);ax=axes[i,j]
            im=ax.imshow(rate.reshape(20,20),origin='lower',extent=[0,20,0,20],
                         cmap='magma',vmin=0,vmax=500)
            ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            if i==0:ax.text(.5,1.07,f'{t/1000:.3f} s',ha='center',transform=ax.transAxes)
            if i==2:ax.set_xlabel('x (mm)')
            if j==0:
                ax.set_ylabel('y (mm)')
                ax.text(-.51,.5,label,transform=ax.transAxes,rotation=90,
                        ha='center',va='center')
                ax.text(-.50,1.04,'ABC'[i],transform=ax.transAxes,fontweight='bold',fontsize=16)
            for c,name in zip(z['centers_mm'],'AB'):
                ax.add_patch(Circle(c,1.5,fill=False,ec='#20c4cf',lw=1.1))
                ax.text(c[0],c[1]+2,name,color='#20c4cf',ha='center',fontsize=9)
            records.append(dict(condition=key,absolute_window_ms=[t-25,t+25],
                mean_E_Hz=float(np.average(rate,weights=z['cell_counts']))))
    cax=fig.add_axes([.89,.22,.018,.56])
    fig.colorbar(im,cax=cax,label='E rate (Hz)',ticks=[0,250,500])
    folder=OUT/'figures';name='fig_native_SNN_late_Z_clamps'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    N.write(folder/f'{name}.json',dict(source=str(PAIR/'late_clamp_result.json'),
        selection='Original first high-entry confirmation10.07s, original broad-entry10.39s, and later12s; centered50ms windows identical between arms',
        windows=records,M='dynamic in all conditions',model='Native SNN',
        claim='Finite-horizon intervention, not a bifurcation or rate-model branch',
        human_visual_acceptance='PENDING'))
    readme=folder/'README.md';text=readme.read_text();heading=f'### {name}.png / .pdf / .svg'
    if heading not in text:
        readme.write_text(text+'\n'+heading+'\n'
            '原生SNN分别在原图9.420秒和9.870秒的完整状态固定Z，此后M、快速状态及原始外部输入继续更新；下排为已逐位复现的Z动态参考。'
            '三列统一显示10.070、10.390和12.000秒的50ms空间窗口，共用0–500Hz色标。'
            '**关注点**：在进入前后固定Z，是否阻止后续全局持续募集；这些是同一原生历史上的干预，不是三个独立样本，也不是分岔支或永久不进入的证明。\n')
    print(folder/f'{name}.png',flush=True)


if __name__=='__main__':main()
