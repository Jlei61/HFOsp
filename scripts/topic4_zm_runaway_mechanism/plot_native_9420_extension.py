"""Spatial persistence during the continuous 72-s held-Z continuation."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    q=read(OUT/'native_9420_matched_extension_readout.json');assert q['status']=='COMPLETE'
    field=np.concatenate([np.load(p)['field_E_hz'] for p in q['sources']])
    s=model();windows=[(8000,12000),(32000,36000),(68000,72000)]
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(9.1,6.2),sharex=True,sharey=True)
    fig.subplots_adjust(left=.18,right=.87,bottom=.1,top=.91,wspace=.13,hspace=.2)
    summaries=[]
    for j,(lo,hi) in enumerate(windows):
        f=field[lo:hi].reshape(-1,10,400).mean(1)
        vals=[f.mean(0),(f>=50).mean(0)]
        for i,ax in enumerate(axs[:,j]):
            im=ax.imshow(vals[i].reshape(20,20),origin='lower',extent=[0,20,0,20],
                cmap='magma' if i==0 else 'viridis',vmin=0,vmax=500 if i==0 else 1)
            ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            if i==0:ax.text(.5,1.07,f'{lo//1000}–{hi//1000} s',ha='center',transform=ax.transAxes)
            if i==1:ax.set_xlabel('x (mm)')
            if j==0:
                ax.set_ylabel('y (mm)')
                ax.text(-.55,.5,'Mean E rate' if i==0 else 'Time above 50 Hz',
                    transform=ax.transAxes,rotation=90,ha='center',va='center')
                ax.text(-.55,1.07,'A' if i==0 else 'B',transform=ax.transAxes,fontweight='bold',fontsize=16)
            for c,label in zip(s.geo['centers_mm'],'AB'):
                ax.add_patch(Circle(c,1.5,fill=False,ec='#20c4cf',lw=1.1))
                ax.text(c[0],c[1]+2,label,color='#20c4cf',ha='center',fontsize=9)
            if j==2:
                cax=fig.add_axes([.91,.565 if i==0 else .13,.017,.30])
                fig.colorbar(im,cax=cax,label='E rate (Hz)' if i==0 else 'Fraction',ticks=[0,250,500] if i==0 else [0,.5,1])
        summaries.append(dict(window_ms=[lo,hi],spatial_mean_E_hz=vals[0],fraction_time_above50Hz=vals[1]))
    folder=OUT/'figures';name='fig_native9420_Z_held_72s_spatial'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(folder/f'{name}.json',dict(source=str(OUT/'native_9420_matched_extension_readout.json'),
        D=q['D'],global_Z=q['global_Z'],Z='held',M='dynamic',windows=summaries,
        time_origin='Conditional rate intervention, not original SNN clock',
        description='Spatial means and duty over three predetermined 4-s windows of one continuous rate trajectory; not snapshots or bifurcation branches',
        human_visual_acceptance='PENDING'))
    p=folder/'README.md';text=p.read_text();heading=f'### {name}.png / .pdf / .svg'
    if heading not in text:p.write_text(text+'\n'+heading+'\n固定原生9.42秒空间Z场，在同一条rate轨迹的8–12、32–36和68–72秒窗口比较空间活动，M始终动态。上排为平均E率，下排为10ms分箱中达到50Hz的时间比例，时间从条件干预开始计算。**关注点**：局部持续活动是否随长时间延长而终止或扩展；这是有限时间空间统计，不是瞬时快照或已认证分岔。\n')
    print(folder/f'{name}.png',flush=True)


if __name__=='__main__':main()
