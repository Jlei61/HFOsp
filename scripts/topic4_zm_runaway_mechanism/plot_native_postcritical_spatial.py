"""Fixed-window spatial recruitment on the matched native-Z path."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    data=read(OUT/'native_postcritical_endpoint_audit.json')
    assert data['status']=='COMPLETE'
    # Prespecified representative fields, not selected for visual appearance.
    selected=[data['rows'][j] for j in [0,3,4,6]]
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,4,figsize=(11.7,6.2),sharex=True,sharey=True)
    fig.subplots_adjust(left=.15,right=.88,bottom=.1,top=.91,wspace=.13,hspace=.20)
    s=model();summaries=[]
    for j,row in enumerate(selected):
        path=Path(row['spatial']['source']);z=np.load(path)
        field=z['field_E_hz'][-4000:].reshape(400,10,400).mean(1)
        values=[field.mean(0),(field>=50).mean(0)]
        for i,ax in enumerate(axes[:,j]):
            im=ax.imshow(values[i].reshape(20,20),origin='lower',extent=[0,20,0,20],
                         cmap='magma' if i==0 else 'viridis',vmin=0,vmax=500 if i==0 else 1)
            ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            if i==0:
                ax.text(.5,1.07,f'$D={row["D"]:.4f}$',transform=ax.transAxes,ha='center')
            if i==1:ax.set_xlabel('x (mm)')
            if j==0:
                ax.set_ylabel('y (mm)')
                ax.text(-.48,.5,'Mean E rate' if i==0 else 'Time above 50 Hz',
                        transform=ax.transAxes,rotation=90,va='center',ha='center')
                ax.text(-.48,1.07,'A' if i==0 else 'B',transform=ax.transAxes,
                        fontweight='bold',fontsize=16)
            for c,label in zip(s.geo['centers_mm'],'AB'):
                ax.add_patch(Circle(c,1.5,fill=False,ec='#20c4cf',lw=1.1))
                ax.text(c[0],c[1]+2,label,color='#20c4cf',ha='center',fontsize=9)
            if j==3:
                cax=fig.add_axes([.91,.565 if i==0 else .13,.015,.30])
                fig.colorbar(im,cax=cax,label='E rate (Hz)' if i==0 else 'Fraction',
                             ticks=[0,250,500] if i==0 else [0,.5,1])
        summaries.append(dict(source=str(path),D=row['D'],global_Z=row['global_Z'],
                              samples=[8000,12000],category=row['canonical']['category']))
    folder=OUT/'figures';name='fig_native_path_postcritical_spatial'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(folder/f'{name}.json',dict(source=str(OUT/'native_postcritical_endpoint_audit.json'),
        panels=summaries,Z='held',M='dynamic',time_window_ms=[8000,12000],
        row_A='Spatial rate averaged over the final 4 s',
        row_B='Fraction of final-4-s non-overlapping 10-ms bins at or above 50 Hz',
        limitation='These are conditional finite-time spatial statistics, not snapshots or bifurcation branches.',
        human_visual_acceptance='PENDING'))
    readme=folder/'README.md';text=readme.read_text();heading=f'### {name}.png / .pdf / .svg'
    description=('固定原生检查点插值路径上的四个Z场，从同一完整周期初态分别运行12秒，M动态，延迟历史采用相同端点积分。'
        '上排为末4秒空间平均E率，下排为每格在10ms分箱中达到50Hz的时间比例，表示哪些区域持续参与；D=1−全E加权平均Z。'
        '**关注点**：平均率的增加与持续募集范围是否同步变化；这是匹配条件下的有限窗空间统计，不是瞬时快照、周期支或分岔认证。\n')
    if heading not in text:readme.write_text(text+'\n'+heading+'\n'+description)
    print(folder/f'{name}.png',flush=True)


if __name__=='__main__':main()
