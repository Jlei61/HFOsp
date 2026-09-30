"""Native SNN matched-clock spatial fields; no trajectory is a bifurcation branch."""
from native_same_history_audit import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    result=N.read(PAIR/'result.json');assert result['status']=='COMPLETE'
    assert result['dynamic_replay_qa']=='PASS'
    z=np.load(PAIR/'paired_fields.npz');counts=z['cell_counts'];windows=[]
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(9.0,6.1),sharex=True,sharey=True)
    fig.subplots_adjust(left=.15,right=.86,bottom=.1,top=.92,wspace=.14,hspace=.19)
    # These original Fig.5 clocks are fixed before reading the paired outcome.
    centers=[9420,9870,10070]
    for i,condition in enumerate(['held','dynamic']):
        field=z[f'field_{condition}_Hz']
        for j,t in enumerate(centers):
            ax=axes[i,j];lo,hi=t-9000-25,t-9000+25
            rate=field[lo:hi].mean(0)
            im=ax.imshow(rate.reshape(20,20),origin='lower',extent=[0,20,0,20],
                         cmap='magma',vmin=0,vmax=500)
            ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            if i==0:ax.text(.5,1.07,f'{t/1000:.3f} s',ha='center',transform=ax.transAxes)
            if i==1:ax.set_xlabel('x (mm)')
            if j==0:
                ax.set_ylabel('y (mm)')
                ax.text(-.46,.5,'Z held' if condition=='held' else 'Z dynamic',
                        transform=ax.transAxes,rotation=90,va='center',ha='center')
                ax.text(-.45,1.05,'A' if i==0 else 'B',transform=ax.transAxes,
                        fontweight='bold',fontsize=16)
            for c,label in zip(z['centers_mm'],'AB'):
                ax.add_patch(Circle(c,1.5,fill=False,ec='#20c4cf',lw=1.1))
                ax.text(c[0],c[1]+2,label,color='#20c4cf',ha='center',fontsize=9)
            windows.append(dict(condition=condition,absolute_window_ms=[t-25,t+25],
                global_E_Hz=float(rate@(counts/counts.sum()))))
    cax=fig.add_axes([.89,.20,.018,.58]);fig.colorbar(im,cax=cax,label='E rate (Hz)',ticks=[0,250,500])
    folder=OUT/'figures';name='fig_native_SNN_same_history_Z_feedback'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    N.write(folder/f'{name}.json',dict(source=str(PAIR/'result.json'),windows=windows,
        same_complete_initial_state_s=9.,M='dynamic in both arms',
        selection='Original Fig.5 preentry, first high entry, and entry confirmation clocks; identical50ms windows in both conditions',
        human_visual_acceptance='PENDING'))
    readme=folder/'README.md';text=readme.read_text();heading=f'### {name}.png / .pdf / .svg'
    description=('原生SNN从原图9.000秒的同一完整状态续接至12.500秒，包含相同外部时钟、随机流、延迟历史和M初值；上排仅固定Z的更新，下排Z动态，M始终动态。'
        '动态Z组须逐位复现原参考轨迹后才生成此图；三列统一取原图9.420、9.870、10.070秒的50ms空间窗，色标为0–500Hz。'
        '**关注点**：两组在相同绝对时刻的空间募集差别；这是一条原生历史上的有限窗干预，不独立确定分岔类型或长期吸引态。\n')
    if heading not in text:readme.write_text(text+'\n'+heading+'\n'+description)
    print(folder/f'{name}.png',flush=True)


if __name__=='__main__':main()
