"""Common-time native spatial snapshots for the four feedback conditions."""
from native_ZM_factorial_audit import DEST,N,OUT,np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    q=N.read(DEST/'result.json');assert q['status']=='COMPLETE'
    z=np.load(DEST/'fields.npz')
    arms=[('Zdynamic_Mdynamic','Z dynamic, M dynamic'),('Zdynamic_Mheld','Z dynamic, M held'),
          ('Zheld_Mdynamic','Z held, M dynamic'),('Zheld_Mheld','Z held, M held')]
    times=[10390,12000]
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(4,2,figsize=(6.9,10.6),sharex=True,sharey=True)
    fig.subplots_adjust(left=.24,right=.83,bottom=.065,top=.96,wspace=.12,hspace=.15)
    rows=[]
    for i,(arm,label) in enumerate(arms):
        field=z[arm+'_field_Hz']
        for j,t in enumerate(times):
            sample=field[t-9000-25:t-9000+25];assert len(sample)==50
            rate=sample.mean(0);ax=axes[i,j]
            im=ax.imshow(rate.reshape(20,20),origin='lower',extent=[0,20,0,20],
                cmap='magma',vmin=0,vmax=500)
            ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            if i==0:ax.text(.5,1.07,f'{t/1000:.3f} s',ha='center',transform=ax.transAxes)
            if i==3:ax.set_xlabel('x (mm)')
            if j==0:
                ax.set_ylabel('y (mm)')
                ax.text(-.62,.5,label,rotation=90,ha='center',va='center',transform=ax.transAxes)
                ax.text(-.62,1.04,'ABCD'[i],fontsize=16,fontweight='bold',transform=ax.transAxes)
            for c,name in zip(z['centers_mm'],'AB'):
                ax.add_patch(Circle(c,1.5,fill=False,ec='#20c4cf',lw=1.1))
                ax.text(c[0],c[1]+2,name,color='#20c4cf',ha='center',fontsize=9)
            rows.append(dict(arm=arm,absolute_window_ms=[t-25,t+25],
                mean_E_Hz=float(np.average(rate,weights=z['cell_counts']))))
    cax=fig.add_axes([.875,.30,.022,.42]);fig.colorbar(im,cax=cax,label='E rate (Hz)',ticks=[0,250,500])
    folder=OUT/'figures';name='fig_native_SNN_ZM_factorial'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    N.write(folder/f'{name}.json',dict(source=str(DEST/'result.json'),model='Native SNN',
        selection='Original broad-entry10.39s and common12s, centered50ms windows; fixed before figure generation',
        rows=rows,scope='One original9s history, four feedback interventions. No bifurcation type inferred.',
        human_visual_acceptance='PENDING'))
    readme=folder/'README.md';text=readme.read_text();heading=f'### {name}.png / .pdf / .svg'
    if heading not in text:
        readme.write_text(text+'\n'+heading+'\n'
            '原生SNN从同一个9秒完整状态出发，交叉比较Z、M各自动态或冻结更新；冻结值仍参与膜电流。'
            '两列为共同10.390及12.000秒的50ms空间窗口，统一0–500Hz色标；事件与持续占据的整窗结果另见native_ZM_factorial/summary.csv。'
            '**关注点**：这次高活动进入及空间募集是否需要M继续更新，以及固定Z的效果是否依赖M动态；四组来自同一历史，不是独立样本或数学分岔证明。\n')
    print(folder/f'{name}.png',flush=True)


if __name__=='__main__':main()
