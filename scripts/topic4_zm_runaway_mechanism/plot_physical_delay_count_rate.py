"""Same-clock native/legacy/corrected rate comparison after complete readout."""
from common import OUT,BASE,model,np,read,write,log
from physical_delay_count_rate import DEST,BASELINE
from audit_fine_rate_frozen_Z_fields import moving_mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    assert read(DEST/'independent_comparison.json')['status']=='READOUT_AUDIT_PASS'
    assert read(DEST/'scientific_comparison.json')['status']=='COMPARISON_COMPLETE'
    s=model(40);native=np.load(BASE/'native_reference/seed9108401_readouts.npz')
    checkpoints=read(BASE/'native_reference/checkpoint_projections.json')
    sources=[None,BASELINE,DEST/'recorded_drive_binomial_seed1']
    names=['Native SNN','Original rate model','Corrected rate model'];colors=['#202020','#b5813e','#237d88']
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(10.4,9.4));gs=fig.add_gridspec(4,3,left=.10,right=.85,top=.94,bottom=.075,
        hspace=.48,wspace=.32,height_ratios=[.85,.7,1,1]);snapshots=[]
    for col,(source,name,color) in enumerate(zip(sources,names,colors)):
        if source is None:
            t=native['t'];rate=native['allE'];field=native['rate_cells'];ts=np.array(sorted(map(int,checkpoints)))
            D=np.array([checkpoints[str(k)]['D'] for k in ts]);M=np.array([checkpoints[str(k)]['mean_M']*.0005 for k in ts]);ls='none';marker='o'
        else:
            z=np.load(source/'trajectory.npz');t=z['time_ms'];rate=z['global_E_hz'];field=z['field_E_hz'];ts=z['state_time_ms'];D=z['D'];M=z['M_current'][:,s.E]@s.mean_weights;ls='-';marker=None
        ax=fig.add_subplot(gs[0,col]);ax.plot(t/1000,moving_mean(rate,10),color=color,lw=.65)
        ax.set(xlim=(0,12.5),ylim=(0,510),xticks=[0,4,8,12],yticks=[0,250,500],xlabel='Time (s)')
        ax.text(.5,1.12,name,ha='center',transform=ax.transAxes)
        for tm in [4.025,9.870]:ax.axvline(tm,color='black',ls=':',lw=.55)
        if col==0:
            ax.set_ylabel('Global E rate (Hz)');ax.text(-.32,1.12,'A',transform=ax.transAxes,fontsize=17,fontweight='bold')
        else:ax.tick_params(labelleft=False)
        ax=fig.add_subplot(gs[1,col]);ax.plot(ts/1000,D,color='#202020',lw=1,ls=ls,marker=marker,ms=3)
        other=ax.twinx();other.spines['right'].set_visible(True);other.plot(ts/1000,M,color='#9053a2',lw=1,ls=ls,marker=marker,ms=3)
        ax.set(xlim=(0,12.5),ylim=(0,1),xticks=[0,4,8,12],yticks=[0,.5,1],xlabel='Time (s)');other.set(ylim=(0,.3),yticks=[0,.15,.3])
        if col==0:
            ax.set_ylabel(r'$D=1-\langle Z_E\rangle$');ax.text(-.32,1.12,'B',transform=ax.transAxes,fontsize=17,fontweight='bold')
        else:ax.tick_params(labelleft=False)
        if col==2:other.set_ylabel(r'$\eta_M\langle M\rangle$ (mV)',color='#9053a2')
        else:other.tick_params(labelright=False)
        other.tick_params(axis='y',colors='#9053a2')
        for row,tm in enumerate([4025.,9870.],2):
            ax=fig.add_subplot(gs[row,col]);mask=(t>=tm-25)&(t<tm+25);assert mask.sum()==50
            im=ax.imshow(field[mask].mean(0).reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500)
            ax.set(xticks=[0,10,20],yticks=[0,10,20])
            if row==3:ax.set_xlabel('x (mm)')
            else:ax.tick_params(labelbottom=False)
            if col==0:
                ax.set_ylabel(f'{tm/1000:.3f} s\ny (mm)');ax.text(-.32,1.03,'CD'[row-2],transform=ax.transAxes,fontsize=17,fontweight='bold')
            else:ax.tick_params(labelleft=False)
            for center,letter in zip(s.geo['centers_mm'],'AB'):
                ax.add_patch(Circle(center,1.75,fill=False,color='#20c4cf',lw=.9));ax.text(center[0],center[1]+2.2,letter,color='#20c4cf',ha='center',fontsize=9)
            snapshots.append(dict(condition=name,window_ms=[tm-25,tm+25],samples=50))
    cax=fig.add_axes([.91,.075,.013,.445]);bar=fig.colorbar(im,cax=cax,ticks=[0,250,500]);bar.set_label('E rate (Hz)')
    folder=OUT/'figures';name='fig_physical_delay_count_rate'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(folder/f'{name}.json',dict(source=str(DEST/'scientific_comparison.json'),snapshots=snapshots,
        sources=[str(p) if p else str(BASE/'native_reference/seed9108401_readouts.npz') for p in sources],
        meaning='Original andcorrectedfinite-count ratefield: only actual0.1msdelaylag inprivatevariance changed; commonphysicalclock, noevent/onsetrealignment. ZandMbothdynamic.',
        native_slow='Exactnativecheckpointobservations, unconnected dots, not an interpolated trajectory.',
        scope='Directcorrespondence diagnostic, not abifurcationplot or scientificacceptance.',agent_PNG_PDF_check='PENDING',human_visual_acceptance='PENDING'))
    f=folder/'README.md';heading=f'### {name}.png / .pdf / .svg'
    if heading not in f.read_text():
        with f.open('a') as h:h.write('\n'+heading+'\n三列为原生SNN、修复前有限计数率模型、修复延迟方差单位后的同一率模型；后两列仅改变私有方差分拆使用的延迟格距。所有轨迹Z/M均动态，按同一物理时钟显示全局率、D/M及4.025与9.870秒的50ms空间场，原生慢变量只画真实检查点，不按事件对齐。**关注点**：已确认的单位修复是否改善间期传播、自由Z耗减路径和进入持续活动，而非仅使某一个指标接近；这是对应检查图，不是分岔图。\n')
    log('PHYSICAL DELAY FIGURE',folder/f'{name}.png')


if __name__=='__main__':main()
