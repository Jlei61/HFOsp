"""Fixed-clock spatial consequences of the registered Z-field interventions."""
from common import OUT, model, np, read, write, log
from fine_rate_frozen_Z_fields import DEST
from audit_fine_rate_frozen_Z_fields import moving_mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    audit=read(DEST/'independent_comparison.json')
    assert audit['status']=='READOUT_AUDIT_PASS' and len(audit['rows'])==5
    s=model(40); labels=read(DEST/'contract.json')['arms']
    names=['Dynamic Z','Rate Z, 9.00 s','SNN Z, 9.00 s','SNN Z, 9.42 s','SNN Z, 9.87 s']
    colors=['#202020','#776a56','#287f8b','#be852c','#b74b5b']
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(14.5,7.4))
    grid=fig.add_gridspec(3,5,left=.075,right=.923,top=.925,bottom=.082,
                          height_ratios=[.85,1,1],hspace=.38,wspace=.24)
    metadata=[]; im=None
    for col,(label,name,color) in enumerate(zip(labels,names,colors)):
        z=np.load(DEST/label/'trajectory.npz'); t=z['time_ms']; field=z['field_E_hz']
        initial=read(DEST/label/'initial.json'); mean_Z=1-initial['initial_D']
        ax=fig.add_subplot(grid[0,col]);ax.plot(t/1000,moving_mean(z['global_E_hz'],10),c=color,lw=.65)
        ax.set(xlim=(9,12.5),ylim=(0,510),xticks=[9,10,11,12],yticks=[0,250,500],xlabel='Time (s)')
        coordinate=(r'$\langle Z_E(9\,\mathrm{s})\rangle='+f'{mean_Z:.3f}'+r'$') if col==0 else (r'$\langle Z_E\rangle='+f'{mean_Z:.3f}'+r'$')
        ax.text(.5,1.10,name+'\n'+coordinate,transform=ax.transAxes,ha='center',va='bottom')
        for tm in [10.,12.]:ax.axvline(tm,color='black',ls=':',lw=.6)
        if col==0:
            ax.set_ylabel('Global E rate (Hz)')
            ax.text(-.34,1.14,'A',transform=ax.transAxes,fontweight='bold',fontsize=17)
        else:ax.tick_params(labelleft=False)
        for row,tm in enumerate([10000.,12000.],1):
            ax=fig.add_subplot(grid[row,col]);mask=(t>=tm-25)&(t<tm+25);assert mask.sum()==50
            im=ax.imshow(field[mask].mean(0).reshape(20,20),origin='lower',extent=[0,20,0,20],
                         cmap='magma',vmin=0,vmax=500)
            ax.set(xticks=[0,10,20],yticks=[0,10,20])
            if row==2:ax.set_xlabel('x (mm)')
            else:ax.tick_params(labelbottom=False)
            if col==0:
                ax.set_ylabel(f'{tm/1000:.2f} s\ny (mm)')
                ax.text(-.34,1.03,'BC'[row-1],transform=ax.transAxes,fontweight='bold',fontsize=17)
            else:ax.tick_params(labelleft=False)
            for center,name_core in zip(s.geo['centers_mm'],'AB'):
                ax.add_patch(Circle(center,1.75,fill=False,ec='#20c4cf',lw=.9))
                ax.text(center[0],center[1]+2.2,name_core,ha='center',color='#20c4cf',fontsize=9)
            metadata.append(dict(label=label,center_ms=tm,window_ms=[tm-25,tm+25],samples=50))
    cax=fig.add_axes([.946,.088,.011,.512]);bar=fig.colorbar(im,cax=cax,ticks=[0,250,500]);bar.set_label('E rate (Hz)')
    folder=OUT/'figures';name='fig_rate_same_history_frozen_Z_fields'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(folder/f'{name}.json',dict(source=str(DEST/'independent_comparison.json'),labels=labels,
        trajectories=[str(DEST/x/'trajectory.npz') for x in labels],snapshots=metadata,
        initial_history='Same own-rate full9s state; M dynamic in every condition; shared external clock and innovation keys.',
        column_semantics='First dynamicZ; other columns freeze entireZfields. Times in column identities label field sources; plotted times label shared continuation clock.',
        scope='Finite-time spatial conditional comparison. Not a bifurcation figure, no stability convention or critical point claim.',
        agent_PNG_PDF_check='PENDING',human_visual_acceptance='PENDING'))
    p=folder/'README.md';heading=f'### {name}.png / .pdf / .svg'
    if heading not in p.read_text():
        with p.open('a') as f:f.write('\n'+heading+'\n五列均从同一率模型9秒完整状态出发，第一列Z动态，其余分别固定模型自身9秒及原生9.00、9.42、9.87秒完整Z场，M均动态；未来外部输入和随机创新键相同。列名时间指Z场来源，空间行时间指共同续接时钟，图像使用统一50ms窗口和0–500Hz色标。**关注点**：固定原生前后Z场是否分别恢复自限与持续空间活动；这是有限窗对应检验，不是分岔图，也不证明普遍标量Z阈值。\n')
    log('PLOT',folder/f'{name}.png')


if __name__=='__main__':main()
