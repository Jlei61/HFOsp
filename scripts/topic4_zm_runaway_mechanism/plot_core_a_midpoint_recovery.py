"""Finite-time local resource controls, explicitly not a bifurcation plot."""
from common import OUT,np,read,write,model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    parent=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap/midpoint_short_side_search'
    labels=['restored_A_fine','native4000_fine','native8000_fine','native9000_fine']
    colors=['#237c88','#7c76a5','#c87522','#b1475a'];s=model(40)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(10.3,5.9),layout='constrained')
    gs=fig.add_gridspec(2,5,height_ratios=[1.25,1],width_ratios=[1,1,1,1,.055])
    top=fig.add_subplot(gs[0,:4]);rows=[]
    for i,label in enumerate(labels):
        folder=parent/label;a=read(folder/'independent_audit.json');assert a['status']=='INDEPENDENT_READOUT_PASS'
        c=read(folder/'numerical_contract.json');assert c['method']=='exponential_midpoint' and c['dt_ms']==.025
        events=a['regions'][1]['activities'];y=3-i
        for event in events:
            top.broken_barh([(event['start_ms']/1000,event['duration_ms']/1000)],(y-.17,.34),facecolors=colors[i])
            if event['right_censored']:top.plot(event['end_ms']/1000,y,marker='>',color=colors[i],clip_on=False)
        complete=[e for e in events if not e['left_censored'] and not e['right_censored'] and e['start_ms']>=2000]
        event=max(complete,key=lambda e:e['duration_ms'])
        with np.load(folder/'trajectory.npz') as z:
            tm=int((event['start_ms']+event['end_ms'])/2)
            use=(z['time_ms']>tm-25)&(z['time_ms']<=tm+25);assert use.sum()==50
            field=z['field_E_hz'][use].astype(float).mean(0)
        top.plot(tm/1000,y+.27,marker='v',color='black',markersize=5)
        ax=fig.add_subplot(gs[1,i]);im=ax.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],
            cmap='magma',vmin=0,vmax=500,interpolation='nearest')
        ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20]);ax.set_xlabel('x (mm)')
        if i==0:ax.set_ylabel('y (mm)')
        else:ax.set_yticklabels([])
        ax.text(.5,1.03,f'$D_A={a["D_A"]:.3f}$\n{tm/1000:.3f} s',ha='center',transform=ax.transAxes,color=colors[i])
        for name,center in zip('AB',s.geo['centers_mm']):
            ax.add_patch(Circle(center,1.5,fill=False,color='#20ccd0',lw=1.1))
            ax.text(center[0],center[1]+1.9,name,ha='center',color='#20ccd0',fontsize=10)
        rows.append(dict(label=label,D_A=a['D_A'],Z_A=a['Z_A'],selected_event=event,time_ms=tm,field_E_hz=field.tolist()))
    top.set_xlim(0,5);top.set_ylim(-.5,3.65);top.set_xlabel('Time after local Z intervention (s)')
    top.set_yticks([3,2,1,0],[f'$D_A={r["D_A"]:.3f}$' for r in rows]);top.spines['left'].set_visible(False);top.tick_params(axis='y',length=0)
    top.text(-.15,1.02,'A',transform=top.transAxes,weight='bold',fontsize=17)
    top.plot([],[],marker='v',color='black',ls='',label='Spatial sample')
    top.legend(loc='lower right',bbox_to_anchor=(1.,1.01),frameon=False)
    fig.axes[1].text(-.25,1.17,'B',transform=fig.axes[1].transAxes,weight='bold',fontsize=17)
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[1,4]));cb.set_ticks([0,250,500]);cb.set_label('E rate (Hz / neuron)')
    dest=parent/'figures';dest.mkdir(exist_ok=True);name='fig_local_resource_recovery'
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=220)
    plt.close(fig)
    write(dest/f'{name}.json',dict(numerical_method='exponential_midpoint_v1',dt_ms=.025,rows=rows,
        scope='Matched full initial history, outsideA native9s, allZheld and allE Mdynamic. Five-second finite-state controls, not equilibrium/cycle branches, stability, a criticalZ or bifurcation classification.',
        sample_selection='Midpoint of the longest complete CoreA activity starting after2s; each map uses50ms originalrate bins.',
        human_visual_acceptance='PENDING',model_promoted=False))
    (dest/'README.md').write_text('### fig_local_resource_recovery.png / .pdf / .svg\n\n同一完整初态下，只改变核 A 的 Z 空间场；核外固定原生9秒场，所有兴奋性 M 动态。上方是5秒内核 A 活动段，三角表示下方50ms空间取样时刻，右箭头只表示活动被记录终点截断。使用通过短窗收敛检查的指数中点法、dt=0.025ms；这些有限窗对照不是分岔分支或临界点证明。\n\n**关注点**：恢复核 A 资源后是否重新出现反复终止的短活动，以及较深耗减时活动集中在哪个核和外围；分岔类型需另行周期和稳定性验收，PNG/PDF待用户人工检查。\n')


if __name__=='__main__':main()
