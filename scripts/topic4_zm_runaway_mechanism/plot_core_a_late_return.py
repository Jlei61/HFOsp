"""Actual long-event termination and its local numerical control, no SN label."""
from common import OUT,np,read,write,model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Patch


def main():
    root=OUT/'core_a_bifurcation_type_20260924'
    source=root/'late_return_counterexample';audit=read(source/'result.json')
    assert audit['status']=='AUDIT_PASS'
    control=read(source/'refinement/result.json')
    assert control['status'] in ['LOCAL_RETURN_ROBUST_TO_TWO_STEP_HALVINGS','LOCAL_RETURN_MESH_CHECK_NOT_PASSED']
    s=model(40);actual=root/'actual_sustained_growth/coarse'
    fields=np.concatenate([np.load(actual/f'block{b:02d}.npz')['field_E_hz'] for b in [0,1]])
    times=[64650,64820,64880,65130]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(10.6,8.0),layout='constrained')
    gs=fig.add_gridspec(3,4,height_ratios=[.7,1.2,1.2])
    ax=fig.add_subplot(gs[0,:]);ax.broken_barh([(0,70)],(.65,.24),facecolors='#c0447b',edgecolors='none')
    for a,b in audit['quiet_intervals_ms']:
        ax.broken_barh([(a/1000,(b-a)/1000)],(.65,.24),facecolors='white',edgecolors='none')
    ax.set_xlim(0,70);ax.set_ylim(.55,1.0);ax.set_yticks([]);ax.set_xlabel('Time under fixed Z (s)');ax.set_ylabel('Δt = 0.05 ms',rotation=0,ha='right',va='center')
    ax.set_xticks([0,20,40,60,70]);ax.spines[['left','right','top']].set_visible(False)
    ax.text(0,1.1,'A',transform=ax.transAxes,fontweight='bold',fontsize=17)
    ax.annotate('',xy=(64.830,.96),xytext=(.467,.96),arrowprops=dict(arrowstyle='<->',lw=1.2,color='black'))
    ax.text(32.65,.975,'64.363 s',ha='center',va='bottom')
    ax.legend(handles=[Patch(facecolor='#c0447b',label='Active'),Patch(facecolor='white',edgecolor='black',label='Quiet ≥20 ms')],
        loc='upper left',bbox_to_anchor=(.54,1.66),ncol=2,frameon=False)
    spatial=[]
    for j,tm in enumerate(times):
        ax=fig.add_subplot(gs[1,j]);field=fields[tm-60000-25:tm-60000+25].mean(0);spatial.append(field)
        im=ax.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],vmin=0,vmax=500,cmap='magma',interpolation='nearest')
        for k,center in enumerate(s.geo['centers_mm']):
            ax.add_patch(Circle(center,1.5,fill=False,color='#22ccd2',lw=1.3))
            ax.text(center[0],center[1]+1.8,'AB'[k],color='#22ccd2',ha='center',fontsize=11)
        ax.set_xlabel('x (mm)');ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
        if j==0:ax.set_ylabel('y (mm)')
        else:ax.set_yticklabels([])
        ax.text(0,1.06,f'{tm/1000:.3f} s',transform=ax.transAxes)
        ax.text(-.16,1.06,'BCDE'[j],transform=ax.transAxes,fontweight='bold',fontsize=17)
    cb=fig.colorbar(im,ax=fig.axes[1:5],fraction=.03,pad=.02);cb.set_label('E rate (Hz / neuron)');cb.set_ticks([0,250,500])
    ax=fig.add_subplot(gs[2,:]);styles=[('#c0447b','-'),('#159b9f','--'),('#111111',':')]
    for row,(color,ls) in zip(control['rows'],styles):
        dt=row['dt_ms'];z=np.load(source/f'refinement/local_dt{dt:g}.npz')
        ax.plot(z['time_ms']/1000,z['Core_A_smoothed_hz'],color=color,ls=ls,lw=1.6,label=f'Δt = {dt:g} ms')
    ax.set_xlim(64.65,65.4);ax.set_ylim(-8,500);ax.set_yticks([0,250,500]);ax.set_xlabel('Time under fixed Z (s)');ax.set_ylabel('Core A E rate\n(Hz / neuron)')
    ax.spines[['right','top']].set_visible(False);ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.52,1.30))
    ax.text(-.065,1.06,'F',transform=ax.transAxes,fontweight='bold',fontsize=17)
    dest=source/'figures';dest.mkdir(exist_ok=True);name='fig_core_a_long_event_returns'
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=220)
    plt.close(fig)
    write(dest/f'{name}.json',dict(source=str(source/'result.json'),D_A=audit['D_A'],Z_A=audit['Z_A'],
        ended_activity=audit['ended_activity'],spatial_times_ms=times,spatial_window_ms=50,
        spatial_windows_ms_open_left_closed_right=[[tm-25,tm+25] for tm in times],spatial_source_dt_ms=.05,
        spatial_fields_20x20=np.array(spatial).tolist(),local_refinement=control,
        meaning='Actual finite long episode, its termination and local timestep controls. No equilibrium, periodic or critical-point marker is implied.',
        line_styles='Integration-step comparison only; these are actual transient rate traces, not stable/unstable branches.',
        all_Z_held=True,all_M_dynamic=True,human_visual_acceptance='PENDING'))
    numerical=('三种步长均出现合格静息，局部返回通过两次减半控制。' if control['status']=='LOCAL_RETURN_ROBUST_TO_TWO_STEP_HALVINGS' else
        '局部返回未通过步长复验：两条细步轨迹没有出现同一合格静息，因此不能把0.05ms的具体返回时间升级为时间步收敛结论。另一个从60秒末态出发的完整0.025ms续接在64.521–64.682秒仍出现A静息，故此局部失败不等于细步轨迹永不返回。')
    (dest/'README.md').write_text('### fig_core_a_long_event_returns.png / .pdf / .svg\n\n'
        '固定同一个核A资源场、核外Z与全部其他参数，M始终动态，0.05ms同场轨迹中0.467–64.830秒的长活动随后终止。A显示该步长完整70秒观察的活动/静息，B–E为该轨迹返回附近的50ms二维放电场；F从64.650秒同一完整状态出发做两次时间步减半，线型只区分数值步长，不表示稳定性。'+numerical+'这是一张时窗与步长诊断图，不是已认定类型的分岔图，也不代表原生SNN临界值。\n\n'
        '**关注点**：原“持续侧”端点的有限观察，以及三种步长是否给出一致的局部返回；不能只看上方粗步轨迹而忽略F。PNG/PDF候选待用户人工检查。\n')


if __name__=='__main__':main()
