"""Spatial readout of the completed early M-clamp counterexample.

No equilibrium/periodic branch or unverified bifurcation symbol is drawn.
"""
from common import OUT,np,read,write,model
from refractory_spatial_resolution import mapping,projections
from onset_state_continuation import regional_weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    source=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap/first_long_M_intervention_from2800'
    audit=read(source/'independent_audit.json');assert audit['status']=='PASS'
    s=model(40);coarse=model(20);parent,_=mapping(coarse,s);P,count=projections(s,coarse,parent)[20]
    W=regional_weights(s);times=[2900,6220,7730];fields=[];segments=[];raw=[];checks=[]
    for j,label in enumerate(['dynamic_M','held_M']):
        z=np.load(source/label/'trajectory.npz');rate=z['group_rate_hz'].astype(float);t=z['elapsed_time_ms']
        assert abs(np.average(z['Z'][s.E&(s.geo['group_region']==0)],weights=s.sizes[s.E&(s.geo['group_region']==0)])-.7)<1e-12
        q=audit['arms'][j]['quiet_intervals'];active=[];start=2800
        for row in q:
            if row['start_ms']>start:active.append([start,row['start_ms']])
            start=row['end_ms']
        if start<7800:active.append([start,7800])
        segments.append(active);row_fields=[]
        for tm in times:
            use=(t>tm-25)&(t<=tm+25);assert use.sum()==50
            group=rate[use].mean(0);field=P@group
            err=float(abs(field@(count/count.sum())-group@W[0]));assert err<1e-10
            row_fields.append(field);checks.append(dict(label=label,time_ms=tm,weighted_global_error_hz=err,
                cells_above_display_max=int((field>500).sum())))
        fields.append(row_fields);raw.append(dict(label=label,source=str(source/label/'trajectory.npz'),
            windows_ms=[[tm-25,tm+25] for tm in times],fields_E_hz=[f.tolist() for f in row_fields]))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(9.2,7.0),layout='constrained')
    gs=fig.add_gridspec(3,4,height_ratios=[.7,1,1],width_ratios=[1,1,1,.055],hspace=.09)
    ax=fig.add_subplot(gs[0,:3]);colors=['#7350A0','#C47922']
    for j,items in enumerate(segments):
        y=1-j
        for a,b in items:ax.broken_barh([(a/1000,(b-a)/1000)],(y-.13,.26),facecolors=colors[j])
        a,b=items[0];ax.text((a+b)/2000,y+.18,f'{(b-a)/1000:.2f} s',ha='center',color=colors[j])
    ax.set_xlim(2.8,7.8);ax.set_ylim(-.4,1.65);ax.set_xticks([3,4,5,6,7]);ax.set_xlabel('Time (s)')
    ax.set_yticks([1,0],['M dynamic','M held (2.8 s)']);ax.tick_params(axis='y',length=0)
    ax.spines['left'].set_visible(False)
    ax.text(-.13,1.02,'A',transform=ax.transAxes,fontweight='bold',fontsize=16)
    ax.text(1.,1.03,r'$Z_A=0.700$',transform=ax.transAxes,ha='right')
    for j in range(2):
        for k,tm in enumerate(times):
            panel=fig.add_subplot(gs[j+1,k]);field=fields[j][k]
            im=panel.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],vmin=0,vmax=500,cmap='magma',interpolation='nearest')
            panel.set_xticks([0,10,20]);panel.set_yticks([0,10,20]);panel.set_aspect('equal')
            panel.set_xlabel('x (mm)' if j==1 else '')
            panel.set_ylabel('y (mm)' if k==0 else '')
            if j==0:panel.text(.5,1.04,f'{tm/1000:.3f} s',transform=panel.transAxes,ha='center')
            if k==0:
                panel.text(-.21,1.04,'BC'[j],transform=panel.transAxes,fontweight='bold',fontsize=16)
                panel.text(.02,.97,['M dynamic','M held'][j],transform=panel.transAxes,va='top',color='white',fontsize=10)
            else:panel.set_yticklabels([])
            for name,center in zip('AB',s.geo['centers_mm']):
                panel.add_patch(Circle(center,1.5,fill=False,color='#20CCD0',lw=1.1))
                panel.text(center[0],center[1]+1.9,name,color='#20CCD0',ha='center',fontsize=10)
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[1:,3]));cb.set_ticks([0,250,500]);cb.set_label('E rate (Hz / neuron)')
    dest=source/'figures';dest.mkdir(exist_ok=True);name='fig_M_clamp_spatial_return'
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=220)
    plt.close(fig)
    write(dest/f'{name}.json',dict(source=str(source),independent_audit='PASS',
        question='Does holding M before the first long activity eliminate its return to low local activity?',
        activity_bars='CoreA10ms-smoothed5Hz activity separated by at least20ms quiet, originaltimeaxis. Final activities may be rightcensored.',
        spatial_windows='Mean over50 original1ms bins ending at t+25ms (t-25,t+25]. Same clock windows in both arms.',
        source_group_grid=40,display_grid=20,E_cell_weighted=True,projection_checks=checks,
        fields=raw,first_complete_durations_ms=[3290,4823],
        scope='Paired finite conditional rate-model intervention. EveryZ held; everyM dynamic or held at original2.8s spatialfield. Not a bifurcation diagram, attractor certificate or a native-SNN intervention.',
        human_visual_acceptance='PENDING',model_promoted=False))
    (dest/'README.md').write_text('### fig_M_clamp_spatial_return.png / .pdf / .svg\n\n'
        '上排比较从同一个2.8s完整状态出发、M正常变化或固定为当时空间场后，Core A的活动时段；两臂完整Z场均保持，Z_A=0.700。下两排给出同一三个时刻、同一50ms窗及0–500Hz色标的二维E活动场，不叠加振荡轨迹或未证实的临界点。固定M使首个长活动从3.290s延长至4.823s，但仍会回落，因此该事件的结束不以M继续积累为必要条件；这是一次条件率模型的有限窗干预，不是完整分岔判型或原生SNN结果。\n\n'
        '**关注点**：6.220s时两臂的局部状态差别，以及7.730s时固定M的CoreA已回落；CoreB/外围仍可活动，局部低活动不等于全局静息。候选PNG/PDF待用户人工检查。\n')


if __name__=='__main__':main()
