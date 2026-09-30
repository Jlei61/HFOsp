"""Show actual computed Z/M branches and spatial fields without invented cycles."""
from zm_model import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import argparse

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':14,
    'xtick.labelsize':11,'ytick.labelsize':11,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False})

def main(a):
    s=ZMRate();base=DEST/'g20';figdir=DEST/'figures';figdir.mkdir(exist_ok=True)
    branches=['D_arclength_lower','D_arclength_upper_focused','D_arclength_upper_onset_range',
        'D_arclength_upper_onset_range_v2','D_arclength_upper_onset_range_v3','D_arclength_upper_onset_range_v4']
    if a.include_gap:
        branches+=['D_gap_lower_guarded','D_gap_rate_slices','D_gap_middle_down','D_gap_middle_up','D_gap_middle_to_low']
    all_states=[];fig=plt.figure(figsize=(12.5,8.8));gs=fig.add_gridspec(3,2,width_ratios=[2.7,1],wspace=.27,hspace=.4)
    ax=fig.add_subplot(gs[:,0]);ax.set_yscale('log');ax.set_xlim(0,1);ax.set_ylim(.11,550)
    unknown=[];proven=[];proof_by_file={}
    for name in branches:
        p=base/name;data=[row for row in read(p/'result.json')['rows'] if row.get('converged',True)]
        verified={}
        for rootsfile in [p/'temporal_core_A/result.json',p/'temporal_selected_mode/result.json',p/'temporal_gap_modes/result.json']:
            if rootsfile.exists():verified.update({x['point']:x.get('equilibrium_stability') for x in read(rootsfile)['rows']})
        x=np.array([v['D'] for v in data]);y=np.array([v['global_E_hz'] for v in data])
        good=np.array([verified.get(f'point{row["index"]:04d}.npz')=='UNSTABLE' for row in data])
        ax.plot(x[~good],y[~good],'o',ms=1.9,mfc='white',mec='#333333',mew=.45,rasterized=False)
        if good.all() and name!='D_gap_rate_slices':ax.plot(x,y,'--',lw=1.7,color='#111111')
        else:ax.plot(x[good],y[good],'x',ms=3,color='#111111')
        for row in data:all_states.append(dict(**row,file=str(p/f'point{row["index"]:04d}.npz')))
        for row,yes in zip(data,good):
            if yes:proof_by_file[str(p/f'point{row["index"]:04d}.npz')]='UNSTABLE'
        unknown.extend([row for k,row in enumerate(data) if not good[k]]);proven.extend([row for k,row in enumerate(data) if good[k]])
    coarse=read(base/'conditional_dynamic_M/result.json')['rows']
    for row in coarse:
        if row['converged'] and row['direction']=='decreasing' and row['D']>.4:
            ax.plot(row['D'],row['global_E_hz'],'o',ms=2.6,mfc='white',mec='#333333',mew=.65)
    rootsfile=base/'conditional_dynamic_M/temporal_upper_21Hz/result.json'
    if rootsfile.exists():
        rows=read(rootsfile)['rows'];uu=[x for x in rows if x.get('equilibrium_stability')=='UNSTABLE']
        if uu:ax.plot([x['D'] for x in uu],[x['global_E_hz'] for x in uu],'x',ms=3.6,color='#111111')
    ax.set_xlabel(r'$D=1-\langle Z_E\rangle$');ax.set_ylabel('Global E rate (Hz / neuron)')
    ax.set_yticks([.2,1,10,100,500]);ax.set_yticklabels(['0.2','1','10','100','500'])
    ax.text(-.1,1.015,'A',transform=ax.transAxes,fontsize=20,fontweight='bold')
    handles=[Line2D([],[],ls='--',color='k',lw=1.7,label='Unstable equilibrium'),
        Line2D([],[],ls='none',marker='x',ms=5,color='k',label='Verified unstable point'),
        Line2D([],[],ls='none',marker='o',ms=4,mfc='white',mec='#333333',label='Equilibrium: stability pending'),
        Line2D([],[],ls='none',marker='*',ms=11,color='#b03c39',label='Verified equilibrium fold')]
    ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.49,.34) if a.include_gap else (.03,.76),frameon=False,fontsize=10 if a.include_gap else 11)
    lower=read(base/'D_arclength_lower/refined_folds/result.json')['rows']
    ins=ax.inset_axes([.16,.12,.32,.28] if a.include_gap else [.09,.12,.37,.28]);ll=read(base/'D_arclength_lower/result.json')['rows']
    ins.plot([x['D'] for x in ll],[x['global_E_hz'] for x in ll],'--',color='k',lw=1.25)
    for k,row in enumerate(lower):
        ins.plot(row['D'],row['global_E_hz'],'*',color='#b03c39',ms=11)
        ins.annotate(f'SN{k+1}',(row['D'],row['global_E_hz']),xytext=(-26,11),textcoords='offset points',fontsize=10)
        ax.plot(row['D'],row['global_E_hz'],'*',color='#b03c39',ms=10)
    ins.set_xlim(0,.017);ins.set_ylim(.12,.56);ins.set_xticks([0,.008,.016]);ins.tick_params(labelsize=9)
    ins.set_xlabel(r'$D$',fontsize=10);ins.set_ylabel('Hz / neuron',fontsize=10)
    upper=read(base/'D_arclength_upper_focused/refined_folds_stable_response/result.json')['rows']
    ui=ax.inset_axes([.56,.43,.36,.24]);ur=read(base/'D_arclength_upper_focused/result.json')['rows']
    ui.plot([x['D'] for x in ur],[x['global_E_hz'] for x in ur],'o',ms=2,mfc='white',mec='#333333',mew=.5)
    for k,row in enumerate(upper):
        ui.plot(row['D'],row['global_E_hz'],'*',color='#b03c39',ms=9)
        ui.annotate(f'SN{k+3}',(row['D'],row['global_E_hz']),xytext=[(-20,-22),(-22,17),(8,-24),(0,16)][k],
            textcoords='offset points',fontsize=9,arrowprops=dict(arrowstyle='-',color='k',lw=.5))
    ui.set_xlim(.365,.401);ui.set_ylim(409,420);ui.set_xticks([.37,.385,.40]);ui.set_yticks([410,415,420])
    ui.set_xlabel(r'$D$',fontsize=10);ui.set_ylabel('Hz / neuron',fontsize=10);ui.tick_params(labelsize=9)
    selected=[];targets=[.228,.256,.275]
    candidates=[x for x in all_states if x['global_E_hz']>150 and '/D_gap_' not in x['file']]
    if a.include_gap:
        fixed_files=[base/'D_gap_lower_guarded/point0740.npz',base/'D_gap_middle_down/point0000.npz',
            base/'D_arclength_upper_onset_range_v4/point0078.npz']
        fixed_selection=[next(row for row in all_states if row['file']==str(f)) for f in fixed_files]
        proof_by_file[str(fixed_files[-1])]='UNSTABLE' if read(base/'contour_modes/snapshot1_D0p228/result.json')['stability']=='UNSTABLE' else 'UNRESOLVED'
        fold=read(base/'D_gap_lower_guarded/verified_outer_fold/result.json')['rows'][0]
        if fold['critical_type']=='SN':
            fixed_selection[0]=dict(**fold,file=str(base/'D_gap_lower_guarded/verified_outer_fold/fold31.npz'))
            ax.plot(fold['D'],fold['global_E_hz'],'*',color='#b03c39',ms=12,zorder=10)
            ax.annotate('SN7',(fold['D'],fold['global_E_hz']),xytext=(38,-22),textcoords='offset points',
                fontsize=11,color='#b03c39',arrowprops=dict(arrowstyle='-',color='#b03c39',lw=.7))
    for k,target in enumerate(targets):
        row=fixed_selection[k] if a.include_gap else min(candidates,key=lambda x:abs(x['D']-target));selected.append(row)
        d=np.load(row['file']);rates=d['r'];cell=s.geo['group_cell'][s.E];n=s.sizes[s.E]
        field=np.bincount(cell,weights=rates[s.E]*n,minlength=400)/np.bincount(cell,weights=n,minlength=400)*1000
        bx=fig.add_subplot(gs[k,1]);im=bx.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],vmin=0,vmax=500,cmap='magma',interpolation='nearest')
        for letter,xy in zip(['A','B'],s.geo['centers_mm']):
            bx.add_patch(Circle(xy,1.5,fill=False,ec='#32c7cb',lw=1.25));bx.text(xy[0],xy[1]+2,letter,color='#168d92',fontsize=10,ha='center')
        bx.set_xticks([0,10,20]);bx.set_yticks([0,10,20]);bx.set_ylabel('y (mm)',fontsize=12)
        if k==2:bx.set_xlabel('x (mm)',fontsize=12)
        bx.text(-.26,1.04,chr(66+k),transform=bx.transAxes,fontsize=20,fontweight='bold')
        bx.set_title(f'{k+1}   D = {row["D"]:.3f}',fontsize=12,pad=5)
        if not (a.include_gap and k==0):ax.plot(row['D'],row['global_E_hz'],'o',mfc='#c2761b',mec='white',ms=8,zorder=8)
        certified=base/'contour_modes'/['snapshot1_D0p228','snapshot2_D0p256','snapshot3_D0p275'][k]/'result.json'
        unstable=(proof_by_file.get(row['file'])=='UNSTABLE') if a.include_gap else (certified.exists() and read(certified)['stability']=='UNSTABLE')
        if unstable:
            ax.plot(row['D'],row['global_E_hz'],'x',ms=5,mew=1,color='k',zorder=9)
        offsets=[(-36,-22),(31,3),(29,14)] if a.include_gap else [(24,-15),(36,1),(8,23)]
        ax.annotate(str(k+1),(row['D'],row['global_E_hz']),xytext=offsets[k],
            textcoords='offset points',color='#9a5400',fontsize=12,
            arrowprops=dict(arrowstyle='-',color='#9a5400',lw=.7))
    fig.subplots_adjust(left=.09,right=.91,bottom=.09,top=.96)
    cax=fig.add_axes([.925,.25,.016,.5]);cb=fig.colorbar(im,cax=cax);cb.set_label('Equilibrium E rate (Hz)',fontsize=12);cb.set_ticks([0,250,500])
    for ext in ['png','pdf','svg']:fig.savefig(figdir/f'{a.stem}.{ext}',dpi=200)
    plt.close(fig)
    write(figdir/f'{a.stem}_metadata.json',dict(type='INTERIM_EQUILIBRIUM_DIAGNOSTIC_NOT_FINAL_BIFURCATION_FIGURE',
        fixed_Z='9.42s spatial field power path with D=1-weighted_mean_Z',M='dynamic',J_EE_core=1,
        snapshots='Actual equilibria of the same fixed rate framework; not native snapshots and not time-dependent events',
        selected_states=selected,stored_equilibrium_points=len(all_states),verified_unstable_points=len(proven),
        branches=branches,rate_slice_connection='Independent rate-selected roots are not connected as a continuous branch',
        periodic_branches='NOT_COMPUTED_MISSING_SAME_MODEL_NONLINEAR_RHS',human_visual_acceptance='PENDING'))
    if not a.include_gap:(figdir/'README.md').write_text('''### fig_zm_equilibrium_progress.png / .pdf / .svg
本图是固定空间率框架的阶段性平衡分支诊断，不是已完成的onset分岔图。横轴D对应明示的空间Z路径，M保留动态反馈；虚线为已有正实部动态根证据的不稳定平衡，空心圆为稳定性尚待确认的平衡，星形为已验证的条件平衡鞍结，不能直接称作全局onset。右侧是同一方程、同一参数处的实际平衡空间率场，不是SNN原图截图或时间轨迹瞬时场；周期轨道尚未加入。**关注点**：先前的高活动支缺口已向D约0.228延拓；不能把局部鞍结或待判稳定性的高率平衡直接认作发作起始。
''')
    print(figdir/f'{a.stem}.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--include-gap',action='store_true')
    p.add_argument('--stem',default='fig_zm_equilibrium_progress');main(p.parse_args())
