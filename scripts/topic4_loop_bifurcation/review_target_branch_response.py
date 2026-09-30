#!/usr/bin/env python3
"""Explain why the first numerical equilibrium turn is not certified exit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from coupled_density_exit import ADAPTED
from analyze_exit_branch_density import native_prefix


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    out=ROOT/'target_branch_response_review';out.mkdir(exist_ok=True)
    old=read(ROOT/'target_exit_equilibrium_branch/branch.json')['rows']
    new=read(ROOT/'target_exit_equilibrium_branch_v2/branch.json')['rows']
    rows=old+new[1:]
    assert read(ROOT/'target_exit_equilibrium_branch_v2/branch.json')['status']=='STOPPED_AFTER_INDEPENDENT_LOCAL_DC_FAILURE'
    geo=dict(np.load(OPS/'geometry.npz'));display=dict(np.load(ADAPTED/'geometry.npz'))
    group=geo['cell_group'];region=geo['group_region'][group];E=np.arange(40000)<32000
    cells=display['group_cell'][group];counts=np.bincount(cells[E],minlength=400)
    def field(rate):return np.bincount(cells[E],weights=rate[E],minlength=400)/np.maximum(counts,1)
    folder=ROOT/'target_exit_equilibrium_branch_v2'
    before=np.load(folder/'point_006.npz')['cell_rate_per_ms']*1000
    after=np.load(folder/'point_008.npz')['cell_rate_per_ms']*1000
    delta=after-before;mag=abs(delta[E]);order=np.argsort(-mag)
    n=int(np.searchsorted(np.cumsum(mag[order]),.9*mag.sum())+1)
    native=native_prefix('exit_z0.21_k9_fields16p7_high')
    native_field=native['fields'][1000:2000].mean(0)
    root=np.load(ROOT/'target_stationary_root/root_candidate.npz')['cell_rate_per_ms']*1000
    root_field=field(root)
    source=read(ROOT/'target_root_response_audit/result.json')
    local=[q for q in source['rows'] if q['kernel']=='density_discrete' and q['channel']=='mean_mV' and q['amplitude_factor']==1]
    result=dict(status='COMPLETE_REVIEW_OF_REJECTED_STATIC_RESPONSE',
        accepted_numerical_roots=len(rows),maximum_K_sampled=max(q['K'] for q in rows),
        last_allE_coreA_coreB_Hz=new[-1]['rates_Hz_allE_A_B_other_I'][:3],
        turn_change_E_cells_for_90percent_absolute_change=n,turn_change_E_fraction=n/32000,
        turn_change_absolute_fraction_by_coreA_coreB_surround=[float(mag[region[E]==j].sum()/mag.sum()) for j in range(3)],
        turn_change_definition='Finite difference of individual target rates between v2 points006 and008. This is NOT a Jacobian eigenmode or certified nullvector.',
        root_native_weighted_field_RMS_Hz=float(np.sqrt(np.average((root_field-native_field)**2,weights=counts))),
        root_native_weighted_field_MAE_Hz=float(np.average(abs(root_field-native_field),weights=counts)),
        static_response_audit_summary=source['summary'],
        decision='Reject use of these roots/turns for formal bifurcation until the local response is repaired. The first numerical turn changes recruitment in the surround; high global and both core rates remain. No certified stable/unstable labels or native fold.',
        native_physics_changed=False,formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    np.savez_compressed(out/'readouts.npz',root_field_Hz=root_field,native_field_Hz=native_field,
        cell_rate_change_Hz=delta,display_rate_change_Hz=field(delta),K=np.array([q['K'] for q in rows]),
        branch_allE_Hz=np.array([q['rates_Hz_allE_A_B_other_I'][0] for q in rows]))
    write(out/'analysis.json',result)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10.5,8.2),layout='constrained')
    ax=axes[0,0];K=np.array([q['K'] for q in rows]);rate=np.array([q['rates_Hz_allE_A_B_other_I'][0] for q in rows])
    ax.plot(K,rate,'o-',color='#83756a',ms=3,lw=1.1,label='Numerical equilibrium')
    ax.scatter([9],[native['rate5'][1000:2000,0].mean()],s=65,c='black',marker='s',label='Native 5–10 s',zorder=4)
    ax.set(xlabel=r'Mean held $K$ ($g_K/g_L$)',ylabel='All E rate (Hz)',title='A  Connected high branch',ylim=(225,251),xlim=(8.99,9.215))
    ax.legend(frameon=False,fontsize=9,loc='upper right')
    ax.text(.05,.06,'Both cores remain above 466 Hz\nStability not established',transform=ax.transAxes,fontsize=9)
    ax=axes[0,1];df=field(delta);limit=max(abs(df).max(),1)
    im=ax.imshow(df.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=-limit,vmax=limit,cmap='RdBu_r',interpolation='nearest')
    for xy in np.load(ROOT/'native_slices/geometry.npz')['centers_mm']:ax.add_patch(Circle(xy,1.5,fill=False,edgecolor='#00c3c5',lw=1))
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title='B  Rate change across the turn',xticks=[0,10,20],yticks=[0,10,20])
    fig.colorbar(im,ax=ax,label=r'$\Delta r_E$ (Hz)',shrink=.9)
    ax=axes[1,0]
    for population,color in [('E','#c97724'),('I','#617787')]:
        a=[q for q in local if q['population']==population]
        ax.scatter([q['mean_rate_Hz'] for q in a],[q['prediction_Hz'] for q in a],s=30,c=color,label=population)
    ax.plot([0,800],[0,800],color='.65',lw=1,ls=':')
    ax.set(xscale='symlog',yscale='symlog',xlim=(-.05,850),ylim=(-.05,850),
        xlabel='Direct cell simulation (Hz)',ylabel='Static approximation (Hz)',title='C  Local mean rate')
    ax.legend(frameon=False,fontsize=9)
    ax=axes[1,1]
    for population,color in [('E','#c97724'),('I','#617787')]:
        a=[q for q in local if q['population']==population and q['estimable'] and q['predicted_gain_Hz']>0]
        ax.errorbar([q['measured_gain_Hz'] for q in a],[q['predicted_gain_Hz'] for q in a],
            xerr=[2*q['SEM'] for q in a],fmt='o',ms=5,color=color,lw=.8,label=population)
    ax.plot([.1,100],[.1,100],color='.65',lw=1,ls=':')
    ax.set(xscale='log',yscale='log',xlim=(.1,100),ylim=(.1,100),xlabel='Direct gain (Hz/mV)',
        ylabel='Static derivative (Hz/mV)',title='D  Local response slope')
    fig.suptitle('Actual exit fields: equilibrium and response audit',weight='bold')
    fig.text(.5,-.015,'Held Z mean = 0.21; actual 16.7 s spatial field family. Numerical turn is unqualified; no native bifurcation is asserted.',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'target_branch_response_audit.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    write(out/'figure_metadata.json',dict(agent_visual='PENDING',human_review='PENDING',producer_sha256=sha(__file__)))
    print(result,flush=True)


if __name__=='__main__':main()
