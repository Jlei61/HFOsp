#!/usr/bin/env python3
"""Show directly checked local branch candidates without stability labels."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,read,write,sha
from direct_response_system import DirectDC

OUT=ROOT/'direct_branch_step_review'


def main(wait):
    pair=ROOT/'direct_exit_K_pair/result.json'
    while not pair.exists():
        if not wait:return
        time.sleep(10)
    OUT.mkdir(exist_ok=True);e=DirectDC();paths=[];ks=[];rates=[];gs=[];precision=[]
    with np.load(ROOT/'direct_newton_validation/response.npz') as z:rates.append(z['cell_rate_Hz'])
    with np.load(ROOT/'direct_newton_validation/inputs.npz') as z:gs.append(float(z['Graw']))
    ks.append(9.);paths.append(str(ROOT/'direct_newton_validation'))
    first=read(ROOT/'direct_exit_small_K_step/result.json');assert first['candidate_near_equilibrium']
    folders=[ROOT/'direct_exit_small_K_step'/f"evaluation_{first['rows'][-1]['iteration']}"]
    result=read(pair)
    for row in result['rows']:
        if row['candidate_near_equilibrium']:
            folders.append(ROOT/'direct_exit_K_pair'/f"point_{row['point']:02d}"/f"evaluation_{row['evaluations'][-1]['iteration']}")
    for folder in folders:
        with np.load(folder/'response.npz') as z:rates.append(z['cell_rate_Hz'])
        with np.load(folder/'inputs.npz') as z:ks.append(float(z['K_mean']));gs.append(float(z['Graw']))
        paths.append(str(folder));precision.append(read(folder/'analysis.json'))
    rates=np.array(rates);ks=np.array(ks);gs=np.array(gs);region=e.geo['group_region'][e.group]
    masks=[e.E]+[e.E&(region==j) for j in range(3)]
    regional=np.array([[r[mask].mean() for mask in masks] for r in rates]);delta=rates[-1]-rates[0]
    mass=abs(delta[e.E]);fraction=[float(abs(delta[e.E&(region==j)]).sum()/mass.sum()) for j in range(3)]
    df=e.field(delta);threshold=95.19851312666987/(18+17.662847938268442)
    np.savez_compressed(OUT/'readouts.npz',K=ks,Graw=gs,regional_Hz=regional,cell_delta_Hz=delta,field_delta_Hz=df)
    write(OUT/'analysis.json',dict(status='DIRECT_LOCAL_BRANCH_CANDIDATES',K=ks.tolist(),Graw=gs.tolist(),
        regional_Hz_allE_coreA_coreB_surround=regional.tolist(),sources=paths,
        absolute_E_cell_rate_change_fraction_coreA_coreB_surround=fraction,
        Graw_recovery_block_threshold=threshold,precision=precision,
        scope='Small related interval only. Every plotted new point independently checked by fresh direct cell counts and prescribed residual precision. No exact root, dynamic stability, fold, native termination or autonomous recovery is inferred.',
        full_bifurcation_completed=False,producer_sha256=sha(__file__)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    colors=['#8a63b4','#d63378','#008eb3','#647582'];labels=['All E','Core A','Core B','Surround E']
    ax=axes[0]
    for j,(color,label) in enumerate(zip(colors,labels)):
        ax.plot(ks,regional[:,j]-regional[0,j],'o-',ms=4,lw=1.3,color=color,label=label)
    ax.set(xlabel=r'Mean held $K$ ($g_K/g_L$)',ylabel=r'Rate change from $K=9$ (Hz)',title='A  Local conditional continuation')
    ax.ticklabel_format(axis='x',style='plain',useOffset=False);ax.locator_params(axis='x',nbins=4)
    ax.legend(frameon=False,fontsize=9,loc='lower left')
    ax=axes[1];limit=max(float(abs(df).max()),.1)
    im=ax.imshow(df.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
    for label,xy in zip(['A','B'],np.load(ROOT/'native_slices/geometry.npz')['centers_mm']):
        ax.add_patch(Circle(xy,1.5,fill=False,edgecolor='#00bac1',lw=1.2));ax.text(xy[0],xy[1]+1.8,label,ha='center',color='#009ba2',fontsize=10)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title='B  Spatial change over this interval',xticks=[0,10,20],yticks=[0,10,20])
    fig.colorbar(im,ax=ax,label=r'$\Delta r_E$ (Hz)',shrink=.88)
    ax=axes[2];ax.axhspan(0,threshold,color='#edf4ec');ax.axhline(threshold,color='#71836d',ls=':',lw=1)
    ax.plot(ks,gs,'o-',color='#b27228',ms=4,lw=1.4)
    ax.text(.06,.32,'Necessary bound for\npossible Z recovery',transform=ax.transAxes,color='#52614e',fontsize=10)
    ax.set(xlabel=r'Mean held $K$ ($g_K/g_L$)',ylabel=r'Global conductance $G_{\rm raw}$',title='C  Recovery remains blocked',ylim=(0,5.2))
    ax.ticklabel_format(axis='x',style='plain',useOffset=False);ax.locator_params(axis='x',nbins=4)
    fig.suptitle('Direct cell responses in the actual exit-field family',fontsize=14,weight='bold')
    fig.text(.5,-.045,'Held mean Z = 0.21. Both cores remain in sustained high activity. Dynamic stability and termination boundary are not yet established.',ha='center',fontsize=10)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'direct_branch_first_steps.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    write(OUT/'figure_metadata.json',dict(agent_visual='PENDING',human_review='PENDING',formal_Fig5_replaced=False,producer_sha256=sha(__file__)))
    print('DIRECT BRANCH REVIEW',ks.tolist(),fraction,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
