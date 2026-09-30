"""Measure group-input closure errors; never identify a bifurcation from this audit."""
from common import OUT, np, read, write, model
from scipy.special import ndtr
from scipy.ndimage import uniform_filter1d
from numba import njit
from pathlib import Path
import pickle, csv, hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEST=OUT/'native_input_bridge';RUN=DEST/'runs/native_t8000_inputs_observe'


@njit
def z_replay(target,initial):
    out=np.empty_like(target);z=initial.copy()
    for k in range(len(target)):
        out[k]=z
        z += .1/5000*(target[k]-z)
    return out


def main():
    assert read(DEST/'replay_audit.json')['status']=='PASS'
    assert read(DEST/'reconstruction_qa.json')['status']=='PASS'
    c=read(DEST/'contract.json');s=model();names=c['moment_names'];sel=c['selected_groups']
    records=[np.load(p) for p in sorted((RUN/'inputs').glob('*.npz'))]
    t=np.concatenate([q['time_ms'] for q in records]);mom=np.concatenate([q['moments'] for q in records]);spikes=np.concatenate([q['spikes'] for q in records])
    rec=np.load(DEST/'reconstructed_moments.npz');r=rec['moments'];assert np.array_equal(t,rec['time_ms'])
    m={n:mom[:,j] for j,n in enumerate(names)}
    variance={key:m[key+'2']-m[key]**2 for key in ['ampa','gaba','zgaba','mcurrent','net','z','voltage']}
    minimum={k:float(v.min()) for k,v in variance.items()};assert min(minimum.values())>-1e-6,minimum
    variance={k:np.maximum(v,0) for k,v in variance.items()}
    covariance={k:m[k]-m[a]*m[b] for k,a,b in [('ampa_gaba','ampa','gaba'),('ampa_zgaba','ampa','zgaba'),
        ('ampa_m','ampa','mcurrent'),('zgaba_m','zgaba','mcurrent')]}
    netv=variance['ampa']+variance['zgaba']+variance['mcurrent']-2*covariance['ampa_zgaba']-2*covariance['ampa_m']+2*covariance['zgaba_m']
    algebra=float(abs(netv-variance['net']).max());assert algebra<1e-6
    assert np.max(abs(m['net']-(m['ampa']-m['zgaba']-m['mcurrent'])))<1e-8
    mean_prediction=r[:,0]-m['z']*r[:,1]-m['mcurrent']
    disc_prediction=r[:,2]-m['z']*r[:,3]-m['mcurrent']
    var_full=r[:,4]+m['z']**2*r[:,5];var_private=r[:,6]+m['z']**2*r[:,7]
    independent_observed=variance['ampa']+variance['zgaba']+variance['mcurrent']
    with (RUN/'checkpoint.pkl').open('rb') as f:final=pickle.load(f)['engine']['slow']['z']
    lastz=np.bincount(s.geo['cell_group'],weights=final,minlength=s.P)/s.sizes
    target=(np.diff(np.concatenate([m['z'],lastz[None]],axis=0),axis=0)/(.1/5000)+m['z'])
    assert target[:,s.E].min()>-1e-8 and target[:,s.E].max()<1+1e-8
    # Number below the threshold is integer in every actual group and timestep.
    ntarget=target[:,s.E]*s.sizes[s.E];integer_error=float(abs(ntarget-np.round(ntarget)).max());assert integer_error<1e-6
    thr=95.19851312666987
    gaussian_actual=ndtr((thr-m['gaba'])/np.sqrt(np.maximum(variance['gaba'],1e-20)))
    gaussian_full=ndtr((thr-r[:,1])/np.sqrt(np.maximum(r[:,5],1e-20)))
    gaussian_private=ndtr((thr-r[:,1])/np.sqrt(np.maximum(r[:,7],1e-20)))
    start=np.flatnonzero(t>=9000)[0]
    zp={name:z_replay(tg[start:],m['z'][start]) for name,tg in
        [('actual_target',target),('actual_moment_gaussian',gaussian_actual),('reconstructed_full',gaussian_full),('reconstructed_private',gaussian_private)]}
    zerr=float(abs(zp['actual_target']-m['z'][start:]).max());assert zerr<1e-10
    regions=[s.E&(s.geo['group_region']==i) for i in range(3)]
    regions += [~s.E];region_names=['Core A E','Core B E','Surround E','I']
    def avg(x,mask):return np.average(x[:,mask],weights=s.sizes[mask],axis=1)
    window_ranges=[(9000,9420),(9420,9868.5),(9868.5,10370)]
    rows=[]
    for rn,mask in zip(region_names,regions):
        for lo,hi in window_ranges:
            keep=(t>=lo)&(t<hi)
            def pooled(x):return float(avg(x[keep],mask).mean())
            row=dict(region=rn,start_ms=lo,end_ms=hi,neurons=int(s.sizes[mask].sum()),samples=int(keep.sum()),
                rate_hz=float(spikes[keep][:,mask].sum()/s.sizes[mask].sum()/keep.sum()*10000),
                net_mean_native_mv=pooled(m['net']),net_mean_bias_mv=pooled(mean_prediction-m['net']),
                net_mean_rmse_mv=np.sqrt(pooled((mean_prediction-m['net'])**2)),
                net_mean_native_step_rmse_mv=np.sqrt(pooled((disc_prediction-m['net'])**2)),
                ampa_mean_rmse_mv=np.sqrt(pooled((r[:,0]-m['ampa'])**2)),
                gaba_mean_rmse_mv=np.sqrt(pooled((r[:,1]-m['gaba'])**2)),
                ZG_factorization_bias_mv=pooled(m['zgaba']-m['z']*m['gaba']),
                variance_net_native_mv2=pooled(variance['net']),
                variance_net_full_mv2=pooled(var_full),variance_net_private_mv2=pooled(var_private),
                variance_net_independent_actual_mv2=pooled(independent_observed),
                E_ZG_covariance_mv2=pooled(covariance['ampa_zgaba']),
                meanZ=pooled(m['z']),meanM_mv=pooled(m['mcurrent']))
            row['private_over_native_netvariance']=row['variance_net_private_mv2']/row['variance_net_native_mv2']
            row['full_over_native_netvariance']=row['variance_net_full_mv2']/row['variance_net_native_mv2']
            row['independent_actual_over_native_netvariance']=row['variance_net_independent_actual_mv2']/row['variance_net_native_mv2']
            if rn!='I':
                row.update(target_native=pooled(target),target_gaussian_actual_moments=pooled(gaussian_actual),
                    target_gaussian_reconstructed_private=pooled(gaussian_private),
                    target_gaussian_actual_mae=pooled(abs(gaussian_actual-target)),
                    target_gaussian_reconstructed_mae=pooled(abs(gaussian_private-target)))
            rows.append(row)
    endpoints=[]
    for rn,mask in zip(region_names[:3],regions[:3]):
        endpoints.append(dict(region=rn,observed_Z=float(avg(m['z'][-1:],mask)[0]),
            **{name:float(avg(z[-1:],mask)[0]) for name,z in zp.items()}))
    # All fixed selected groups, including low-threshold initiators, retained.
    np.savez_compressed(DEST/'selected_input_history.npz',time_ms=t,groups=sel,moments=mom[:,:,sel],moment_names=names,
        reconstructed=r[:,:,sel],reconstructed_names=rec['names'],spikes=spikes[:,sel],group_size=s.sizes[sel],
        theta=s.theta[sel],population=(~s.E[sel]).astype(int),target=target[:,sel],
        actual_current_covariance=covariance['ampa_zgaba'][:,sel],net_current_variance=variance['net'][:,sel])
    keylist=list(dict.fromkeys(k for row in rows for k in row))
    with (DEST/'input_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keylist);w.writeheader();w.writerows(rows)
    write(DEST/'input_comparison.json',dict(status='INPUT_BRIDGE_DIAGNOSTIC_COMPLETE',rows=rows,Z_teacher_forced_endpoints=endpoints,
        QA=dict(variance_minima_before_roundoff_cleanup=minimum,net_variance_algebra_max_error=algebra,
                reconstructed_threshold_integer_error=integer_error,actual_target_Z_replay_error=zerr),
        definitions='Native variances are within-group cross-cell spread, including quenched heterogeneity and correlations. Reconstructed variances assume independent stationary Poisson input/private subtraction; these are not identical estimands.',
        scope='Single native history with actual future group spikes supplied only for diagnostic localization. No model training, autonomous acceptance, universal Zthreshold or onset bifurcation label.',
        local_response_test='NOT_RUN: selected exact input histories exported for the next fixed-response diagnosis.'))
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(4,3,figsize=(13,10),sharex=True)
    fig.subplots_adjust(left=.085,right=.98,bottom=.075,top=.95,hspace=.2,wspace=.25)
    for col,(rn,mask) in enumerate(zip(region_names[:3],regions[:3])):
        def plot(ax,y,color,label,ls='-'):
            # Full10mswindows only: no endpoint padding of spike counts.
            v=np.convolve(avg(y,mask),np.ones(100)/100,mode='valid')
            tx=np.convolve(t,np.ones(100)/100,mode='valid')/1000
            ax.plot(tx,v,color=color,label=label,lw=1,ls=ls)
        axes[0,col].text(.5,1.05,rn,ha='center',transform=axes[0,col].transAxes)
        plot(axes[0,col],spikes/s.sizes/.1*1000,'#202020','Native')
        plot(axes[1,col],m['net'],'#202020','Native input')
        plot(axes[1,col],mean_prediction,'#c56829','Projected input')
        plot(axes[2,col],variance['net'],'#202020','Native spread')
        plot(axes[2,col],var_full,'#bb7d35','Full Poisson')
        plot(axes[2,col],var_private,'#198777','Private Poisson',ls='--')
        axes[3,col].plot(t[start:]/1000,avg(m['z'][start:],mask),c='#202020',label='Native')
        for key,color,ls,label in [('actual_moment_gaussian','#9363ad','-','Measured moments + Gaussian'),
            ('reconstructed_private','#198777','--','Projected moments + Gaussian')]:
            axes[3,col].plot(t[start:]/1000,avg(zp[key],mask),c=color,ls=ls,label=label)
        axes[3,col].set_xlabel('Time (s)')
        for ax in axes[:,col]:ax.set_xlim(9,10.37);ax.set_xticks([9,9.42,9.87,10.37]);ax.axvline(9.8685,color='#7d7d7d',ls=':',lw=.8)
    for row,label in enumerate(['Rate (Hz / neuron)','Net input (mV)','Net input variance (mV²)','Resource Z']):
        axes[row,0].set_ylabel(label);axes[row,0].text(-.24,1.03,'ABCD'[row],transform=axes[row,0].transAxes,fontweight='bold',fontsize=16)
    axes[1,2].legend(frameon=False,fontsize=9);axes[2,2].legend(frameon=False,fontsize=9);axes[3,2].legend(frameon=False,fontsize=8,loc='lower left')
    folder=OUT/'figures';stem='fig_native_input_bridge'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{stem}.{ext}',dpi=190)
    plt.close(fig)
    write(folder/f'{stem}.json',dict(source=str(DEST/'input_comparison.json'),time_range_ms=[9000,10370],
        line_smoothing_ms=10,vertical_marker='Native global high-activity entry9868.5ms; not a bifurcation',
        bottom_panel='Z only integrated from observed9s state under supplied future moments. Observed native inputs or projected native groupcounts; not autonomous model runs.',
        human_visual_acceptance='PENDING',agent_visual_check='PENDING'))
    path=folder/'README.md';text=path.read_text();heading=f'### {stem}.png'
    if heading not in text:path.write_text(text+'\n'+heading+'\n三列为核A、核B和核外E；依次显示原生率、实际与群体投影输入均值、细胞间输入方差，以及只重演Z规则得到的慢变量。所有输入来自同一条逐位复现的原生8–10.37秒轨迹；前1秒准备，正式比较9–10.37秒，前三排仅用10ms显示平滑。底排从共同9秒Z出发，分别用实测矩或投影矩的高斯近似更新；这不是自主网络轨迹。**关注点**：空间群体投影和输入分布近似在哪里丢失原生Z耗减及传播所需信息；图中虚线时间标记不是分岔认证。\n')
    print(json.dumps(dict(rows=rows,endpoints=endpoints),indent=2),flush=True)


if __name__=='__main__':main()
