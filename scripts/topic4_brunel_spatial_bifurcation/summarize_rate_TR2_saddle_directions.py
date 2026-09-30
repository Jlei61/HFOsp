"""Check and visualize a finite saddle-direction diagnostic, without promotion."""
from check_rate_TR2_saddle_directions import *
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--result',type=Path,default=OUT/'result.json')
    parser.add_argument('--no-figure',action='store_true',help='Run numerical checks only, without touching shared figures or README')
    args=parser.parse_args();destination=args.result.parent
    q = read(args.result)
    exact_parameter=q['status']=='FINITE_EXACT_PARAMETER_DIRECTION_DIAGNOSTIC'
    s = RateField()
    z = np.load(q['torus']); c, k, ell = coefficients(z['r'])
    local = np.load(q.get('torus_local_coefficients',OUT/'torus_a0038_local_coefficients.npy'), mmap_mode='r').reshape(9,len(k),len(ell),s.P)
    fine = np.load(q['eigenvector_source'])
    coarse = np.load(q.get('coarse_eigenvector_source',PER/'poincare_floquet/TR2_endpoint_middle_N128_dt0.1.npz'))
    selected = q.get('selected_mode_indices',[0,1])
    from scipy.optimize import linear_sum_assignment
    ii,jj=linear_sum_assignment(abs(coarse['multipliers'][:,None]-fine['multipliers'][None,:]))
    coarse_selected=[int(ii[np.flatnonzero(jj==i)[0]]) for i in selected]
    fv = np.r_[fine['local_vectors'][:,selected],fine['history_vectors'][:,selected]].real
    D = fine['history_vectors'].shape[0]//s.P
    ages = np.arange(1,D+1)*float(fine['dt'])
    hist = coarse['history_vectors'][:,coarse_selected].real.reshape(-1,s.P,2)
    loc = coarse['local_vectors'][:,coarse_selected].real.reshape(9,s.P,2)
    h0 = s.alpha[:,None]*loc[0]+(1-s.alpha[:,None])*loc[1]
    ch = interp1d(np.arange(len(hist)+1)*float(coarse['dt']),np.concatenate([h0[None],hist]),axis=0)(ages)
    cv = np.r_[loc.reshape(9*s.P,2),ch.reshape(-1,2)]
    old = np.load(ROOT/read(Path(q['eigenvector_source']).with_suffix('.json'))['orbit'])
    oc, ko, _ = coefficients(old['r'][:,None])
    ol = np.load(q.get('mode_source_local_coefficients',OUT/'mode_source_local_coefficients.npy'), mmap_mode='r')
    phase = periodic_state(ol,oc[:,0],ko,2*np.pi/float(old['T']),ages,True)
    phase /= np.linalg.norm(phase)
    w = np.sqrt(s.geo['group_size']/s.geo['group_size'].sum())
    scale = np.r_[(np.array([1000,1000,1,1,1,1,.1,.1,1])[:,None]*w).ravel(),
                  np.broadcast_to(1000*w/np.sqrt(D),(D,s.P)).ravel()]
    def normalized(v):
        v = (v-phase[:,None]*(phase@v)[None])*scale[:,None]
        return v/np.linalg.norm(v,axis=0)
    f,cross = normalized(fv),normalized(cv)
    angles = np.degrees(np.arccos(np.clip(abs(np.sum(f*cross,axis=0)),0,1)))
    controls=[]
    for i in range(2):
        test=diagnostics(fv[:,i],phase,fv,scale)
        assert test['relative_two_mode_residual']<1e-12
        assert test['angles_to_unstable_stable_degrees'][i]<1e-5
        controls.append(test)
    source = read(PER/'TR2_same_parameter_saddle_approach.json')['rows'][-1]
    checks=[]
    for index in [0,158,248]:
        theta = 2*np.pi*source['fast_phase_shifts_cycles'][index]
        psi = 2*np.pi*index/256
        aa = np.array([ages[0],max(s.delays),ages[-1]])
        y,order,bound=torus_state(local,c,k,ell,2*np.pi/float(z['T']),float(z['nu']),theta,psi,aa)
        approximate=y[9*s.P:].reshape(3,s.P)
        exact=np.array([np.einsum('klp,k,l->p',c,
            np.exp(1j*k*(theta-2*np.pi/float(z['T'])*a)),
            np.exp(1j*ell*(psi-float(z['nu'])*a)),optimize=True).real for a in aa])
        error=float(abs(approximate-exact).max())
        assert error<1e-14
        checks.append(dict(slow_index=index,maximum_history_evaluation_difference_per_ms=error,
                           Taylor_remainder_bound_per_ms=bound,order=order))
    rows=sorted(q['rows'],key=lambda r:(r['slow_index']-q['closest_fast_averaged_rate_index']+128)%256)
    t=np.array([((r['slow_index']-q['closest_fast_averaged_rate_index']+128)%256-128)/256*q['slow_period_s'] for r in rows])
    distance=np.array([r['rate_phase_averaged_distance_Hz'] for r in rows])
    coeff=np.array([r['matched_target']['weighted']['coefficients'] for r in rows])
    dirs=np.array([r['matched_target']['weighted']['angles_to_unstable_stable_degrees'] for r in rows])
    residual=np.array([r['matched_target']['weighted']['relative_two_mode_residual'] for r in rows])
    fits=[]
    for cutoff in [5,10,20]:
        for i,name in [(1,'approach_stable'),(0,'departure_unstable')]:
            keep=(t<0 if i==1 else t>0)&(distance>cutoff*distance.min())&(distance<1e-3)&(dirs[:,i]<5)
            if keep.sum()<3:continue
            x=t[keep]; y=np.log(abs(coeff[keep,i]));slope,intercept=np.polyfit(x,y,1)
            fits.append(dict(side=name,minimum_distance_over_closest=cutoff,n=int(keep.sum()),
                slow_indices=[rows[j]['slow_index'] for j in np.flatnonzero(keep)],
                fitted_exponent_per_s=float(slope),log_fit_RMS=float(np.std(y-slope*x-intercept)),
                maximum_mode_angle_degrees=float(max(dirs[keep,i])),
                maximum_two_mode_relative_residual=float(max(residual[keep]))))
    source_spec=read(Path(q['eigenvector_source']).with_suffix('.json'))
    exponents=np.log(np.array(q['multipliers'])[:,0])/(source_spec['T_ms']/1000)
    out=dict(status='FINITE_DIRECTION_AND_EVALUATION_CHECK_COMPLETE',source=str(args.result),
        exact_parameter_eigenvectors=exact_parameter,parameter_offset=q['parameter_offset'],
        fine_mode_indices=selected,coarse_mode_indices=coarse_selected,
        paired_history_interpolated_mode_angles_degrees=angles.tolist(),
        pure_mode_controls=controls,history_direct_evaluation_checks=checks,exploratory_local_slope_checks=fits,
        reference_exponents_per_s=exponents.tolist(),
        selection='Exploratory sensitivity: within 5 degrees of the relevant right mode, rate distance below 1e-3 Hz and above 5/10/20 times its sampled minimum; at least three samples.',
        scope=q['scope'])
    write(destination/'checks.json',out)
    print('DIRECTION CHECKS',angles.tolist(),exponents.tolist(),fits,flush=True)
    if args.no_figure:return
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    axs[0].semilogy(t,distance,color='black',lw=1.4)
    axs[0].set(xlabel='Slow time from closest approach (s)',ylabel='Fast-phase RMS rate distance (Hz)',title='A  Same-J saddle-cycle approach')
    for i,color,label in [(1,'#2166ac','Stable direction'),(0,'#b2182b','Unstable direction')]:
        axs[1].plot(t,dirs[:,i],color=color,lw=1.4,label=label)
        axs[2].semilogy(t,abs(coeff[:,i]),color=color,lw=1.4,label=label)
    axs[1].set(xlabel='Slow time from closest approach (s)',ylabel='Angle to right Floquet direction (degrees)',ylim=(0,90),title='B  Approach and departure directions')
    axs[1].legend(frameon=False,fontsize=9)
    axs[2].set(xlabel='Slow time from closest approach (s)',ylabel='Absolute two-mode fit coefficient',title='C  Full-state section projection')
    for ax in axs:
        ax.axvline(0,color='#777777',ls=':',lw=.7)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('TR2 connection candidate: same-J targets, '+('same-J' if exact_parameter else 'nearby-J')+' Floquet directions',fontsize=12)
    folder=OUT.parent/'figures';name='TR2_full_state_approach_departure'+('_exactJ' if exact_parameter else '')
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    path=folder/'README.md';body=path.read_text()
    body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    entry='\n\n### '+name+'.png\n以同参数鞍周期为参照，展示 TR2 慢调制解的接近距离，以及全部935群体、九个局部状态和延迟历史在两个主导 Floquet 方向上的投影。两个方向来自极近参数的鞍周期，已单独记录参数偏差及时间步长加密后的方向变化。**关注点**：这是有限样本的方向诊断；投影系数不是伴随模态坐标，尚未求解连接流形，也未证明该调制解稳定。\n'
    if exact_parameter:
        entry=entry.replace('两个方向来自极近参数的鞍周期，已单独记录参数偏差及时间步长加密后的方向变化。',
            '两个方向来自与该慢调制解完全相同参数的鞍周期，使用配对时间步长验证的谱，并检查加密后的方向变化。')
    body+=entry
    path.write_text(body)


if __name__=='__main__':main()
