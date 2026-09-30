"""Independent native-current audit of the Gaussian resource target.

This supplies native currents, never rate-model-predicted currents. The
measured-variance version isolates Gaussian shape; the stationary surrogate
does NOT validate the rate model's time-dependent variance state.
"""
from common import *
from scipy.special import ndtr
from scipy import sparse


def main():
    c=read(OUT/'native_Z_target_reaudit_contract.json');s=model();ne=32000
    idx=s.members;assert len(idx)==ne
    counts=np.bincount(idx,minlength=s.P);den=np.maximum(counts,1)
    assert np.array_equal(counts[s.E],s.sizes[s.E])
    cfg=read(OUT/'native_same_history_feedback/runs/native_t9000_Zdynamic/applied_configuration.json')
    threshold=cfg['threshold'];tau=1000*cfg['tau_Z_s']
    assert threshold==c['threshold_mV'] and tau==c['tau_Z_ms']
    a=sparse.load_npz(s.folder/'mean_gaba.npz');q=sparse.load_npz(s.folder/'variance_gaba.npz')
    jeff=np.asarray(q.sum(1)).ravel()/np.maximum(np.asarray(a.sum(1)).ravel(),1e-300)
    def project(x):return np.bincount(idx,weights=x,minlength=s.P)/den
    source=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/fields'
    rows=[];sources=[];group_errors=[]
    for f in sorted(source.glob('*.npz')):
        z=np.load(f);times=z['zm_step']*.1
        if times[-1]<c['start_ms']:continue
        assert np.array_equal(s.geo['group_cell'][idx],z['cell_e']) if 'cell_e' in z else True
        ii=z['ii'];zz=z['z']
        for j,t in enumerate(times):
            if t<c['start_ms']:continue
            current=ii[j].astype(float);actual=project(current>=threshold)
            mean=project(current);var=project(current*current)-mean*mean
            # Independently use survival-CDF symmetry rather than 1-CDF.
            measured=ndtr((mean-threshold)/np.sqrt(np.maximum(var,1e-20)))
            predicted_variance=s.tm*s.area[1]*jeff*mean/(2*s.tau[1])
            stationary=ndtr((mean-threshold)/np.sqrt(np.maximum(predicted_variance,1e-20)))
            weighted=np.array([x[s.E]@s.mean_weights for x in [actual,stationary,measured]])
            assert abs(weighted[0]-(current>=threshold).mean())<1e-14
            rows.append([t,*weighted,float(zz[j].mean(dtype=float))])
            group_errors.append([abs((stationary-actual)[s.E])@s.mean_weights,
                                 abs((measured-actual)[s.E])@s.mean_weights])
        sources.append(str(f))
    data=np.asarray(rows);ge=np.asarray(group_errors);t=data[:,0];h=np.diff(t)
    assert np.all(h>0) and np.max(h)<=10.0000001
    old=np.load(BASE/'diagnostics/native_z_closure_rows.npz')['global_rows']
    assert np.array_equal(data[:,0],old[:,0])
    reproduction=float(np.max(abs(data[:,1:4]-old[:,1:4])));assert reproduction<1e-11
    # Isolate only the target substitution, using the same sampled currents.
    trajectories={};bounds={}
    for method in ['left_hold','linear_between_samples']:
        delta=np.zeros((len(t),2));absolute_bound=np.zeros_like(delta)
        for j,dt in enumerate(h):
            k=np.exp(-dt/tau);w1=0. if method=='left_hold' else 1.+np.expm1(-dt/tau)/(dt/tau)
            w0=-np.expm1(-dt/tau)-w1
            d0=data[j,2:4]-data[j,1];d1=data[j+1,2:4]-data[j+1,1]
            delta[j+1]=k*delta[j]+w0*d0+w1*d1
            absolute_bound[j+1]=k*absolute_bound[j]+w0*abs(d0)+w1*abs(d1)
        trajectories[method]=delta;bounds[method]=absolute_bound
    windows=[]
    for lo,hi in c['windows_ms']:
        weight=np.maximum(0,np.minimum(t[1:],hi)-np.maximum(t[:-1],lo))
        assert weight.sum()>0
        values=np.average(data[:-1,1:4],axis=0,weights=weight)
        err=np.average(ge[:-1],axis=0,weights=weight)
        windows.append(dict(window_ms=[lo,hi],covered_ms=float(weight.sum()),
                            target_above_native_stationary_measured=values.tolist(),
                            E_weighted_absolute_group_target_error=err.tolist()))
    dest=OUT/'native_Z_target_reaudit';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'target_series.npz',time_ms=t,target_above=data[:,1:4],native_Z=data[:,4],
         group_absolute_error=ge,deltaD_left=trajectories['left_hold'],deltaD_linear=trajectories['linear_between_samples'])
    result=dict(status='NATIVE_CURRENT_TARGET_AUDIT_COMPLETE',source_files=sources,samples=len(t),
        sample_range_ms=[float(t[0]),float(t[-1])],old_global_target_reproduction_max_error=reproduction,
        windows=windows,deltaD_at_last_sample=trajectories['linear_between_samples'][-1].tolist(),
        max_absolute_deltaD=np.max(abs(trajectories['linear_between_samples']),axis=0).tolist(),
        conservative_sampled_absolute_error_bound=bounds['linear_between_samples'][-1].tolist(),
        quadrature_method_max_difference=float(np.max(abs(trajectories['left_hold']-trajectories['linear_between_samples']))),
        interpretation_order=['stationary_variance_surrogate','native_measured_variance'],
        scope=c['scope'])
    write(dest/'result.json',result);log('NATIVE TARGET AUDIT',result)


if __name__=='__main__':main()
