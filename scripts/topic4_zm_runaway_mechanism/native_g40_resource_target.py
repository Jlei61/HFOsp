"""Read-only native current/threshold audit at the current g40 resolution."""
from common import ROOT, OUT, model, np, read, write, log
from scipy.special import ndtr

DEST=OUT/'transient_Z_feedback_shadow_20260923'


def main():
    assert 'additional_readonly_comparison' in read(DEST/'contract.json')
    s=model(40);ids=s.geo['cell_group'][:32000];N=np.bincount(ids,minlength=s.P);den=np.maximum(N,1)
    assert np.array_equal(N[s.E],s.sizes[s.E])
    region=s.geo['group_region'];weights=[]
    for mask in [s.E]+[s.E&(region==i) for i in range(3)]:
        w=mask*s.sizes;weights.append(w/w.sum())
    W=np.array(weights);time=[];means=[];variances=[];targets=[];Z=[]
    src=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401'
    threshold=95.19851312666987
    def append(t,current,z):
        current=current.astype(float);mean=np.bincount(ids,weights=current,minlength=s.P)/den
        var=np.maximum(0,np.bincount(ids,weights=current**2,minlength=s.P)/den-mean**2)
        actual=np.bincount(ids,weights=(current<threshold),minlength=s.P)/den
        normal=ndtr((threshold-mean)/np.sqrt(np.maximum(var,1e-20)))
        actual[~s.E]=1.;normal[~s.E]=1.
        zz=np.ones(s.P);zz[s.E]=(np.bincount(ids,weights=z.astype(float),minlength=s.P)/den)[s.E]
        assert abs(W[0]@actual-(current<threshold).mean())<1e-14
        time.append(t);means.append(mean);variances.append(var);targets.append([actual,normal]);Z.append(zz)
    for p in sorted((src/'fields').glob('*.npz')):
        with np.load(p) as data:
            ts=data['zm_step']*.1;ii=data['ii'];zz=data['z']
            for t,c,z in zip(ts,ii,zz):append(t,c,z)
    with np.load(src/'checkpoints/t12500ms.npz') as data:
        append(12500.,data['I_I'][:32000],data['slow__z'][:32000])
    t=np.array(time);T=np.array(targets);Z=np.array(Z)
    assert set(np.diff(t))=={5.,10.}
    regional=np.einsum('tcp,rp->tcr',T,W)
    estimates={}
    for method in ['left','linear']:
        delta=np.zeros((len(t),4));predicted=np.ones((len(t),2,4))
        for k,dt in enumerate(np.diff(t)):
            e=np.exp(-dt/5000);w1=0. if method=='left' else 1+np.expm1(-dt/5000)/(dt/5000)
            w0=-np.expm1(-dt/5000)-w1
            predicted[k+1]=e*predicted[k]+w0*regional[k]+w1*regional[k+1]
            delta[k+1]=predicted[k+1,1]-predicted[k+1,0]
        estimates[method]=(predicted,delta)
    # Regional signed means can hide opposite local errors. Retain the full
    # field substitution error and report the E-count weighted absolute norm.
    group_delta=np.zeros((len(t),s.P));difference=T[:,1]-T[:,0]
    for k,dt in enumerate(np.diff(t)):
        e=np.exp(-dt/5000);w1=1+np.expm1(-dt/5000)/(dt/5000);w0=-np.expm1(-dt/5000)-w1
        group_delta[k+1]=e*group_delta[k]+w0*difference[k]+w1*difference[k+1]
    assert np.max(abs(group_delta@W.T-estimates['linear'][1]))<1e-13
    np.savez_compressed(DEST/'native_current_targets.npz',time_ms=t,GABA_mean=np.array(means),
        GABA_variance=np.array(variances),target_empirical=T[:,0],target_Gaussian=T[:,1],Z=Z,
        regional_Z=Z@W.T,regional_target=regional,region_weights=W,
        sampled_Z_left=estimates['left'][0],sampled_Z_linear=estimates['linear'][0],Gaussian_shape_Z_error_field=group_delta)
    result=dict(status='COMPLETE',grid=40,groups=s.P,samples=len(t),regions=['global_E','coreA','coreB','surround'],
        maximum_Gaussian_minus_empirical_Z_error_by_region=np.max(abs(estimates['linear'][1]),axis=0).tolist(),
        maximum_quadrature_difference_by_region=np.max(abs(estimates['left'][1]-estimates['linear'][1]),axis=0).tolist(),
        sampled_empirical_Z_reconstruction_error_by_region=np.max(abs(estimates['linear'][0][:,0]-Z@W.T),axis=0).tolist(),
        maximum_E_count_weighted_absolute_spatial_Z_error=float(np.max(abs(group_delta)@W[0])),
        maximum_individual_E_group_Z_error=float(np.max(abs(group_delta[:,s.E]))),
        scope='Both target methods use the same observed native currents. Shape substitution error is estimated on5/10ms samples; this is not the rate model current-statistics validation or exact unsampled native trajectory.',model_promoted=False)
    write(DEST/'native_target_audit.json',result);log('NATIVE G40 TARGET',result)


if __name__=='__main__':main()
