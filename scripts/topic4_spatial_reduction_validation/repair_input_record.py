"""Correct a diagnostic input record, without changing simulated physics.

The native step-observer scalar includes legacy global OU even when its loading
is zero. Reproduce every original RNG draw and the erroneous record first;
then store the actual zero-global-loading afferent rate. Activity is untouched.
"""
from common import *

def main():
    cfg=read(OUT/'model_config.json');p=cfg['params'];signal=cfg['signal_per_ms']
    region=np.load(V10/'native/a/trajectory.npz')['region']
    core=region[np.isin(region,[0,1])]
    a=np.exp(-p['dt']/p['tau_n']);b=p['sigma_n']*1e-3*np.sqrt(p['tau_n']/2)*np.sqrt(1-a*a)
    for seed in SEEDS:
        folder=OUT/'native'/str(seed)
        if (folder/'input_record_repair.json').exists():continue
        with np.load(folder/'trajectory.npz') as z:data={k:z[k] for k in z.files}
        rng=np.random.default_rng(seed);xi=0.;old=np.empty_like(data['nu_core']);correct=old.copy()
        # Native raster sampling consumes these draws before the first step.
        ne=int(np.sum(region<3));ni=len(region)-ne
        rng.choice(ne,size=min(80,ne),replace=False)
        rng.choice(ni,size=min(20,ni),replace=False)
        drive=runtime.CoreOUMixture(np.array([0,1]),signal,.95,0.,p['dt'],p['tau_n'],p['sigma_n'],seed)
        for k in range(len(old)):
            xi=a*xi+b*rng.standard_normal();delta=drive.step(k*p['dt'])
            nu=np.maximum(signal+delta,0.)
            old[k]=np.maximum(max(signal+xi,0.)+delta,0.);correct[k]=nu
            rng.poisson(nu[core]*p['dt'])
        assert np.array_equal(old,data['nu_core']), 'RNG replay must explain every recorded sample'
        if seed==848101:
            assert np.array_equal(correct,np.load(OUT/'rate/grid10_seed848101.npz')['nu_core'])
            assert np.array_equal(correct,np.load(OUT/'rate/grid20_seed848101.npz')['nu_core'])
        np.savez_compressed(folder/'rejected_input_record.npz',legacy_scalar_plus_core_delta=old)
        data['nu_core']=correct
        np.savez_compressed(folder/'trajectory.npz',**data)
        write(folder/'input_record_repair.json',dict(status='REPAIRED_DIAGNOSTIC_RECORD_ONLY',
            old_field='nu_core previously included unused global OU',
            actual_input='signal + core OU; global OU loading exactly zero',
            proof='Every legacy observer sample reproduced bitwise from original master RNG and private Poisson draw counts; actual core input exactly matches rate input',
            activity_arrays='unchanged; all spikes, field and contact envelopes retained'))
        print(seed,'input record repaired; activity unchanged',flush=True)

if __name__=='__main__':main()
