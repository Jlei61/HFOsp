"""Full native state and observation parity for this exact scientific model."""
import copy, json, time
import numpy as np
import runtime as r
import checkpoint

def same(a,b,path='root'):
    if isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape,path
        assert np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for key in a:same(a[key],b[key],path+'.'+key)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'.'+str(i))
    else: assert a==b,path

def main():
    start=time.time()
    sub,groups,loading,det,applied,cores=r.setup(.85,1.,848101)
    native_capture=checkpoint.capture
    def capture(**kwargs):
        drive=kwargs.pop('external_drive')
        state=native_capture(external_drive=None,**kwargs)
        state['core_ou']=dict(state=drive.state.copy(),values=drive.values.copy(),n_steps=drive.n_steps,rng=copy.deepcopy(drive.rng.bit_generator.state))
        return state
    checkpoint.capture=capture
    saved=[]
    for fast in (False,True):
        snapshots=[]
        result,observer=r.simulate(sub,groups,loading,det,cores[0],848101,1000.,r.OUT/f'qa_{fast}_progress.json',fast=fast,capture=snapshots)
        assert len(snapshots)==1
        saved.append((snapshots,observer.arrays(),{k:result[k] for k in ('rate_E','rate_I','spk_inside','spk_outside','initial_V')}))
    same(*saved)
    # A matched earlier frozen trajectory checks the adapter against the actual historical executor.
    result,observer=r.simulate(sub,groups,loading,det,cores[0],847401,1000.,r.OUT/'qa_archived_progress.json',fast=True)
    path=r.REFERENCE/'response/units/curve_circle_out1_EI1/2511_847401/workers/trajectory.npz'
    with np.load(path) as old:
        # Archive saves float32, so compare after the same serialization conversion.
        for key in ('rate_E','rate_I'):
            assert np.array_equal(np.asarray(result[key],np.float32),old[key][:len(result[key])]),key
    payload=dict(status='PASS',full_state_bitwise=True,observer_bitwise=True,archived_baseline_rate_prefix_identical=True,
        duration_ms=1000.,checkpoint_ms=1000.-sub.params.dt,n_e=sub.n_e,n_i=sub.n_i,source_engine_sha256=r.sha(r.inspect.getfile(r.ORIGINAL)),
        runtime_sha256=r.sha(r.Path(r.__file__)),elapsed_s=time.time()-start,
        scope='This lowering-only core input model, plasticity and slow variables off; no patient or phase-state acceptance implied')
    r.write(r.OUT/'validation.json',payload)
    print(json.dumps(payload),flush=True)

if __name__=='__main__':main()
