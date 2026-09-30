"""Native unchanged physics, recording coarse projections to test information loss."""
from common import *
import argparse,time,inspect
from src.topic4_streaming_spike_readout import RecorderNumpy

def run(seed):
    folder=OUT/'native'/str(seed);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    start=time.time();s,groups,loading,det,applied,cores=runtime.setup(J,1.,seed)
    expected=read(V10/'native/a/applied_physics.json')['identity']
    assert applied['identity']==expected
    assert runtime.sha(inspect.getfile(runtime.ORIGINAL))==read(V10/'native/a/result.json')['engine_sha256']
    region=np.r_[np.where(cores[0]>=0,cores[0],2),np.where(cores[1]>=0,cores[1]+3,5)]
    positions=np.r_[s.positions_e,s.positions_i]
    maps={g:partition(positions,region,g) for g in (10,20,40)}
    original=runtime.Observer
    signal=read(OUT/'model_config.json')['signal_per_ms']
    class Observer(original):
        def __init__(self,*args):
            super().__init__(*args)
            self.projected={g:np.zeros((round(DURATION/2),len(labels)),np.uint16) for g,(_,labels,_) in maps.items()}
            self.six=np.zeros((round(DURATION/2),6),np.uint32)
            self.nu_core=np.empty((round(DURATION/.1),2),np.float32)
        def observe(self,t,tm,xi,nu,ext,delta,V,I_E,I_I,spk):
            # The scalar nu passed by the generic observer includes the legacy
            # global OU; its loading is zero in this executor. Actual input is
            # the constant signal plus the separate core drive.
            self.nu_core[t]=[max(0.,signal+delta[groups['coreAE'][0]]),max(0.,signal+delta[groups['coreBE'][0]])]
            if (t+1)%self.stride==0:
                cnt=self.count+spk;frame=(t+1)//self.stride-1
                for g,(idx,_,_) in maps.items():self.projected[g][frame]=np.bincount(idx,weights=cnt,minlength=self.projected[g].shape[1])
                self.six[frame]=np.bincount(region,weights=cnt,minlength=6)
            super().observe(t,tm,xi,nu,ext,delta,V,I_E,I_I,spk)
    runtime.Observer=Observer
    runtime.NumpyProxy=lambda shape:RecorderNumpy(shape,s.positions_e,s.montage,s.params.dt)
    result,obs=runtime.simulate(s,groups,loading,det,cores[0],seed,DURATION,folder/'progress.json')
    assert obs.nsteps==round(DURATION/s.params.dt)
    sp=result['E_spk_bool'];env,_,_=sp.envelope()
    arrays=dict(six_counts=obs.six,contact_envelope=env.T,field_counts=sp.native()['activity_counts'],nu_core=obs.nu_core)
    for g,(idx,lab,cell) in maps.items():
        sizes=np.bincount(idx);projected=obs.projected[g]
        weights=np.stack([np.bincount(idx[:s.n_e],weights=w,minlength=len(lab)) for w in sp.weights])
        coarse_env=smooth_contacts((projected/sizes)@weights.T)
        arrays.update({f'counts_{g}':projected,f'group_{g}':idx,f'region_{g}':lab,f'cell_{g}':cell,
                       f'contact_weights_{g}':weights,f'contact_envelope_{g}':coarse_env})
        assert np.array_equal(projected[:,lab<3].sum(1),obs.six[:,:3].sum(1))
    if seed==848101:
        old=np.load(V10/'native/a/trajectory.npz')
        assert np.array_equal(obs.six,old['six_group_counts_2ms'])
        assert np.array_equal(env.T.astype(np.float32),old['contact_envelope'])
        assert np.array_equal(arrays['field_counts'],old['sheet_activity_counts'])
    np.savez_compressed(folder/'trajectory.npz',**arrays)
    write(folder/'applied_physics.json',applied)
    write(folder/'result.json',dict(status='COMPLETE',seed=seed,J=J,duration_ms=DURATION,
        seconds=time.time()-start,source_parity='exact counts/contacts/field' if seed==848101 else 'same frozen executor',
        purpose='native variability and observation loss; no changed SNN physics'))
    write(folder/'progress.json',dict(status='COMPLETE',simulated_ms=DURATION,wall_s=time.time()-start))
    print(seed,'COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
