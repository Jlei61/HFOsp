"""Read-only resource-law prediction driven by the native-Z conditioned rate run.

The shadow Z never enters firing, currents, M or the supplied Z trajectory.
Exact trajectory replay checks protect the causal interpretation.
"""
from common import OUT, np, read, write, log
from transient_native_Z_path import PrescribedZEngine, DEST as SOURCE
from transient_response_network import install, LABEL, DEST as FREE
from physical_delay_count_rate import PhysicalDelayCountEngine
from fine_rate_frozen_Z_fields import capture
from datetime import datetime
import argparse, os, time

DEST=OUT/'transient_Z_feedback_shadow_20260923'


class ShadowEngine(PrescribedZEngine):
    def __init__(self,*args,prescribed=True,**kwargs):
        super().__init__(*args,**kwargs);self.prescribed=prescribed
        self.shadow=self.cp.ones(self.s.P);self.observed=self.cp.zeros((3,self.s.P))
        if not prescribed:self.transport.pars[19].fill(1)
        self.shadow_kernel=self.cp.RawKernel(r'''
extern "C" __global__ void shadow_Z(double* z,double* output,const double* syn,const double* local,
 const double* pars,const double* constants,int P,double dt){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 double sd=sqrt(fmax(local[5*P+g],1e-20));
 double target=.5*erfc((syn[3*P+g]-constants[8])/(sqrt(2.)*sd));
 double ez=exp(-dt/constants[7]);
 if(pars[3*P+g]>.5)z[g]=ez*z[g]+(1.-ez)*target;
 output[g]=syn[3*P+g];output[P+g]=local[5*P+g];output[2*P+g]=target;
}''','shadow_Z',options=('--fmad=false',))
        install(self)

    def step(self):
        if self.prescribed:super().step()
        else:PhysicalDelayCountEngine.step(self)
        self.shadow_kernel(((self.s.P+127)//128,),(128,),
            (self.shadow,self.observed,self.syn,self.local.state,self.transport.pars,self.transport.consts,
             np.int32(self.s.P),self.dt))

    def reset(self):
        super().reset();self.shadow.fill(1);self.observed.fill(0)


def register():
    assert read(SOURCE/'jobs.json')['status']=='COMPLETE'
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='With the correct full native Z supplied and fast/M correspondence improved, does the rate model\'s own GABA-current resource target still predict the wrong Z spatial path?',
        design='Replay the completed prescribed-native-Z corrected rate trajectory exactly. Integrate a separate shadowZ using precisely the unchanged Gaussian GABA threshold target/tauZ, at every0.05ms step. ShadowZ starts1 and has no feedback into the actual network.',
        shadow_equation='dZshadow/dt=(P(rawGABA<threshold)-Zshadow)/5000ms. Uses actual rate-engine postfilter GABA mean and raw-current variance state; not normalized diffusion, not Z-scaled variance.',
        controls='Free-shadow100ms must equal true freeZ bitwise. Prescribed replay must match all stored1ms sampled/expected rates and10ms M/Z, plus full saved checkpoints. Independent CPUone-step Gaussian/slowupdate must match GPU.',
        unchanged='Same graph, input, physicalprivateQ, response, countseed1, M law and prescribednativeZ. No response/slowparameter fit.',
        readouts='Native/free/conditional-shadow full spatial Z and cell-weighted global/coreA/coreB/surround paths. Exact exponentially weighted10ms target reconstructed from Z endpoints. GABA mean/rawvariance/target recorded at10ms for interpretation only.',
        interpretation='Wrongshadowpath shows target-input mismatch persists even conditional on nativeZ, but does not alone distinguish incorrect current statistics from the Gaussian target shape. Near-correctshadow with wrong freeZ implicates closed-loop amplification; neither alone certifies a bifurcation.',
        budget='One12.5s exact replay with nonfeedback observer, implementation checks and independent readout. No new seed, parameter scan or fitting.',model_promoted=False))


def check(device):
    import gc
    from scipy.special import ndtr
    for prescribed in [False,True]:
        e=ShadowEngine(seed=1,device=device,prescribed=prescribed);e.graph()
        original=np.load((SOURCE if prescribed else FREE)/LABEL/'trajectory.npz')
        r=original['group_rate_hz'][:100];expected=original['group_expected_rate_hz'][:100]
        for k in range(10):
            x=e.chunk()
            assert np.array_equal(x[:,0].astype('f4'),r[10*k:10*(k+1)])
            assert np.array_equal(x[:,1].astype('f4'),expected[10*k:10*(k+1)])
            if not prescribed:assert np.array_equal(e.shadow.get(),e.syn[5].get())
        old=e.shadow.get();e.step();e.cp.cuda.get_current_stream().synchronize()
        syn,local,con=e.syn.get(),e.local.state.get(),e.transport.consts.get()
        target=ndtr((con[8]-syn[3])/np.sqrt(np.maximum(local[5],1e-20)))
        ez=np.exp(-e.dt/con[7]);want=old.copy();want[e.s.E]=ez*old[e.s.E]+(1-ez)*target[e.s.E]
        err=float(abs(e.shadow.get()-want).max());assert err<1e-14
        del e;gc.collect()
    write(DEST/'implementation_check.json',dict(status='PASS',free_shadow_equals_actual_Z_bitwise=True,
        original_free_and_prescribed100ms_rates_bitwise=True,CPU_one_step_shadow_error=err,
        nonfeedback_observer=True))


def run(device):
    assert read(DEST/'implementation_check.json')['status']=='PASS'
    assert not (DEST/'jobs.json').exists()
    jobs=dict(status='RUNNING',pid=os.getpid(),time_ms=0);write(DEST/'jobs.json',jobs)
    e=ShadowEngine(seed=1,device=device);e.graph();source=np.load(SOURCE/LABEL/'trajectory.npz')
    # Cache compressed arrays once, never decompress within each chunk.
    rates=source['group_rate_hz'];expected=source['group_expected_rate_hz'];Z=source['Z'];M=source['M_current']
    shadow=[];observed=[];start=time.time();checkpoints=[]
    for k in range(1250):
        x=e.chunk();tm=(k+1)*10
        assert np.array_equal(x[:,0].astype('f4'),rates[10*k:10*(k+1)])
        assert np.array_equal(x[:,1].astype('f4'),expected[10*k:10*(k+1)])
        assert np.array_equal(e.syn[5].get().astype('f4'),Z[k])
        assert np.array_equal(e.syn[4].get().astype('f4'),M[k])
        shadow.append(e.shadow.get());observed.append(e.observed.get())
        if tm in [3000,8000,9000,9420,9870]:
            exact=np.load(SOURCE/LABEL/f'checkpoint{tm}.npz');state=capture(e)
            assert all(np.array_equal(v,exact[key]) for key,v in state.items())
            checkpoints.append(tm)
        if tm%1000==0:
            jobs.update(time_ms=tm);write(DEST/'jobs.json',jobs)
            log('Z FEEDBACK SHADOW',tm,'seconds',round(time.time()-start,1))
    zs=np.array(shadow);obs=np.array(observed)
    np.savez_compressed(DEST/'trajectory.npz',time_ms=np.arange(10,12501.,10),Z_shadow=zs,
        raw_GABA_mean=obs[:,0],raw_GABA_variance=obs[:,1],instantaneous_target=obs[:,2])
    write(DEST/'replay_audit.json',dict(status='PASS',full_1ms_sampled_expected_rates_bitwise=True,
        full_10ms_Z_M_bitwise=True,complete_state_checkpoints_bitwise=checkpoints,
        shadow_bounds=[float(zs.min()),float(zs.max())],seconds=time.time()-start))
    jobs.update(status='COMPLETE',time_ms=12500);write(DEST/'jobs.json',jobs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    {'register':register,'check':lambda:check(a.device),'run':lambda:run(a.device)}[a.command]()
