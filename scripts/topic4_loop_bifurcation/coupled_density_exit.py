#!/usr/bin/env python3
"""Free recurrent/G/Z/M/K density continuation from the actual native entry.

The only prescribed future is the exact external mean-rate trajectory. Original
pending pulses belong to the initial state, not future neural teacher forcing.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,pickle,time
import numpy as np
from campaign import ROOT,read,write,sha
import density_spatial as physical
from reconstruct_loop_future_drive import OUT,INITIAL

ADAPTED=ROOT/'density_spatial_grouping/operators'
EXTRA=r'''
__device__ bool carried_cell(double* x,int& ref,const double* pars,const double* constants,
 const double* arr,double drive,const double* global,double gain,double nx,double ni,
 const double* pending,int tick,int depth,int member,int N){
 if(tick<depth){x[1]+=pending[(long long)tick*N+member]/constants[0];
               x[3]+=pending[((long long)depth+tick)*N+member]/constants[2];}
 return native_cell(x,ref,pars,constants,arr,drive,global,gain,nx,ni);
}
extern "C" __global__ void carried_particles(double* state,int* refs,unsigned char* memory,
 const double* pars,const double* constants,const double* arr,const double* drive,
 const int* clock,const double* global,double gain,int* spikes,int P,int R,int ndrive,
 const double* pending,const int* members,int depth,int N){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;int g=id/R,tick=clock[0];
 curandStatePhilox4_32_10_t* states=(curandStatePhilox4_32_10_t*)memory;
 curandStatePhilox4_32_10_t rng=states[id];float4 n=curand_normal4(&rng);states[id]=rng;
 double ar[4],x[8];int ref=refs[id];for(int j=0;j<4;j++)ar[j]=arr[j*P+g];
 for(int j=0;j<8;j++)x[j]=state[id*8+j];
 spikes[id]=carried_cell(x,ref,pars+6*g,constants,ar,drive[(long long)min(tick,ndrive-1)*P+g],
   global,gain,n.x,n.z,pending,tick,depth,members[id],N);
 refs[id]=ref;for(int j=0;j<8;j++)state[id*8+j]=x[j];
}
extern "C" __global__ void carried_supplied(double* state,int* refs,const double* normals,
 const double* pars,const double* constants,const double* arr,const double* drive,
 const int* clock,const double* global,double gain,int* spikes,int P,int R,
 const double* pending,const int* members,int depth,int N){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;int g=id/R,tick=clock[0],ref=refs[id];
 double x[8],ar[4];for(int j=0;j<8;j++)x[j]=state[id*8+j];for(int j=0;j<4;j++)ar[j]=arr[j*P+g];
 spikes[id]=carried_cell(x,ref,pars+6*g,constants,ar,drive[g],global,gain,normals[id*2],normals[id*2+1],
   pending,tick,depth,members[id],N);
 refs[id]=ref;for(int j=0;j<8;j++)state[id*8+j]=x[j];
}
extern "C" __global__ void observe_global(const double* global,const int* clock,double* output){
 int k=clock[0];if(k%10==0){int row=(k/10-1)%10;output[row*2]=global[0];output[row*2+1]=global[1];}
}
'''


class CarriedDensity(physical.DensityNetwork):
    def __init__(self,replicas,seed,device):
        self.carried_ready=False
        original=physical.OPERATORS
        try:
            physical.OPERATORS=ADAPTED
            super().__init__(replicas=replicas,seed=seed,device=device,duration_ms=10000,gain=30.)
        finally:physical.OPERATORS=original
        cp=self.cp
        with INITIAL.open('rb') as f:saved=pickle.load(f)
        assert saved['identity']==self.prep['graph_identity']
        state=saved['engine'];assert state['step']==100000 and len(state['V'])==40000
        ids=self.geo['cell_group'];assert np.array_equal(self.sizes,self.sizes.astype(int))
        members=np.array([np.flatnonzero(ids==g)[np.arange(replicas)%int(self.sizes[g])] for g in range(self.P)],dtype='i4')
        k=np.zeros(40000);k[:32000]=state['termination_mechanism']['sahp_g']
        native=np.stack([state['V'],state['s_E'],state['I_E'],state['s_I'],state['I_I'],
                         state['slow']['m'],state['slow']['z'],k],axis=1)
        self.members_cpu=members;self.native_initial=native
        self.initial_state_cpu=native[members];self.initial_ref_cpu=state['ref'][members]
        self.initial_global=np.array([state['termination_mechanism']['r_global'],state['global_feedback_response']['global_state']])
        order=(state['step']+np.arange(self.depth))%self.depth
        pending=np.stack([state['ring_sE'][order],state['ring_sI'][order]])
        assert pending.shape==(2,self.depth,40000)
        self.pending=cp.asarray(pending);self.members=cp.asarray(members)
        self.initial_state=cp.asarray(self.initial_state_cpu);self.initial_ref=cp.asarray(self.initial_ref_cpu)
        self.drive_cpu=np.load(OUT/'drive_0p1ms.npy',mmap_mode='r');assert self.drive_cpu.shape==(100000,self.P)
        self.drive=cp.asarray(self.drive_cpu);self.ndrive=100000
        self.extra=cp.RawModule(code=physical.CODE+EXTRA,options=('--fmad=false',),
            name_expressions=['carried_particles','carried_supplied','observe_global'])
        self.extra_k={name:self.extra.get_function(name) for name in ['carried_particles','carried_supplied','observe_global']}
        self.global_output=cp.zeros((10,2));self.carried_ready=True;self.reset()

    def reset(self):
        super().reset()
        if self.carried_ready:
            self.state[:]=self.initial_state;self.ref[:]=self.initial_ref
            self.global_state[:]=self.cp.asarray(self.initial_global);self.global_output.fill(0)

    def particles(self):
        self.extra_k['carried_particles'](((self.P*self.R+127)//128,),(128,),
            (self.state,self.ref,self.rng,self.pars,self.constants,self.arr,self.drive,self.clock,self.global_state,
             float(self.gain),self.spikes,np.int32(self.P),np.int32(self.R),np.int32(self.ndrive),
             self.pending,self.members,np.int32(self.depth),np.int32(40000)))

    def step(self):
        P=np.int32(self.P);R=np.int32(self.R);depth=np.int32(self.depth)
        self.k['delayed']((self.P,),(128,),(*self.ops,self.history,self.arr,self.clock,depth,P))
        self.particles()
        self.k['collect']((self.P,),(128,),(self.state,self.spikes,self.pars,self.history,self.rate,self.accumulator,self.output,self.clock,P,R,depth))
        self.k['global_step']((1,),(128,),(self.rate,self.pars,self.global_state,self.clock,P))
        self.extra_k['observe_global']((1,),(1,),(self.global_state,self.clock,self.global_output))


def check(device):
    e=CarriedDensity(64,927671,device);cp=e.cp;random=np.random.default_rng(927670);rows=[]
    for tick in [0,17,e.depth-1,e.depth]:
        e.reset();e.clock.fill(tick)
        arr=random.uniform(0,.5,(4,e.P));normal=random.normal(size=(e.P,e.R,2));nu=np.array(e.drive_cpu[tick])
        adjusted=e.initial_state_cpu.copy()
        if tick<e.depth:
            pulse=e.pending[:,tick].get()[:,e.members_cpu]
            adjusted[:,:,1]+=pulse[0]/e.constants_cpu[0];adjusted[:,:,3]+=pulse[1]/e.constants_cpu[2]
        expected=physical.cpu_cell(adjusted,e.initial_ref_cpu,e.pars_cpu,e.constants_cpu,arr,nu,e.initial_global,30.,normal)
        e.extra_k['carried_supplied'](((e.P*e.R+127)//128,),(128,),
            (e.state,e.ref,cp.asarray(normal),e.pars,e.constants,cp.asarray(arr),cp.asarray(nu),e.clock,e.global_state,
             30.,e.spikes,np.int32(e.P),np.int32(e.R),e.pending,e.members,np.int32(e.depth),np.int32(40000)))
        error=float(abs(e.state.get()-expected[0]).max());assert error<1e-10,error
        assert np.array_equal(e.ref.get(),expected[1]) and np.array_equal(e.spikes.get(),expected[2])
        rows.append(dict(tick=tick,max_state_error=error,spikes_ref_exact=True))
    # At a time after the pending queue, the changed sampling interface must
    # reduce bitwise to the original density particle kernel for constant drive.
    e.reset();e.clock.fill(e.depth+5);e.arr[:]=cp.asarray(random.uniform(0,.5,(4,e.P)))
    initial={k:getattr(e,k).get() for k in ['state','ref','rng']};nu=np.array(e.drive_cpu[e.depth+5])
    e.particles();got={k:getattr(e,k).get() for k in ['state','ref','rng','spikes']}
    for k,v in initial.items():getattr(e,k)[:]=cp.asarray(v)
    e.k['particles'](((e.P*e.R+127)//128,),(128,),
        (e.state,e.ref,e.rng,e.pars,e.constants,e.arr,cp.asarray(nu[None,:]),e.clock,e.global_state,30.,e.spikes,
         np.int32(e.P),np.int32(e.R),np.int32(1)))
    equal={k:np.array_equal(v,getattr(e,k).get()) for k,v in got.items()};assert all(equal.values()),equal
    e.reset()
    for _ in range(100):e.step()
    expected={k:getattr(e,k).get() for k in ['state','ref','rng','history','clock','global_state','output','global_output']}
    e.graph();e.chunk()
    captured={k:np.array_equal(v,getattr(e,k).get()) for k,v in expected.items()};assert all(captured.values()),captured
    assert np.array_equal(e.state.shape,(e.P,e.R,8))
    # Quantify finite replication of the initial empirical distribution.
    true=np.array([e.native_initial[e.geo['cell_group']==g].mean(0) for g in range(e.P)])
    error=np.max(abs(e.initial_state_cpu.mean(1)-true),axis=0)
    result=dict(status='PASS',pending_and_native_local_CPU=rows,after_queue_original_particle_bitwise=equal,
        graph_capture_bitwise=captured,initial_empirical64_max_group_mean_error=error.tolist(),
        initial_R_G_restored=True,initial_pending_history_retained=True,
        scope='Implementation only. Originalgroupthresholds/G/K/graph remain; Gaussian coupling andautonomous correspondence stillrequire measurements.')
    write(OUT/'implementation_check.json',result);print('COUPLED IMPLEMENTATION PASS',result,flush=True)
    del e;cp.get_default_memory_pool().free_all_blocks()


def run_one(seed,device):
    folder=OUT/f'num{seed}';folder.mkdir();start=time.time();e=CarriedDensity(2048,seed,device);cp=e.cp
    true=np.array([e.native_initial[e.geo['cell_group']==g].mean(0) for g in range(e.P)])
    initial_error=np.max(abs(e.initial_state_cpu.mean(1)-true),axis=0)
    write(folder/'identity.json',dict(initial_checkpoint=str(INITIAL),initial_sha256=sha(INITIAL),graph_identity=e.prep['graph_identity'],
        numerical_particles_per_group=e.R,numerical_seed=seed,initial_empirical_mean_max_error=initial_error.tolist(),
        native_source_input_seed=9108405,endogenous_future_prescribed=False,exogenous_expected_input_paired=True))
    e.graph();data=[];glob=[]
    for offset in range(0,10000,10):
        x=e.chunk();g=e.global_output.get();assert np.isfinite(x).all() and np.isfinite(g).all()
        assert x[:,1].min()>=0 and x[:,1].max()<=1 and x[:,3].min()>=0
        data.append(x.astype('f4'));glob.append(g)
        if (offset+10)%250==0:
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),time_s=10+(offset+10)/1000,
                mean_E_rate_Hz=float(np.average(x[:,0,e.E].mean(0),weights=e.sizes[e.E])),
                mean_Z=float(np.average(x[-1,1,e.E],weights=e.sizes[e.E])),mean_K=float(np.average(x[-1,3,e.E],weights=e.sizes[e.E])),
                R=float(g[-1,0]),Graw=float(30*g[-1,1]),elapsed_s=time.time()-start))
    data=np.concatenate(data);glob=np.concatenate(glob);cell=e.geo['group_cell'];field=np.zeros((len(data),400));count=np.zeros(400)
    for g in np.flatnonzero(e.E):field[:,cell[g]]+=data[:,0,g]*e.sizes[g];count[cell[g]]+=e.sizes[g]
    field/=count
    arrays=dict(time_ms=10000+np.arange(10000)+1.,field_E_Hz=field.astype('f4'),cell_counts=count,
                group_sizes=e.sizes,population_E=e.E,global_R_Hz=glob[:,0],global_s=glob[:,1])
    for j,key in enumerate(['group_rate_Hz','group_Z','group_M','group_K','group_IE','group_applied_II','group_V','group_abs_current']):arrays[key]=data[:,j]
    np.savez_compressed(folder/'trajectory.npz',**arrays)
    np.savez_compressed(folder/'final_state.npz',state=e.state.get(),ref=e.ref.get(),history=e.history.get(),rng=e.rng.get(),
        clock=e.clock.get(),global_state=e.global_state.get(),accumulator=e.accumulator.get(),particle_count=e.R,numerical_seed=seed)
    result=dict(status='COMPLETE',elapsed_s=time.time()-start,interval_s=[10,20],particles_per_group=e.R,groups=e.P,
        endogenous_R_G_Z_M_K_free=True,native_correspondence_certified=False,formal_bifurcation_allowed=False)
    write(folder/'result.json',result);write(folder/'progress.json',result);print('COUPLED RUN COMPLETE',seed,result,flush=True)
    del e;cp.get_default_memory_pool().free_all_blocks()


def main(device,wait,resume_setup=False):
    OUT.mkdir(exist_ok=True)
    if resume_setup:
        assert not list(OUT.glob('num*')) and not (OUT/'implementation_check.json').exists()
        previous=read(OUT/'contract.json')
        assert previous['initial_sha256']==sha(INITIAL) and previous['engine_sha256']==sha(physical.__file__)
        assert (OUT/'setup_recovery.json').exists(), 'Explicit terminal setup-error audit required'
    else:
        assert not (OUT/'contract.json').exists()
        write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_COUPLED_CONTINUATION',created_epoch=time.time(),
        question='After the actual native entry state, can the distribution-preserving spatial network itself buildG/K,terminate andbeginZrecovery withthe same externalforcing?',
        design='Exactlytwo10s continuations fromnative10s completejointstate,3479g40groups,2048particles/group,seeds927671/72,dt.1ms. DynamicZ/M/K,R/Gand recurrentactivity;nofuture neuronspikesorR/Gprovided.',
        initial='Balancedempiricalcopies ofcompleteoriginalcellstate/ref/pendingdelaypulses. Queuedpastinput isconsumedonce whiledelayednewrecurrentinputcomesfromthecandidate. Groupaveragethresholds retained.',
        forcing='Exact0.1msfloat64expectedexternalrate, matchedoriginalglobalandspatialOUstates/RNG. RealizedexternalPoissoncounts notprovided.',
        diagnostic='Nativeinitializationtests postentrymechanism transfer, not coldstartentry, acompleteautonomousloop,orindependentnative replication. Thesetwo numericalstreams do not establish numericalresolution convergence.',
        readouts='Nativeabsoluteclock10-20s: allE/core/surround ratesand400-cellspatialfields; causalR,G,K,Z; onsetofjointquiet,R<=5andGraw<2.6694,resource driftsand sustainedcoreactivity. No phasealignment ornativeonsetfitting.',
        decision='If noexitwithin10s while nativeexits, or core remainsactive despiteglobalquiet, rejectcoupling correspondence andlocalize it; no branchcertification. If comparableexitoccurs, inspect spatial/time residuals and numericalvariability; this remains a step toward fullnativevalidation, not a stability certificate.',
        resources='One additionaldensityGPU process, two sequentialruns; fixedlist/noautomaticgrid,seedorhorizonexpansion.',
        engine_sha256=sha(physical.__file__),producer_sha256=sha(__file__),initial_sha256=sha(INITIAL),formal_bifurcation_allowed=False))
    while not (OUT/'forcing_qa.json').exists():
        if not wait:raise RuntimeError('Exact forcing gate incomplete')
        write(OUT/'progress.json',dict(status='WAITING_FORCING_GATE',pid=os.getpid(),updated_epoch=time.time()));time.sleep(20)
    assert read(OUT/'forcing_qa.json')['status']=='PASS'
    check(device)
    completed=[]
    for seed in [927671,927672]:
        write(OUT/'progress.json',dict(status='RUNNING',pid=os.getpid(),numerical_seed=seed,completed=completed))
        run_one(seed,device);completed.append(seed)
    assert sha(physical.__file__)==read(OUT/'contract.json')['engine_sha256']
    result=dict(status='COMPLETE',completed=completed,engine_unchanged=True,actual_producer_sha256=sha(__file__),native_correspondence_certified=False,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);p.add_argument('--wait',action='store_true');p.add_argument('--resume-setup',action='store_true')
    a=p.parse_args();main(a.device,a.wait,a.resume_setup)
