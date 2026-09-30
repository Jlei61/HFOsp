"""The current finite-count rate field with corrected physical-delay covariance.

The only changed operator is the private recurrent variance split. Graph,
response, Z/M, forcing, counts and actual transport delays stay unchanged.
"""
from common import OUT,np,read,write,log
from refractory_fine_forcing_pair import ForcedEngine,projections,readouts
from physical_delay_variance_split import physical_split
from fine_rate_frozen_Z_fields import capture
from scipy import sparse
from datetime import datetime
import argparse,os,time,gc

DEST=OUT/'physical_delay_count_rate'
BASELINE=OUT/'conditioned_refractory_fine_forcing/recorded_drive_binomial_seed1'


class PhysicalDelayCountEngine(ForcedEngine):
    def __init__(self,*args,count_sampling=True,constant_input=False,**kwargs):
        assert 'noise' not in kwargs
        super().__init__(*args,noise=True,**kwargs)
        s=self.s;private,qa=physical_split(s);self.legacy_private_data={};self.corrected_private_data={}
        for k,kind in enumerate(['ampa','gaba']):
            full=sparse.load_npz(s.folder/f'mean_{kind}.npz').tocsr();q=private[kind]
            assert np.array_equal(full.indices,q.indices) and np.array_equal(full.indptr,q.indptr)
            index=4*k+3;self.legacy_private_data[kind]=self.transport.ops[index]
            self.corrected_private_data[kind]=self.cp.asarray(q.data)
            self.transport.ops[index]=self.corrected_private_data[kind]
        self.legacy_split_qa=self.split_qa;self.split_qa=qa;self.noise=bool(count_sampling)
        if constant_input:self.transport.drive_on=False
        self.physical_delay_split=True


def register():
    assert read(OUT/'physical_delay_variance_split/result.json')['status']=='PHYSICAL_DELAY_UNIT_ERROR_CONFIRMED'
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='After correcting the proved delay-index unit error, does the same spatial finite-count rate model recover native interictal propagation and the dynamic Z/M onset path?',
        only_change='Private recurrence diffusion uses normalized current-filter covariance at actual0.1ms transmission-delay differences, independent of integrator step. Existing dt.05 incorrectly used half those lags.',
        unchanged='Physicalgraph andweights, originalthresholdmembers, g40 groups andcounts, conditioned39 frozenresponse, dt.05, originalfineexternaldrive, seed1Binomial innovations, initialallzero/Z1/M0, allZM parameters and both dynamiclaws.',
        runs=[dict(label='recorded_drive_binomial_seed1',seed=1)],dt_ms=.05,duration_ms=12500,
        acceptance='UnchangedoriginalA4 sixchecks andindependentoriginalreadouts. Local-responsefailuresstillretained. This is a physics-unit repair, not a new fit or acceptancewaiver.',
        checkpoint='Savecomplete9000msstate duringtheonefulltrajectory to support latermatchedconditionalchecks withoutreplay. Not an extra intervention.',
        implementation='Restoringlegacyprivateoperatorsmustreproduceprior100ms bitwise. Newprivateoperators matchindependentphysicaldelayintegrals, remainidentical atdt.05/.025, anddelayedarrivals matchdirectsparse multiplication; countbounds andZ/M activity checked.',
        budget='One12.5s fulltrajectory, no extraseeds,responsefit,parametersearch orbranchlaunch. Fullresult audited beforeany modelpromotion.'))


def check(device):
    assert (DEST/'contract.json').exists();rows=[];saved_private=None
    for dt in [.05,.025]:
        e=PhysicalDelayCountEngine(dt=dt,seed=1,device=device);s=e.s;cp=e.cp
        private=[e.corrected_private_data[k].get() for k in ['ampa','gaba']]
        if saved_private is None:saved_private=private
        else:assert all(np.array_equal(a,b) for a,b in zip(private,saved_private))
        if dt==.05:
            for k,kind in enumerate(['ampa','gaba']):e.transport.ops[4*k+3]=e.legacy_private_data[kind]
            e.graph();prefix=np.concatenate([e.chunk() for _ in range(10)]);old=np.load(BASELINE/'trajectory.npz')
            assert np.array_equal(prefix[:,0].astype('f4'),old['group_rate_hz'][:100])
            assert np.array_equal(prefix[:,1].astype('f4'),old['group_expected_rate_hz'][:100])
            for k,kind in enumerate(['ampa','gaba']):e.transport.ops[4*k+3]=e.corrected_private_data[kind]
        e.graph();rng=np.random.default_rng(920095);history=rng.uniform(0,.002,e.transport.history.shape)
        e.transport.history[:]=cp.asarray(history);tick=round(1234/dt);e.local.clock.fill(tick);e.arrivals()
        direct_history=history[(tick+1-round(.1/dt)*np.arange(1,len(s.delays)+1))%len(history)].ravel()
        matrices=[sparse.load_npz(s.folder/f'mean_{k}.npz') for k in ['ampa','gaba']]
        matrices += [sparse.load_npz(OUT/f'physical_delay_variance_split/physical_private_{k}.npz') for k in ['ampa','gaba']]
        expected=np.array([m@direct_history for m in matrices]);error=float(np.max(abs(expected-e.transport.arr.get())))
        assert error<1e-10; e.reset();cp.cuda.get_current_stream().synchronize();e.chunk();e.chunk()
        h=e.local.history.get()*s.sizes*dt;assert np.max(abs(h-np.rint(h)))<1e-10
        tick=int(e.local.clock.get()[0]);maximum=0.
        for mask,ref in [(s.E,2.),(~s.E,1.)]:
            used=h[(tick-np.arange(round(ref/dt)))%len(h)][:,mask].sum(0)
            assert np.all(used<=s.sizes[mask]+1e-9);maximum=max(maximum,float(np.max(used/s.sizes[mask])))
        rows.append(dict(dt_ms=dt,physical_private_identical=True,delayed_arrival_max_error=error,
            maximum_final_refractory_occupancy=maximum,finite_prefix_ms=20,Z_bounds=[float(e.syn[5].min().get()),float(e.syn[5].max().get())]))
        del e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(DEST/'implementation_check.json',dict(status='PASS',legacy100msbitwise=True,rows=rows,
        scope='Onlydelayunitsinprivatevariancechange; no scientificpromotion.'))
    log('PHYSICAL DELAY COUNT RATE CHECK PASS',rows)


def run(device):
    c=read(DEST/'contract.json');assert read(DEST/'implementation_check.json')['status']=='PASS'
    assert not (DEST/'jobs.json').exists();folder=DEST/'recorded_drive_binomial_seed1';folder.mkdir()
    jobs=dict(status='RUNNING',pid=os.getpid(),expected=1,completed=[],time_ms=0);write(DEST/'jobs.json',jobs)
    e=PhysicalDelayCountEngine(seed=1,device=device);s=e.s;e.graph();projection=projections(s,e.coarse,e.parent)
    write(folder/'identity.json',dict(graph=s.prep['graph_identity'],grid=40,groups=s.P,
        response='Unchanged conditioned39',native_future_spikes_used=False,private_split=e.split_qa,
        changed_operator='Physical0.1ms delaylags in normalized covariance; timestepindependent'))
    R=[];Z=[];M=[];start=time.time()
    for k in range(1250):
        x=e.chunk();assert np.isfinite(x).all() and x.min()>=-1e-9
        R.append(x);Z.append(e.syn[5].get());M.append(e.syn[4].get());assert Z[-1].min()>=0 and Z[-1].max()<=1
        if (k+1)==900:np.savez_compressed(folder/'checkpoint9000.npz',**capture(e))
        if (k+1)%100==0:
            jobs['time_ms']=(k+1)*10;write(DEST/'jobs.json',jobs)
            log('PHYSICAL DELAY RATE',(k+1)*10,'ms seconds',round(time.time()-start,1),'D',float(1-Z[-1][s.E]@s.mean_weights))
    r=np.concatenate(R);zs=np.array(Z);m=np.array(M);fields={g:(P@r[:,0].T).T for g,(P,n) in projection.items()}
    count=projection[20][1];whole=r[:,0,s.E]@s.mean_weights;t=np.arange(1,12501.);ts=np.arange(10,12501.,10);D=1-zs[:,s.E]@s.mean_weights
    assert np.max(abs(fields[20]@(count/count.sum())-whole))<1e-8
    events,summary,_,_=readouts(t,fields[20],count,'physical_delay_binomial_seed1')
    np.savez_compressed(folder/'trajectory.npz',time_ms=t,state_time_ms=ts,group_rate_hz=r[:,0].astype('f4'),
        group_expected_rate_hz=r[:,1].astype('f4'),field_E_hz=fields[20].astype('f4'),field_E_hz_grid40=fields[40].astype('f4'),
        global_E_hz=whole,cell_counts=count,Z=zs.astype('f4'),M_current=m.astype('f4'),D=D,parent_g20=e.parent,
        final_synaptic_slow_state=e.syn.get(),final_local_state=e.local.state.get(),final_own_history=e.local.history.get(),
        final_emitted_history=e.transport.history.get(),final_tick=e.local.clock.get(),dt_ms=e.dt)
    summary.update(status='COMPLETE',Z_and_M_dynamic=True,D9870=float(D[986]),D_final=float(D[-1]),
        events=[{k:v for k,v in ev.items() if k!='onset'} for ev in events],seconds=time.time()-start,model_promoted=False)
    write(folder/'result.json',summary);jobs.update(status='COMPLETE',completed=['recorded_drive_binomial_seed1'],time_ms=12500)
    write(DEST/'jobs.json',jobs);log('PHYSICAL DELAY RATE COMPLETE',summary['high_onset_ms'],summary['D9870'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    {'register':register,'check':lambda:check(a.device),'run':lambda:run(a.device)}[a.command]()
