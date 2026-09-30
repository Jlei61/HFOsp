"""Fixed-response spatial discretization diagnostic:1mm versus0.5mm.

Same native physical graph andcellthresholds; same existing recorded1mm drive
lifted to subcells, so inputresolution does not change with dynamicresolution.
"""
from common import OUT,BASE,np,read,write,log,model
import refractory_spatial_diagnostic as original
from refractory_count_consistency import CountEngine
from native_readouts import readouts,window_stats
from scipy import sparse
from datetime import datetime
import argparse,time,os,gc

DEST=OUT/'conditioned_refractory_spatial_resolution'


def mapping(coarse,fine):
    assert coarse.prep['graph_identity']==fine.prep['graph_identity']
    assert np.array_equal(coarse.geo['original_positions'],fine.geo['original_positions'])
    lo=np.full(fine.P,coarse.P,dtype=int);hi=np.full(fine.P,-1,dtype=int)
    np.minimum.at(lo,fine.geo['cell_group'],coarse.geo['cell_group'])
    np.maximum.at(hi,fine.geo['cell_group'],coarse.geo['cell_group']);assert np.array_equal(lo,hi)
    Q=sparse.csr_matrix((fine.sizes/coarse.sizes[lo],(lo,np.arange(fine.P))),shape=(coarse.P,fine.P))
    assert np.max(abs(Q@fine.theta-coarse.theta))<1e-12
    assert np.array_equal(np.bincount(lo,weights=fine.sizes,minlength=coarse.P),coarse.sizes)
    assert np.array_equal(fine.E,coarse.E[lo])
    return lo,Q


class FineEngine(CountEngine):
    def __init__(self,*args,**kwargs):
        self.coarse=model(20);fine=model(40);self.parent,self.restrict=mapping(self.coarse,fine)
        old_model,old_drive=original.model,original.group_drive
        def lifted_drive(s,label):
            assert s is fine
            return old_drive(self.coarse,label)[:,self.parent]
        original.model=lambda:fine;original.group_drive=lifted_drive
        try:super().__init__(*args,**kwargs)
        finally:original.model=old_model;original.group_drive=old_drive
        assert self.s is fine


def projections(s,coarse,parent):
    result={}
    for grid,cells in [(20,coarse.geo['group_cell'][parent]),(40,s.geo['group_cell'])]:
        count=np.bincount(cells[s.E],weights=s.sizes[s.E],minlength=grid*grid)
        C=sparse.csr_matrix((s.sizes[s.E]/np.maximum(count[cells[s.E]],1),
            (cells[s.E],np.flatnonzero(s.E))),shape=(grid*grid,s.P))
        result[grid]=(C,count)
    return result


def register():
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    diag=read(OUT/'native_input_bridge/spatial_resolution_spread.json')
    assert diag['status']=='READ_ONLY_DESCRIPTIVE_COMPLETE'
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),status='REGISTERED_BEFORE_FINE_RATE_RUNS',
        question='Does resolving spatialcurrentvariation at0.5mm improve autonomousinterictal/onset/spreading behavior under the unchanged refractoryrate response?',
        reason='Native9-9.42s within-group netcurrentspread falls to31-32percent with actual0.5mm subdivision,versus84-96percent with matchedsize shuffledsubgroups.',
        graph='Original nativegraph projectedg40,3479groups; all40000originalcells,samephysicalweights,delays,thresholdmembers,externalinputandZMparameters.',
        response='Frozen conditioned_refractory_rate weights, unchanged42localstates andownrate refractory integral. BothZandMdynamic; externalmeanAMPAfilter restored.',
        external_drive='Original1mm recordeddrive is lifted via exactgroupmembership to0.5mmsubgroups. No newinputresolution or newexternalrealization; not exactpercell nativeforcing.',
        output='Primarymetrics use original20x20cellfields after originalcell-count-weighted restriction; fine40x40fields also stored. No changed participation threshold or spatialmetricresolution.',
        noise='Expectedrate arm andoneactual-count Binomial/refractory/M arm, with same originalcellN pernewgroup. StationaryPoisson privatevariance splitting remains approximate.',
        initialization='Zero synapses/covariance/history,Z1,M0 at0s; no nativefuture spikes or state transplant.',
        runs=[dict(label='recorded_drive_expected',noise=False,seed=1),dict(label='recorded_drive_binomial_seed1',noise=True,seed=1)],
        dt_ms=.05,duration_ms=12500.,coarse_sources=['conditioned_refractory_external_filter_pair/recorded_drive_expected','conditioned_refractory_count_consistency/recorded_drive_binomial_seed1'],
        verification='Exactnestedmembership andweightedthresholdmean; delayedoperators commute under parent-constant history; externalforcing invariant; count/conservation checks before fullruns.',
        acceptance='Same nativeearlyinterictal/lateinterictal,quiettime,eventsize,propagation,entryandDgates. Mesh sensitivity alone cannot waive localresponse failures or certify bifurcation.',
        budget='Two12.5s runs only; no fitting, extra seeds, networkparametersearch or continuation.'))


def check(device):
    e=FineEngine(device=device);s=e.s;c=e.coarse;cp=e.cp;P=s.P;rng=np.random.default_rng(920120)
    history=rng.uniform(0,.002,(e.transport.depth,c.P));e.transport.history[:]=cp.asarray(history[:,e.parent]);checks=[]
    ops=[sparse.load_npz(c.folder/(n+'.npz')) for n in ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']]
    for clock in [0,e.transport.depth-1,e.transport.depth+10]:
        e.local.clock.fill(clock);e.arrivals();fine=e.transport.arr.get()
        vals=history[(clock+1-2*np.arange(1,c.prep['max_delay_steps']+1))%len(history)].ravel()
        coarse=np.array([q@vals for q in ops]);error=float(abs((e.restrict@fine.T).T-coarse).max());assert error<1e-9
        checks.append(dict(clock=clock,restricted_operator_error=error))
    for k in [0,4999,9420]:
        coarse=original.group_drive(c,'seed9108401')[k]
        assert np.array_equal(e.transport.drive[k].get(),coarse[e.parent])
    e.reset();e.graph();rate=e.chunk();assert np.isfinite(rate).all()
    pro=projections(s,c,e.parent);C,n=pro[20]
    assert np.array_equal(n,np.bincount(c.geo['group_cell'][c.E],weights=c.sizes[c.E],minlength=400))
    err=float(abs((C@rate[:,0].T).T@(n/n.sum())-rate[:,0,s.E]@s.mean_weights).max());assert err<1e-9
    del e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    e=FineEngine(noise=True,seed=1,device=device);e.graph();e.chunk();e.chunk();s=e.s
    assert e.local.history.data.ptr==e.transport.history.data.ptr
    h=e.transport.history.get()*s.sizes*e.dt;integer_error=float(abs(h-np.rint(h)).max());assert integer_error<1e-10
    tick=int(e.local.clock.get()[0]);minimum_available=s.sizes.copy()
    for mask in [s.E,~s.E]:
        nref=round(float(s.ref[mask][0])/e.dt)
        used=h[(tick+1-np.arange(1,nref))%len(h)][:,mask].sum(0)
        minimum_available[mask]-=used
    assert minimum_available.min()>=-1e-9
    write(DEST/'implementation_check.json',dict(status='PASS',fine_groups=s.P,coarse_groups=c.P,operator_checks=checks,
        forcing_lift_bitwise=True,rate_restriction_error=err,emitted_count_integer_error=integer_error,
        minimum_available_neurons=float(minimum_available.min()),count_history_shared=True,
        scope='Numerical andmodelidentity checks; no autonomousscientific acceptance.'))
    log('FINE RATE IMPLEMENTATION PASS',checks)


def run(device):
    c=read(DEST/'contract.json');assert read(DEST/'implementation_check.json')['status']=='PASS'
    jobs=dict(status='RUNNING',pid=os.getpid(),expected=2,completed=[]);assert not (DEST/'jobs.json').exists();write(DEST/'jobs.json',jobs)
    for row in c['runs']:
        folder=DEST/row['label'];folder.mkdir();e=FineEngine(dt=c['dt_ms'],noise=row['noise'],seed=row['seed'],device=device)
        s=e.s;projection=projections(s,e.coarse,e.parent);e.graph();R=[];Z=[];M=[];start=time.time()
        write(folder/'identity.json',dict(graph=s.prep['graph_identity'],groups=s.P,grid=40,input_grid=20,readout_grid=20,
            original_cells=int(s.sizes.sum()),native_future_spikes_used=False,response='conditioned_refractory_rate/fit/locked_weights.json',noise_split=e.split_qa))
        for k in range(1250):
            x=e.chunk();assert np.isfinite(x).all() and x.min()>=-1e-9
            R.append(x);Z.append(e.syn[5].get());M.append(e.syn[4].get());assert Z[-1].min()>=0 and Z[-1].max()<=1
            if (k+1)%100==0:
                log('FINE RATE',row['label'],(k+1)*10,'ms','elapsed',round(time.time()-start,1),'D',float(1-Z[-1][s.E]@s.mean_weights))
                write(folder/'progress.json',dict(status='RUNNING',time_ms=(k+1)*10,pid=os.getpid()))
        r=np.concatenate(R);z=np.array(Z);m=np.array(M);fields={}
        for grid,(C,count) in projection.items():fields[grid]=(C@r[:,0].T).T
        field=fields[20];count=projection[20][1];whole=r[:,0,s.E]@s.mean_weights
        assert np.max(abs(field@(count/count.sum())-whole))<1e-8
        t=np.arange(1,12501.);ts=np.arange(10,12501.,10);D=1-z[:,s.E]@s.mean_weights
        events,summary,_,_=readouts(t,field,count,row['label'])
        np.savez_compressed(folder/'trajectory.npz',time_ms=t,group_rate_hz=r[:,0].astype('f4'),group_expected_rate_hz=r[:,1].astype('f4'),
            field_E_hz=field.astype('f4'),field_E_hz_grid40=fields[40].astype('f4'),global_E_hz=whole,cell_counts=count,
            Z=z.astype('f4'),M_current=m.astype('f4'),D=D,state_time_ms=ts,parent_g20=e.parent,
            final_synaptic_slow_state=e.syn.get(),final_local_state=e.local.state.get(),final_own_history=e.local.history.get(),
            final_emitted_history=e.transport.history.get(),final_tick=e.local.clock.get(),dt_ms=e.dt)
        summary.update(status='COMPLETE',Z_and_M_dynamic=True,D9870=float(D[986]),D_final=float(D[-1]),
            events=[{k:v for k,v in ev.items() if k!='onset'} for ev in events],seconds=time.time()-start,model_promoted=False)
        write(folder/'result.json',summary);write(folder/'progress.json',dict(status='COMPLETE',time_ms=12500))
        jobs['completed'].append(row['label']);write(DEST/'jobs.json',jobs);log('FINE RATE COMPLETE',row['label'],summary['high_onset_ms'],summary['D9870'])
        cp=e.cp;del e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    jobs['status']='COMPLETE';write(DEST/'jobs.json',jobs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run']);p.add_argument('--device',type=int,default=0);a=p.parse_args()
    {'register':register,'check':lambda:check(a.device),'run':lambda:run(a.device)}[a.command]()
