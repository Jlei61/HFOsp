"""Change only externally prescribed spatial forcing resolution at fixed g40."""
from common import OUT, np, read, write, log
from refractory_spatial_resolution import FineEngine, projections
from native_readouts import readouts
from datetime import datetime
import argparse, os, time, gc

DEST = OUT/'conditioned_refractory_fine_forcing'
FORCING = OUT/'native_fine_external_drive'
BASELINE = OUT/'conditioned_refractory_spatial_resolution'


class ForcedEngine(FineEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = np.load(FORCING/'drive.npz')
        assert np.array_equal(self.parent, data['parent_g20'])
        assert np.array_equal(self.transport.drive.get(), data['drive_g40_parent'])
        assert self.transport.drive_on and self.transport.n_drive == 12500
        self.transport.drive[:] = self.cp.asarray(data['drive_g40'])


def register():
    assert read(FORCING/'result.json')['status']=='FORCING_RECONSTRUCTION_COMPLETE'
    assert read(BASELINE/'jobs.json')['status']=='COMPLETE'
    DEST.mkdir(exist_ok=True); assert not (DEST/'contract.json').exists()
    c=read(BASELINE/'contract.json')
    c.update(created_local=datetime.now().astimezone().isoformat(),status='REGISTERED_BEFORE_FORCING_PAIR',
        question='Does restoring original0.5mm group external spatial variation recover both core lead directions and improve recruitment in the same ratefield?',
        only_change='Use original-member means of the reconstructed original spatialOU field instead of lifted1mmcell means. Same1ms forcingclock and originalstoredglobalrateprecision; allgraph,response,Z/M,countnoise andinitialization fixed.',
        reason='Previous finegrid keptcoarseexternalforcing. ExactspatialOU replay nowpasses7checkpoints; lostfinegroupforcing pooledRMSabout.044-.046perms, roughlyhalf theconfigured.1perms spatialfieldscale.',
        external_drive='OriginalSpatialOU replay at actualg40groupresolution; E/Ioriginalmembership. Exogenousonly, nofutureSNNspikes.1msholdingremainsanapproximation.',
        baseline=str(BASELINE),native_drive_source=str(FORCING/'drive.npz'),
        acceptance='Sameoriginalunconditionalnativeevent,quiet,duration,spatialparticipation,coreorder,entryandDreadouts. Inputrepair alone cannot waive localresponse orcontinuationfailures.',
        budget='Two12.5s pairedruns matching completedfineexpected andBinomialseed1. No furtherseed,parametersearch,responsefit orcontinuation.')
    write(DEST/'contract.json',c)


def check(device):
    data=np.load(FORCING/'drive.npz'); checks=[]
    for noise,label in [(False,'recorded_drive_expected'),(True,'recorded_drive_binomial_seed1')]:
        e=ForcedEngine(noise=noise,seed=1,device=device);cp=e.cp
        actual=e.transport.drive.get(); assert np.array_equal(actual,data['drive_g40'])
        assert np.array_equal(actual[:,~e.s.E],data['drive_g40_parent'][:,~e.s.E])
        # Restoring the baseline forcing must recover the saved baseline prefix.
        e.transport.drive[:]=cp.asarray(data['drive_g40_parent']);e.graph()
        prefix=np.concatenate([e.chunk() for _ in range(10)])
        original=np.load(BASELINE/label/'trajectory.npz')
        assert np.array_equal(prefix[:,0].astype('f4'),original['group_rate_hz'][:100])
        assert np.array_equal(prefix[:,1].astype('f4'),original['group_expected_rate_hz'][:100])
        e.reset();e.transport.drive[:]=cp.asarray(data['drive_g40']);e.graph();x=e.chunk()
        assert np.isfinite(x).all() and x.min()>=-1e-9
        checks.append(dict(label=label,baseline_prefix_ms=100,baseline_recorded_precision_bitwise=True,
                           newforcing_exact=True,Iforcing_unchanged=True,newprefix_finite_ms=10))
        del e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(DEST/'implementation_check.json',dict(status='PASS',checks=checks,scope='Only forcingprojection differs; no scientificpromotion.'))
    log('FINE FORCING CHECK PASS')


def run(device):
    c=read(DEST/'contract.json');assert read(DEST/'implementation_check.json')['status']=='PASS'
    jobs=dict(status='RUNNING',pid=os.getpid(),expected=2,completed=[]);assert not (DEST/'jobs.json').exists();write(DEST/'jobs.json',jobs)
    for row in c['runs']:
        folder=DEST/row['label'];folder.mkdir()
        e=ForcedEngine(dt=c['dt_ms'],noise=row['noise'],seed=row['seed'],device=device);s=e.s
        projection=projections(s,e.coarse,e.parent);e.graph();R=[];Z=[];M=[];start=time.time()
        write(folder/'identity.json',dict(graph=s.prep['graph_identity'],groups=s.P,grid=40,input_grid=40,readout_grid=20,
            original_cells=int(s.sizes.sum()),native_future_spikes_used=False,
            forcing_source=str(FORCING/'drive.npz'),response='conditioned_refractory_rate/fit/locked_weights.json',noise_split=e.split_qa))
        for k in range(1250):
            x=e.chunk();assert np.isfinite(x).all() and x.min()>=-1e-9
            R.append(x);Z.append(e.syn[5].get());M.append(e.syn[4].get());assert Z[-1].min()>=0 and Z[-1].max()<=1
            if (k+1)%100==0:
                log('FINE FORCING',row['label'],(k+1)*10,'ms','elapsed',round(time.time()-start,1),'D',float(1-Z[-1][s.E]@s.mean_weights))
                write(folder/'progress.json',dict(status='RUNNING',time_ms=(k+1)*10,pid=os.getpid()))
        r=np.concatenate(R);z=np.array(Z);m=np.array(M)
        fields={grid:(C@r[:,0].T).T for grid,(C,count) in projection.items()}
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
        jobs['completed'].append(row['label']);write(DEST/'jobs.json',jobs)
        log('FINE FORCING COMPLETE',row['label'],summary['high_onset_ms'],summary['D9870'])
        cp=e.cp;del e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    jobs['status']='COMPLETE';write(DEST/'jobs.json',jobs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run']);p.add_argument('--device',type=int,default=0);a=p.parse_args()
    {'register':register,'check':lambda:check(a.device),'run':lambda:run(a.device)}[a.command]()
