"""Factor-two step checks for the completed onset-relevant state pair."""
from common import np,read,write,log
from onset_state_continuation import DEST,build,initialize,capture,restore,run
from datetime import datetime
import argparse


def register(labels):
    conditions=read(DEST/'conditions.json');new=[]
    for label in labels:
        original=conditions[label]
        assert original.get('dt_ms',.05)==.05 and original['previous_elapsed_ms']==0
        assert original['duration_ms']==10000
        assert read(DEST/label/'independent_audit.json')['status']=='AUDIT_PASS'
        name=label+'_dt0025'
        assert name not in conditions
        path=DEST/label/'checkpoint5000.npz';assert path.exists()
        c=dict(label=name,field=original['field'],initial=str(path),source_dt_ms=.05,dt_ms=.025,
               previous_elapsed_ms=0,duration_ms=5000,comparison_condition=label,comparison_window_ms=[5000,10000])
        new.append(c);conditions[name]=c
    path=DEST/'step_refinement_contract.json';assert not path.exists()
    write(path,dict(created_local=datetime.now().astimezone().isoformat(),
        question='Do the same onset-neighbor self-limited/persistent behaviors survive halving the integration step with the complete physical state matched?',
        conditions=new,
        changed='Onlydt .05to.025ms. Preserve synapses, localcovariance/memory, M, Z andphysicaltime. Fill interleaved delay/refractoryhistory nodes bylinearinterpolation; every oldphysical-lag node remains bitwiseidentical. No countinnovations; constantinput andphysicalprivateQ unchanged.',
        comparison='Each fine5s continues the exact coarse5s checkpoint and compares the coarse5-10s window. Irregular trajectories need not match phase/eventbyevent. Originalquiet,event,spatialparticipation andconditionalstate categories are unchanged.',
        numerical_gate='Verifyexactrestored10msfineflow, fixedZ,dynamicM, samephysicaltime, retainedoldhistorynodes, and originalcoarsehistoryreconstruction. Different long-term category invalidates a timestep-independent transition claim.',
        budget=f'{len(new)} fine5s continuations only; no thresholds/physicalparameter/modelpromotion changes.',
        scope='Numericalstateconsistency; notcriticalpointrefinement,Floquet,orfullSNNvalidation.'))
    write(DEST/'conditions.json',conditions)
    log('ONSET STEP REFINEMENT REGISTERED',new)


def check(device):
    contract=read(DEST/'step_refinement_contract.json');e=build(device,dt=.025);rows=[]
    for c in contract['conditions']:
        Z=initialize(e,c);source=np.load(c['initial']);initial=capture(e)
        assert np.array_equal(initial['syn'],source['syn'])
        assert np.array_equal(initial['local'],source['local'])
        old_tick=int(source['clock'][0]);new_tick=int(initial['clock'][0])
        old=source['history'];new=initial['history']
        assert new_tick*e.dt==old_tick*.05
        assert np.array_equal(new[(new_tick-2*np.arange(len(old)))%len(new)],old[(old_tick-np.arange(len(old)))%len(old)])
        x=e.chunk();terminal=capture(e);restore(e,initial);y=e.chunk();repeated=capture(e)
        assert np.array_equal(x,y) and np.array_equal(x[:,0],x[:,1])
        assert all(np.array_equal(v,repeated[k]) for k,v in terminal.items())
        assert np.array_equal(repeated['syn'][5],Z)
        rows.append(dict(label=c['label'],physical_time_exact=True,old_history_nodes_bitwise=True,
                         initial_synapses_local_M_Z_exact=True,fine10ms_full_replay_bitwise=True))
    write(DEST/'step_refinement_implementation.json',dict(status='PASS',rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run'])
    p.add_argument('--labels',nargs='+');p.add_argument('--label');p.add_argument('--device',type=int,default=0);a=p.parse_args()
    if a.command=='register':register(a.labels)
    elif a.command=='check':check(a.device)
    else:
        assert read(DEST/'step_refinement_implementation.json')['status']=='PASS'
        assert a.label in [c['label'] for c in read(DEST/'step_refinement_contract.json')['conditions']]
        run(a.label,a.device)
