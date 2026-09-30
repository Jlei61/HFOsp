"""Exchange M only, retaining each complete onset-neighbor fast history."""
from common import np,read,write,log
from onset_state_continuation import DEST,build,initialize,capture,restore,run
from datetime import datetime
import argparse


def register(field):
    assert field=='native9420'
    labels=[f'{field}_from_{side}' for side in ['lower','upper']]
    audits=[read(DEST/label/'independent_audit.json') for label in labels]
    assert all(a['status']=='AUDIT_PASS' for a in audits)
    assert audits[0]['windows'][-1]['complete_events']['n']>0
    assert audits[1]['windows'][-1]['quiet_fraction']==0
    assert all(read(DEST/(label+'_dt0025')/'independent_audit.json')['status']=='AUDIT_PASS' for label in labels)
    conditions=read(DEST/'conditions.json');new=[]
    for label,opposite in zip(labels,labels[::-1]):
        name=label+'_M_swapped';assert name not in conditions
        c=dict(label=name,field=field,initial=str(DEST/label/'checkpoint5000.npz'),
               M_initial_source=str(DEST/opposite/'checkpoint5000.npz'),
               previous_elapsed_ms=0,duration_ms=5000,comparison_condition=label,comparison_window_ms=[5000,10000])
        new.append(c);conditions[name]=c
    path=DEST/'M_history_control_contract.json';assert not path.exists()
    write(path,dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does the same-Z self-limited/non-self-limited difference survive exchanging only M between complete states?',
        conditions=new,
        intervention='At each existing5s checkpoint, exchange the entire group M current field with the opposite-history checkpoint. Preserve own AMPA/GABA, covariance, all36response memories, refractory/delayhistory, Z andclock. M immediately resumes the unchanged dynamic law; never frozen.',
        controls='The already completed unperturbed5-10s trajectories are exact controls from the same fast/history checkpoints, under identical deterministic input andfixedZ. No new baseline run needed.',
        interpretation='Persistence of each fast-history-associated behavior after M exchange rejects initialM alone as its explanation in this window. If classes exchange/collapse, Mstate is involved; neither outcome alone certifies a separatrix or a bifurcation type.',
        budget='Two5s M-exchange trajectories; original readouts andcomplete savedstates. No physicalparameterchange, newfit,modelpromotion,orcriticalsymbol.'))
    write(DEST/'conditions.json',conditions)
    log('ONSET M HISTORY CONTROL REGISTERED',new)


def check(device):
    contract=read(DEST/'M_history_control_contract.json');e=build(device);rows=[]
    for c in contract['conditions']:
        Z=initialize(e,c);state=capture(e)
        original=np.load(c['initial']);other=np.load(c['M_initial_source'])
        assert np.array_equal(state['syn'][4],other['syn'][4])
        assert np.array_equal(state['syn'][:4],original['syn'][:4])
        for key in ['local','history','clock']:assert np.array_equal(state[key],original[key])
        x=e.chunk();terminal=capture(e);restore(e,state);y=e.chunk();repeated=capture(e)
        assert np.array_equal(x,y)
        assert all(np.array_equal(v,repeated[k]) for k,v in terminal.items())
        assert np.array_equal(repeated['syn'][5],Z)
        rows.append(dict(label=c['label'],only_initial_M_changed=True,M_dynamic=True,full10ms_replay_bitwise=True))
    write(DEST/'M_history_control_implementation.json',dict(status='PASS',rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run'])
    p.add_argument('--field',default='native9420');p.add_argument('--label');p.add_argument('--device',type=int,default=0);a=p.parse_args()
    if a.command=='register':register(a.field)
    elif a.command=='check':check(a.device)
    else:
        assert read(DEST/'M_history_control_implementation.json')['status']=='PASS'
        assert a.label in [c['label'] for c in read(DEST/'M_history_control_contract.json')['conditions']]
        run(a.label,a.device)
