"""Replay identity and paired finite-window readout for the native intervention."""
import sys,argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/topic4_fig5_z_state'))
import native_continue as N
import readouts as R
OUT=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
PAIR=OUT/'native_same_history_feedback'


def replay_audit():
    folder=PAIR/'runs/native_t9000_Zdynamic';rows=[]
    for f in sorted((folder/'chunks').glob('*.npz')):
        if '.tmp.' in f.name:continue
        reference=N.REPLAY_RUN/'chunks'/f.name
        assert reference.exists(), 'This audit uses aligned 9.0--12.5s chunks'
        a=np.load(f);b=np.load(reference)
        mismatches=[k for k in b.files if k not in a or a[k].dtype!=b[k].dtype or not np.array_equal(a[k],b[k])]
        rows.append(dict(chunk=f.name,keys_checked=len(b.files),mismatches=mismatches))
    complete=(folder/'result.json').exists() and N.read(folder/'result.json')['status']=='COMPLETE'
    state_differences=None
    if complete:
        assert len(rows)==7
        current=N.load_pickle(folder/'checkpoint.pkl')['engine']
        reference=N.replay_checkpoint(12500)
        state_differences=N.compare_states(current,reference)
    passed=bool(rows) and all(not q['mismatches'] for q in rows) and (not complete or state_differences==[])
    q=dict(status=('PASS' if complete else 'PREFIX_PASS_INCOMPLETE') if passed else 'FAIL',
        complete=complete,rows=rows,final_state_difference_keys=state_differences,
        source=str(N.REPLAY_RUN),intervention_arm=str(folder),
        scope='Unchanged dynamic Z/M arm must reproduce original native trajectory, all common chunk arrays and complete final engine state')
    N.write(PAIR/'dynamic_replay_qa.json',q);return q


def main(a):
    qa=replay_audit();print('DYNAMIC REPLAY',qa['status'],len(qa['rows']),'chunks',flush=True)
    if a.replay_only:return
    assert qa['status']=='PASS'
    g=np.load(PAIR/'geometry.npz');counts=g['cell_e_counts'];weights=counts/counts.sum()
    rows=[];data={};origins=[];innovations=[];final_inputs=[]
    for condition in ['held','dynamic']:
        folder=PAIR/'runs'/f'native_t9000_Z{condition}'
        assert N.read(folder/'result.json')['status']=='COMPLETE'
        job=N.read(PAIR/'jobs'/f'native_t9000_Z{condition}.json')
        applied=N.read(folder/'applied_configuration.json')
        assert applied['frozen_Z_state_update']==(condition=='held') and not applied['frozen_M_state_update']
        assert applied['Z_enabled'] and applied['M_effective_feedback']
        origins.append(N.read(folder/'continuation.json'))
        d=R.load_chunks(folder,keys=('spikes_1ms','field_1ms'))
        assert d['start_step']==90000 and d['end_step']==125000
        field=d['field_1ms']/counts*1000;cell=field.reshape(-1,10,400).mean(1)
        rate=R.rate_10ms(d['spikes_1ms'][:,0],N.NE)
        assert np.max(abs(rate-cell@weights))<1e-10
        seps,events=R.find_events(rate);high=R.high_rate_entry(rate)
        broad=(rate>=200)&((cell>50)@weights>=.75)
        broad_runs=[(int(lo),int(hi)) for lo,hi in R.runs_of(broad) if hi-lo>=20]
        complete_events=[dict(e,start_s=9+e['start_bin']*.01,end_s=9+e['end_bin']*.01) for e in events if e['qualifies']]
        windows=[]
        for start,end in [(9000,9420),(9420,10070),(10070,12500)]:
            stat=R.window_stats(rate,seps,events,cell,counts,(start-9000)//10,(end-9000)//10,high)
            windows.append(dict(absolute_window_ms=[start,end],**stat))
        final=N.load_pickle(folder/'checkpoint.pkl')['engine']
        final_inputs.append({key:final[key] for key in ['rng_state','external_drive','xi']})
        xi=np.concatenate([np.load(p)['xi'] for p in sorted((folder/'fields').glob('*.npz')) if '.tmp.' not in p.name])
        innovations.append(xi)
        rows.append(dict(condition=condition,source=str(folder),start_s=9.,end_s=12.5,
            global_Z_initial=origins[-1]['initial_global_Z'],global_Z_final=float(final['slow']['z'][:N.NE].mean()),
            M_feedback_final_mv=float(job['eta_m']*final['slow']['m'][:N.NE].mean()),
            high_entry_s=None if high is None else 9+high['onset_bin']*.01,
            high_confirmation_s=None if high is None else 9+high['confirmation_bin']*.01,
            broad_entry_s=None if not broad_runs else 9+broad_runs[0][0]*.01,
            complete_events=complete_events,windows=windows))
        data[f'field_{condition}_Hz']=field.astype(np.float32)
        data[f'global_{condition}_10ms_Hz']=rate
    assert origins[0]['source_sha256']==origins[1]['source_sha256']
    assert all(o['initial_state_bitwise_identical'] and not o['clock_rebased'] and not o['random_streams_replaced'] for o in origins)
    same_input=np.array_equal(*innovations);assert same_input
    input_state_differences=N.compare_states(*final_inputs);assert input_state_differences==[]
    q=dict(status='COMPLETE',rows=rows,dynamic_replay_qa=qa['status'],
        same_complete_initial_state=True,same_global_OU_innovations=same_input,
        final_input_state_difference_keys=input_state_differences,
        statistical_unit='One paired native history, seed9108401, starting at its actual9s checkpoint',
        time_window='Original clock9.0--12.5s; shorter than original4s classification window',
        asymptotic_category='NOT_ASSIGNED; operational entries, events and occupation only',
        Z='Held vs dynamic; only state-update intervention',M='dynamic in both arms',
        onset_type='NOT_INFERRED_FROM_THIS_FEEDBACK_INTERVENTION')
    np.savez_compressed(PAIR/'paired_fields.npz',**data,cell_counts=counts,centers_mm=g['centers_mm'])
    N.write(PAIR/'result.json',q);print(json.dumps(q,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay-only',action='store_true');main(p.parse_args())
