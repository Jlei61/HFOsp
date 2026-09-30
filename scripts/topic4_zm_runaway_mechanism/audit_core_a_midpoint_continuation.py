"""Join complete-state continuations before determining activity boundaries."""
from common import np,read,write,model
from onset_state_continuation import regional_weights
from audit_core_a_natural_entry_step import intervals
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import argparse


def main(source,continuations=None,output_name='combined_to10s_audit.json'):
    parent=Path(source).resolve()
    folders=[parent]+[parent/p for p in (continuations or ['continuation_to10s'])]
    assert len(set(folders))==len(folders)
    for p in folders:assert read(p/'independent_audit.json')['status']=='INDEPENDENT_READOUT_PASS'
    for before,child in zip(folders[:-1],folders[1:]):
        c=read(child/'contract.json');assert Path(c['source']).resolve()==(before/'final_state.npz').resolve()
        assert c['target_native_time_ms'] is None and c['source_dt_ms']==c['dt_ms']
        qa=read(child/'initial_history_qa.json');assert qa['factor']==1 and qa['parent_integral_error']==0
    s=model(40);W=regional_weights(s);rates=[];parts=[];Z=None
    for folder in folders:
        with np.load(folder/'trajectory.npz') as a:
            r=a['group_rate_hz'].astype(float);regional=r@W.T
            bound=abs(np.spacing(a['group_rate_hz']).astype(float))@W.T
            assert np.all(abs(regional-a['regional_rate_hz'])<=bound+1e-10)
            if Z is None:Z=a['Z'].copy()
            else:assert np.array_equal(Z,a['Z'])
            rates.append(regional);parts.append(len(r))
    r=np.concatenate(rates);sm=uniform_filter1d(r,10,axis=0,mode='nearest');rows=[]
    for j,name in enumerate(['Global E','Core A','Core B','Surround']):
        quiet=[(a,b) for a,b in intervals(sm[:,j]<5) if b-a>=20]
        edges=[(0,0),*quiet,(len(r),len(r))]
        activity=[dict(start_ms=b,end_ms=c,duration_ms=c-b,left_censored=b==0,right_censored=c==len(r))
                  for (_,b),(c,_) in zip(edges[:-1],edges[1:]) if c>b]
        complete=[a for a in activity if not a['left_censored'] and not a['right_censored']]
        rows.append(dict(region=name,activities=activity,quiet_intervals_ms=quiet,
            max_complete_ms=max((a['duration_ms'] for a in complete),default=None),
            complete_after5s=[a for a in complete if a['start_ms']>=5000],
            complete_at_least500ms=[a for a in complete if a['duration_ms']>=500],
            mean_hz=float(r[:,j].mean())))
    result=dict(status='COMPLETE_RECORD_RECONSTRUCTION_PASS',Z_A=float(Z@W[1]),duration_ms=len(r),parts_ms=parts,
        full_state_continuation=True,no_Z_or_M_reset=True,rows=rows,
        sources=[str(p) for p in folders],
        interpretation=f'Event boundaries are detected after joining all recordedbins, so a boundary within20ms of any recording restart is not incorrectly censored. This is finite{len(r)}ms flow, not an attractor or bifurcation certificate.',model_promoted=False)
    write(parent/output_name,result)
    print(parent.name,'ZA',result['Z_A'],'completeA>=500ms',rows[1]['complete_at_least500ms'],'last',rows[1]['activities'][-1:],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source')
    p.add_argument('--continuation',action='append',help='Ordered continuation folders relative to source')
    p.add_argument('--output-name',default='combined_to10s_audit.json')
    a=p.parse_args();main(a.source,a.continuation,a.output_name)
