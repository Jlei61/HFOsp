"""Full-history stability of a physically checked SCL-recruiting return cycle."""
from complete_rate_positive_stability import (
    DEST, PERIODIC_OUT, Path, read, write, np, release, compute_return, paired_modes)
from check_rate_live_Bleading_observer import physical_pass
import argparse
import os
import subprocess
import time


def verified_return_witness(source_index=24):
    """Revalidate the exact physical orbit and paired spectrum for consumers."""
    source=DEST/f'Bleading_extension/return_witness{source_index}/result.json'
    if not source.exists():return None
    q=read(source)
    if q['status'] not in ['NUMERICALLY_STABLE','UNSTABLE']:return None
    physical=read(q['physical_evidence'])
    row=next(v for v in physical['rows'] if Path(v['path']).resolve()==Path(q['orbit']).resolve())
    assert row['index']==source_index and physical_pass(row)
    z=np.load(q['orbit'])
    assert z['r'].shape==(4096,935)
    assert abs(float(z['J'])-q['J_EE_core'])<1e-12
    assert abs(float(z['T'])-q['T_ms'])<1e-8
    verdicts=[]
    for attempt in q['attempts']:
        pair=[read(f) for f in attempt['sources']]
        assert len(pair)==2
        assert all(Path(v['orbit']).resolve()==Path(q['orbit']).resolve() for v in pair)
        verdicts.append(paired_modes(*pair))
    accepted=[v for v in verdicts if v['status']!='UNRESOLVED']
    assert accepted and {v['status'] for v in accepted}=={q['status']}
    assert accepted[-1]['numerical_unstable_dimension']==q['classification']['numerical_unstable_dimension']
    return dict(source=str(source),evidence={**q,'classification':accepted[-1]})


def arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--after-pids',type=int,nargs='*',default=[])
    parser.add_argument('--min-free-gib',type=float,default=8.)
    parser.add_argument('--source-index',type=int,default=24)
    parser.add_argument('--physical-evidence',type=Path,
        default=PERIODIC_OUT/'arcBleadingConnection_20260920_prefix28_CPU_checks.json')
    return parser.parse_args()


def main(args):
    assert args.min_free_gib>=8
    assert args.source_index>=0
    folder=DEST/f'Bleading_extension/return_witness{args.source_index}'
    folder.mkdir(parents=True,exist_ok=True)
    evidence=args.physical_evidence
    source=read(evidence)
    assert source['status']=='SELECTED_POINTS_PASS'
    row=next(q for q in source['rows'] if q['index']==args.source_index)
    assert physical_pass(row)
    orbit=Path(row['path'])
    profile=np.load(orbit)
    assert profile['r'].shape==(4096,935)
    assert abs(float(profile['J'])-row['J'])<1e-12
    assert abs(float(profile['T'])-row['T_ms'])<1e-8
    assert read(orbit.with_suffix('.json'))['status']=='CONVERGED'
    attempts=[]
    def status(state,**kw):
        write(folder/'worker.json',dict(status=state,pid=os.getpid(),orbit=str(orbit),
            source_index=args.source_index,J_EE_core=row['J'],physical_evidence=str(evidence),attempts=attempts,**kw))
        print(state,kw,flush=True)
    dependencies={pid:Path(f'/proc/{pid}/cmdline').read_bytes()
        for pid in args.after_pids if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid,identity in list(dependencies.items()):
            try:active=Path(f'/proc/{pid}/cmdline').read_bytes()==identity
            except (FileNotFoundError,ProcessLookupError):active=False
            if not active:dependencies.pop(pid)
        if dependencies:
            status('WAITING_DEPENDENCIES',dependencies=list(dependencies));time.sleep(30)
    for nev,steps in [(6,[.05,.025]),(10,[.025,.0125])]:
        pair=[];sources=[]
        for dt in steps:
            release(args.device)
            while True:
                free=float(subprocess.check_output(['nvidia-smi','-i',str(args.device),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                if free>=args.min_free_gib*1024:break
                status('WAITING_GPU_RESOURCE',free_mib=free,required_free_gib=args.min_free_gib);time.sleep(30)
            label=f'Bleading_return{args.source_index}_20260920_k{nev}'
            target=PERIODIC_OUT/'poincare_floquet'/f'{label}_dt{dt:g}.json'
            status('POINCARE',nev=nev,dt_ms=dt)
            q=read(target) if target.exists() else compute_return(orbit,dt,nev,args.device,
                max(16,2*nev+4),stream_harmonics=True,output_label=label)
            assert Path(q['orbit']).resolve()==orbit.resolve()
            pair.append(q);sources.append(str(target))
        verdict=paired_modes(*pair)
        attempts.append(dict(sources=sources,classification=verdict))
        write(folder/'result.json',dict(status=verdict['status'],orbit=str(orbit),
            source_index=args.source_index,
            J_EE_core=row['J'],T_ms=row['T_ms'],physical_evidence=str(evidence),
            classification=verdict,attempts=attempts,
            scope='One full physical delay-history witness on the returning arm after LPC7. Its classification is not inherited from the stable J=.95 seed and does not classify the whole parameter interval.'))
        if verdict['status']!='UNRESOLVED':break
    release(args.device)
    status('WITNESS_FINISHED',scientific_status=verdict['status'])


if __name__=='__main__':
    args=arguments()
    try:main(args)
    except Exception as exc:
        folder=DEST/f'Bleading_extension/return_witness{args.source_index}';folder.mkdir(parents=True,exist_ok=True)
        path=folder/'worker.json';old=read(path) if path.exists() else {}
        write(path,{**old,'status':'COMPUTATION_FAILED','error':repr(exc)})
        raise
