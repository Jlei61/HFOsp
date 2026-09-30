"""Extend the unresolved native 9.420-s Z-field controls in an independent directory.

All native engine state, M, delay history, OU state and RNG streams are preserved.
No historical output is modified or copied into the new observation window.
"""
import sys,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/topic4_fig5_z_state'))
import native_continue as native


def main(a):
    original=native.NATIVE
    parent=original/'runs'/a.parent
    assert native.read(parent/'result.json')['status']=='COMPLETE'
    protocol=native.read(original/'protocol.json')
    native.check_reference_sources()
    out=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918/native_validation'
    out.mkdir(parents=True,exist_ok=True);(out/'jobs').mkdir(exist_ok=True)
    native.write(out/'protocol.json',dict(protocol,extension_scope='Only new 20-s follow-up observations; historical windows remain at their original paths'))
    job=dict(native.read(original/'jobs'/f'{a.parent}.json'))
    parent_state=native.load_pickle(parent/'checkpoint.pkl')
    start=int(parent_state['engine']['step'])
    name=a.parent+f'_plus{a.duration}ms'
    job.update(name=name,kind='extension',parent=a.parent,start_step=start,duration_ms=a.duration,
               horizon_s=(start+round(a.duration/.1))*.0001)
    native.write(out/'jobs'/f'{name}.json',job)
    folder=out/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'checkpoint.pkl'
    if not checkpoint.exists():
        parent_state['job']=job
        parent_state['tracker']['stop_s']=1e9
        native.save_pickle(checkpoint,parent_state)
        native.write(folder/'continuation.json',dict(parent=str(parent),all_state_preserved=True,
            parent_checkpoint_sha256=native.sha(parent/'checkpoint.pkl'),old_chunks_copied=False,
            observation_start_step=start,created=time.time(),Z_held=True,M_dynamic=True))
    else:assert native.load_pickle(checkpoint)['job']==job
    native.NATIVE=out
    native.prepare=lambda:protocol
    native.seed_folder=lambda job,device:folder
    result=native.worker(name,a.device)
    print(name,result['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--duration',type=int,default=20000)
    p.add_argument('--device',type=int,default=0);main(p.parse_args())
