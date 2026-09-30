"""Carry a corrected complete state to the next physical resource parameter.

Only the prescribed frozen spatial Z field changes. Dynamic M, voltage/noise
density, currents, and ordered delays remain as a numerical continuation guess.
This operation alone asserts no existence or stability of a branch at new D.
"""
from generalized_return_audit import *


def run(a):
    root=a.corrected;rcfg=read(root/'config.json');result=read(root/'result.json')
    assert result['status']=='GENERALIZED_RETURN_CORRECTED'
    old=root/'best_state';cfg=read(old/'config.json');assert 0<=a.D<=1
    base=OUT/'continuation_seeds';storage=Path('/data/hfosp/topic4_sef_hfo/kinetic_population_bifurcation_20260916/continuation_seeds')
    storage.mkdir(parents=True,exist_ok=True)
    if not base.exists():base.symlink_to(storage,target_is_directory=True)
    assert base.resolve()==storage.resolve()
    folder=base/a.label;folder.mkdir(parents=True,exist_ok=False)
    m=AutonomousDensity(a.D,cfg['degree'],cfg['voltage_dv'],a.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(old,allow_D_change=True);m.save(folder)
    write(folder/'config.json',dict(cfg,D=a.D,alpha=m.alpha,source_D=cfg['D'],source_corrected=str(root.resolve()),
        initial_ms=m.step_index*DT,numerical_initial_state='Complete corrected state transported to next prescribed physical Z(D); all dynamic state components retained',
        scientific_acceptance='Continuation initial guess only',suggested_return_time_ms=result['history'][-1]['effective_return_time_ms']))
    with np.load(old/'checkpoint.npz') as before,np.load(folder/'checkpoint.npz') as after:
        differences={k:float(np.max(abs(before[k]-after[k]))) for k in ('F','history','qa','ia','qg','ig','qe','ie','M')}
    assert max(differences.values())==0.
    write(folder/'seed_qa.json',dict(status='INITIAL_GUESS_PREPARED',unchanged_state_max_errors=differences,
        previous_D=cfg['D'],D=a.D,Z_min=float(m.Z.min().get()),Z_max=float(m.Z.max().get()),
        realized_D=1-float((m.e_weights@m.Z).get()),branch_acceptance='NOT_ESTABLISHED'))
    print(folder,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--corrected',type=Path,required=True)
    ap.add_argument('--D',type=float,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--device',type=int,default=0);run(ap.parse_args())
