"""Check actual 10-ms spatial Z fields against the 100-ms affine path slice.

Same frozen equations, same complete fast/M/delay history. This evaluates whether
the coarse conditional path can be mapped to the actual resource trajectory.
No future spike or rate trajectory is used to drive the continuation.
"""
from native_path import *
from endpoint_runs import EndpointIntegrator
import runner
import argparse
import subprocess
import json


def audit(s):
    trajectory=BASE/'runs/A4_det_meandrive/trajectory.npz'
    recorded=np.load(trajectory);path=attach_rate_entry_path(s)
    w=s.sizes*s.E;w=w/w.sum();D0=path['D'][0]
    direction=(path['fields'][1]-path['fields'][0])/(path['D'][1]-D0)
    rows=[]
    for t in range(7700,7801,10):
        Z=recorded['Z'][t//10-1].astype(float)
        D=float(1-Z@w);fit=path['fields'][0]+(D-D0)*direction
        rows.append(dict(time_ms=t,D=D,affine_same_D_Z_rms=float(np.sqrt(w@(Z-fit)**2)),
                         Z_A_B_surround=(np.array(s.regional_rates(Z))/1000).tolist()))
    assert rows[0]['affine_same_D_Z_rms']<1e-7 and rows[-1]['affine_same_D_Z_rms']<1e-7
    write(OUT/'fine_Z_path_audit.json',dict(status='COMPLETE',source=str(trajectory),rows=rows,
        storage='Actual Z sampled after every 10 ms, float32; endpoint checkpoints are float64',
        parameter_caution='The actual D first decreases and then increases within 7700--7800 ms. The existing bifurcation path is the straight spatial endpoint interpolation, not this curved trajectory.',
        claim='A conditional critical D on the affine slice must not be assigned directly as an autonomous trajectory onset time.'))
    return recorded


def main(a):
    s=model();recorded=audit(s)
    if a.audit_only:return
    runner.Integrator=EndpointIntegrator
    initial=OUT/'runs/rate_critical_near_restart_D0.1449700_dt0.05/trajectory.npz'
    rows=[]
    for t in a.times:
        Z=recorded['Z'][t//10-1].astype(float)
        label=f'actual_fine_Z_t{t}_matched_history_dt005'
        result=runner.run_condition(s,Z,label,a.duration,a.device,str(initial),dt=.05)
        q=json.loads(subprocess.check_output([sys.executable,str(HERE/'canonical_case_readout.py'),
                                             str(OUT/'runs'/label)],text=True))
        q.update(actual_Z_time_ms=t,D=result['D_initial'],source=str(OUT/'runs'/label/'trajectory.npz'))
        rows.append(q)
        write(OUT/'fine_Z_path_controls.json',dict(status='RUNNING',shared_initial=str(initial),rows=rows))
    write(OUT/'fine_Z_path_controls.json',dict(status='COMPLETE',shared_initial=str(initial),rows=rows,
        Z='actual recorded spatial fields, held',M='dynamic',
        claim='Finite-history control for the shape of the Z parameter path; not a bifurcation classification'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--times',nargs='+',type=int,default=[7770,7780])
    p.add_argument('--duration',type=int,default=8000);p.add_argument('--device',type=int,default=1)
    p.add_argument('--audit-only',action='store_true');main(p.parse_args())
