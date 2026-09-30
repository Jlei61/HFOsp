"""Independent device control for the failed near-entry segment derivative."""
from common import OUT,np,write,log
from core_a_multiple_shooting import setup
from onset_state_continuation import build
from onset_segment_flow import SegmentDerivative
from fine_rate_frozen_Z_fields import capture
from onset_period_return import errors,dynamical_state
import argparse,os


def main(device):
    p=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap/actual_native_entry_midpoint/late_recurrence'
    d=p/f'segment1_device{device}_control';d.mkdir(exist_ok=True);assert not (d/'jobs.json').exists()
    write(d/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    write(d/'contract.json',dict(device=device,
        question='Does the retained nominal/derivative mismatch at the exact same near-entry segment also occur on the other GPU?',
        method='Same source, common-cycle coordinates,131.708333ms segment andseed92472 direction as the failed check. Two cached andthree original uncached full variational products; preserve all differences and all terminal nominal states.',
        scope='Numerical device control only, not attribution of a hardware fault or relaxation of any derivative or periodic gate.'))
    e=build(device);parts,T=setup(e,p/'three_burst_verified_replay',common_scale='cycle_rms',segments=6)
    A=parts[1];x=A.xref;rng=np.random.default_rng(92472);v=x*rng.normal(size=x.size);v/=np.linalg.norm(v)
    J=SegmentDerivative(A,x,T/6);outputs=[];rows=[];nominal=J.t.nominal_terminal
    np.savez_compressed(d/'nominal_reference.npz',**nominal)
    for k in range(2):
        q=J(v);outputs.append(q);rows.append(dict(kind='cached',repeat=k,norm=float(np.linalg.norm(q)),relative_to_first=float(np.linalg.norm(q-outputs[0])/np.linalg.norm(outputs[0]))))
        log('DEVICE SEGMENT CONTROL',device,rows[-1]);write(d/'progress.json',rows)
    old=SegmentDerivative(A,x,T/6,cached=False)
    for k in range(3):
        q=old(v);outputs.append(q);z=capture(e)
        row=dict(kind='uncached',repeat=k,norm=float(np.linalg.norm(q)),relative_to_first=float(np.linalg.norm(q-outputs[0])/np.linalg.norm(q)),nominal=errors(dynamical_state(nominal),dynamical_state(z),e.s.sizes/e.s.sizes.sum()))
        rows.append(row);write(d/'progress.json',rows);np.savez_compressed(d/f'nominal_uncached{k}.npz',**z)
        log('DEVICE SEGMENT CONTROL',device,row)
    for k,q in enumerate(outputs):np.save(d/f'output{k}.npy',q)
    np.save(d/'direction.npy',v)
    passed=all(r['relative_to_first']<1e-10 for r in rows) and all(r.get('nominal',{}).get('combined_relative_rms',0)<1e-12 for r in rows)
    write(d/'result.json',dict(status='DEVICE_CONTROL_PASS' if passed else 'DEVICE_CONTROL_MISMATCH',rows=rows,model_promoted=False))
    write(d/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args().device)
