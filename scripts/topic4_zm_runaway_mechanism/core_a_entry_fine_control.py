"""Physical time-step control of entry at D_A=.300, with all M dynamic."""
from common import OUT,np,read,write,log
import core_a_local_return_neighborhood as run
from fine_rate_frozen_Z_fields import capture,restore
from onset_state_continuation import regrid_state
import argparse

BASE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
DEST=BASE/'actual_D0300_fine';LABEL='from_interictal_history'
run.DEST=DEST;run.LABEL=LABEL;run.flow.DEST=DEST;run.local.DEST=DEST


def register():
    source=BASE/'actual_D0300';coarse=read(source/'contract.json')
    assert read(source/LABEL/'whole_record_audit.json')['status']=='AUDIT_PASS'
    DEST.mkdir(exist_ok=True);assert not(DEST/'contract.json').exists()
    c=read(source/'conditions.json')[LABEL].copy();c.update(dt_ms=.025,duration_ms=10000)
    np.savez_compressed(DEST/'fields.npz',**dict(np.load(source/'fields.npz')))
    write(DEST/'conditions.json',{LABEL:c})
    write(DEST/'contract.json',dict(
        question='Does the earlier onset-like entry into prolonged Core-A activity at Z_A=.700 remain present after halving the integration step from the identical physical initial history?',
        coarse_source=str(source),coordinates=coarse['coordinates'],conditions={LABEL:c},
        physical_scope=coarse['physical_scope'],
        numerical_change='Onlydt.05to.025ms. Preserve all original lag-grid history values; linearly interpolate intermediate physical lags. Same exact synaptic/covariance/memory/M initial state and full spatial Z field, same constant mean/private variance. No numerical or physical parameter refitting.',
        primary='Same whole-record local activity and quiet definitions, first10s compared withcoarse first10s. Record complete and censored durations and spatial/global readouts. Sensitive trajectories need not match individual event timing; a finite mismatch does not by itself prove either convergence or no bifurcation.',
        budget='One10s refined trajectory at the already observedD_A=.300 entry coordinate; no additional resource field.',
        acceptance='Exact refined10ms replay and preserved physical-lag initialization before run; independent complete readout after run. This is a time-step control, not an independent replicate or a criticality certificate.',model_promoted=False))


def check(device):
    c=read(DEST/'conditions.json')[LABEL];e=run.flow.build(device,c['dt_ms'])
    original=dict(np.load(c['initial']));expected=regrid_state(original,e,.05)
    Z=run.flow.initialize(e,c);base=capture(e)
    for key in base:
        if key=='syn':assert np.array_equal(base[key][:5],original[key][:5])
        else:assert np.array_equal(base[key],expected[key]),key
    a=e.chunk();end=capture(e);restore(e,base);b=e.chunk();again=capture(e)
    assert np.array_equal(a,b) and np.array_equal(a[:,0],a[:,1])
    assert all(np.array_equal(v,again[k]) for k,v in end.items())
    assert np.array_equal(end['syn'][5],Z) and not np.array_equal(end['syn'][4],base['syn'][4])
    assert np.all(end['parameters'][19]==0) and np.all(end['parameters'][20]==1)
    write(DEST/'implementation_check.json',dict(status='PASS',dt_ms=e.dt,
        replayed10ms_bitwise=True,physical_lag_initialization_preserved=True,
        entire_Z_held=True,all_M_dynamic=True,constant_mean_private_variance=True))
    log('EARLIER ENTRY FINE CHECK PASS',e.dt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit']);p.add_argument('--device',type=int,default=0)
    a=p.parse_args();{'register':register,'check':lambda:check(a.device),'run':lambda:run.flow.run(LABEL,a.device),'audit':run.audit}[a.command]()
