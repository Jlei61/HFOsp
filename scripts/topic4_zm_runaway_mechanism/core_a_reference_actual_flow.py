"""Actual same-history bridge between the interictal and long-activity fields.

Entry into prolonged activity and ultimate disappearance of quiet returns are
distinct questions. This20s record targets the former, at the same D_A as the
newly closed short-burst branch, using the actual original reference history.
"""
from common import OUT,np,read,write,log,model
from core_a_equilibrium_branch import Family
import core_a_local_return_neighborhood as run
import argparse

DEST=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap/actual_D0300'
LABEL='from_interictal_history'
run.DEST=DEST;run.LABEL=LABEL;run.flow.DEST=DEST;run.local.DEST=DEST


def register():
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'contract.json').exists()
    source=OUT/'core_a_resource_bifurcation_20260923/reference'
    assert read(source/'local_state_audit.json')['status']=='AUDIT_PASS'
    s=model(40);family=Family(s);z,tm=family.field(.30)
    base=dict(np.load(source/'final_state.npz'));assert np.array_equal(z[~family.A],base['syn'][5,~family.A])
    c=dict(label=LABEL,field=LABEL,initial=str(source/'final_state.npz'),source_dt_ms=.05,dt_ms=.05,
           duration_ms=20000,previous_elapsed_ms=0)
    np.savez_compressed(DEST/'fields.npz',**{LABEL:z});write(DEST/'conditions.json',{LABEL:c})
    write(DEST/'contract.json',dict(
        question='AtD_A=.300, does the actual interictal reference history still produce separated short events, or already access prolonged Core-A high-activity episodes?',
        rationale='The original question concerns entry into onset-like long activity. A later disappearance of all quiet visits could be a different event. Periodic-branch stability must be connected to the actual behavior between the audited D_A=.25572 reference and D_A=.32812 long-event state.',
        coordinates={LABEL:dict(D_A=.30,Z_A=.70,native_interpolation_time_ms=tm)},conditions={LABEL:c},
        physical_scope='Same full3479-group0.5mm spatial rate model and audited native within-A Z pattern family. Only A Z changes, outside-A Z bitwise native9s, entire Z held, every E M dynamic. Original constant input mean and private variance; no future count innovations or refitted parameters.',
        initial='Exact same actual reference final fast/M/history state as the already-audited D_A=.32812160 mid1_from_lower intervention. Starts from the original interictal history, not from a fitted periodic root or the high60s history.',
        primary='Whole20s distribution of complete local activity durations bounded by20ms quiet at5Hz, long episodes and low/high occupancy in continuous windows; spatial/global readout remains separate.10ms smoothing,1/10/50Hz robustness readouts. A few quiet returns do not by themselves disprove entry into a prolonged-activity regime.',
        budget='One20s trajectory at the already-selected branch coordinate. No automatic extra parameter campaign. Review against originalreference and existingmid1_from_lower records before refining the onset-relevant interval.',
        acceptance='Original exact10ms replay, unchanged complete initial state except A Z, independent fullgroup readout and complete source audit. Finite episodes and a numerical periodic branch do not establish a bifurcation without critical stability/manifold evidence.',model_promoted=False))
    log('REFERENCE ACTUAL FLOW REGISTERED',.30,.70)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit']);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();{'register':register,'check':lambda:run.check(a.device),'run':lambda:run.flow.run(LABEL,a.device),'audit':run.audit}[a.command]()
