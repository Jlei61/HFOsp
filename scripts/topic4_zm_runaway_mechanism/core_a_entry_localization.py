"""One earlier Core-A field targeting first entry into prolonged activity."""
from common import OUT,np,read,write,log,model
from core_a_equilibrium_branch import Family
import core_a_local_return_neighborhood as run
import argparse

DEST=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap/actual_D0270'
LABEL='from_interictal_history'
run.DEST=DEST;run.LABEL=LABEL;run.flow.DEST=DEST;run.local.DEST=DEST


def register():
    DEST.mkdir(exist_ok=True);assert not(DEST/'contract.json').exists()
    prefix=read(DEST.parent/'actual_D0300'/LABEL/'prefix10s_audit.json')
    assert prefix['status']=='PREFIX_READOUT_PASS_FULL20S_STILL_RUNNING'
    source=OUT/'core_a_resource_bifurcation_20260923/reference'
    assert read(source/'local_state_audit.json')['status']=='AUDIT_PASS'
    s=model(40);family=Family(s);z,tm=family.field(.27)
    base=dict(np.load(source/'final_state.npz'))
    assert np.array_equal(z[~family.A],base['syn'][5,~family.A])
    c=dict(label=LABEL,field=LABEL,initial=str(source/'final_state.npz'),source_dt_ms=.05,dt_ms=.05,
           duration_ms=20000,previous_elapsed_ms=0)
    np.savez_compressed(DEST/'fields.npz',**{LABEL:z});write(DEST/'conditions.json',{LABEL:c})
    write(DEST/'contract.json',dict(
        question='AtD_A=.270/Z_A=.730, has the actual interictal history already accessed long Core-A activity, or does it continue to produce separated short events over the same20s observation?',
        rationale='D_A=.300 already produces complete3.290s and2.428s episodes after short~.1s events. This places entry into prolonged activity before the laterD_A~.35 disappearance-of-quiet question. Choose the existingD_A=.270 periodic-spectrum coordinate inside the earlier interval, rather than refining an unrelated equilibrium fold.',
        coordinates={LABEL:dict(D_A=.27,Z_A=.73,native_interpolation_time_ms=tm)},conditions={LABEL:c},
        physical_scope='Full3479-group spatial rate model unchanged. Only native within-Core-A Z pattern changes; outside Z bitwise native9s. All Z held, all E M dynamic, complete original fast/M/delay initial state shared withD_A=.300 and.32812 interventions. Original constant external mean and physical private Q, no future native/count innovations.',
        primary='Complete and censored local activity durations separated by20ms quiet at10ms-smoothed5Hz, whole-record occupancy, global/B/surround readouts.1/10/50Hz diagnostics retain threshold sensitivity.',
        budget='One matched-history20s trajectory atone existing branch coordinate. No automatic additional parameter campaign; inspect the actual contrast before refining further.',
        acceptance='Exact10ms replay and full initial-state identity except A Z before launch. Independent all-group spatial/readout audit after completion. Finite observation nominates the interval; no bifurcation type or asymptotic stability from no-return censoring.',model_promoted=False))
    log('EARLIER ENTRY FIELD REGISTERED',.27,.73)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit']);p.add_argument('--device',type=int,default=0)
    a=p.parse_args();{'register':register,'check':lambda:run.check(a.device),'run':lambda:run.flow.run(LABEL,a.device),'audit':run.audit}[a.command]()
