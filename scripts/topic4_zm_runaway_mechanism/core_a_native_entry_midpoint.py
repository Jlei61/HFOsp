"""One continuous-native-path point between short and prolonged local events.

Finite records nominate the entry neighborhood; they do not classify a
bifurcation. The same full initial state and dynamic M remain in use.
"""
from common import OUT,np,read,write,log,model
from core_a_parameter_path_audit import NativeTimeFamily
import core_a_native_entry_point as point
import argparse

BASE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
DEST=BASE/'actual_native_entry_midpoint';LABEL=point.LABEL
point.DEST=DEST;point.run.DEST=DEST;point.run.flow.DEST=DEST;point.run.local.DEST=DEST


def register():
    left=read(BASE/'actual_native9420'/LABEL/'whole_record_audit.json')
    right=read(BASE/'actual_D0300'/LABEL/'whole_record_audit.json')
    assert left['status']==right['status']=='AUDIT_PASS'
    def longest(q):
        row=next(r for r in q['thresholds'] if r['threshold_hz']==5.)
        return max(r['duration_ms'] for r in row['activities'] if not r['left_censored'] and not r['right_censored'])
    assert longest(left)<200 and longest(right)>1000,(longest(left),longest(right))
    theta=.5*(left['coordinates']['native_interpolation_time_ms']+right['coordinates']['native_interpolation_time_ms'])
    s=model(40);family=NativeTimeFamily(s);z,D=family.field_at_time(theta)
    source=OUT/'core_a_resource_bifurcation_20260923/reference/final_state.npz'
    initial=dict(np.load(source));assert np.array_equal(z[~family.A],initial['syn'][5,~family.A])
    DEST.mkdir(exist_ok=True);assert not(DEST/'contract.json').exists()
    c=dict(label=LABEL,field=LABEL,initial=str(source),source_dt_ms=.05,dt_ms=.05,duration_ms=20000,previous_elapsed_ms=0)
    np.savez_compressed(DEST/'fields.npz',**{LABEL:z});write(DEST/'conditions.json',{LABEL:c})
    write(DEST/'contract.json',dict(question='Where along the continuous CoreA spatial-resource path after Fig.5 state3 do separated short events first give way to prolonged local activity from the same complete history?',
        coordinates={LABEL:dict(D_A=D,Z_A=1-D,native_interpolation_time_ms=theta)},conditions={LABEL:c},
        source_endpoints=[str(BASE/'actual_native9420'/LABEL/'whole_record_audit.json'),str(BASE/'actual_D0300'/LABEL/'whole_record_audit.json')],
        selection='One midpoint in native path time between the independently audited20s short-event and prolonged-event records. D_A is only the plotted coordinate; interpolate original per-group fields continuously, no first-upcrossing lookup.',
        physical_scope='Same3479-group full drift, original response/graph/delays/privatevariance. Only withinCoreA Z differs. OutsideA native9s; everyZ held percondition and everyE M dynamic. Same original reference full fast/M/delay state and original constant external mean. No futureSNN spikes.',
        readout='Whole20s original10ms-smoothed A rate; quietbelow5Hz for20ms; complete and censored activities plus1/10/50Hz controls and independent all-group spatial reconstruction.',
        budget='One20s deterministic point; actual root continuation and stability remain the primary type test. No finite-window permanence or unique bifurcation value inferred.',
        acceptance='Full initial identity exceptA Z and exact10ms replay before launch; whole-record independent audit after. No change to periodic root, phase, mesh or Floquet gates.',model_promoted=False))
    log('CONTINUOUS NATIVE ENTRY MIDPOINT REGISTERED',theta,D,1-D)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit']);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();{'register':register,'check':lambda:point.check(a.device),'run':lambda:point.run.flow.run(LABEL,a.device),'audit':point.run.audit}[a.command]()
