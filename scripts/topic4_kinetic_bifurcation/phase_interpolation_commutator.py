"""Measure whether fractional native-step interpolation commutes with the map.

All tested physical steps use the unchanged model. This isolates the local
initialization error in the multi-phase return diagnostic from a full-period
defect. Interpolation order is a numerical check, not a model parameter.
"""
from generalized_return_spectrum import *


def run(a):
    rcfg=read(a.corrected/'config.json');source=Path(rcfg['source']);cfg=read(source/'config.json')
    folder=OUT/'phase_interpolation_checks'/a.label;folder.mkdir(parents=True,exist_ok=False)
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],a.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(source);coords=StateCoordinates(m);m.restore(a.corrected/'best_state')
    initial=capture(m);index=m.step_index;x=coords.pack(m)
    local=[np.zeros(coords.size)]
    for step in range(8):m.advance_step();local.append(cp.asnumpy(coords.pack(m)-x))
    def interpolate(phase,order,shift=0):
        value=x.copy()
        for c,d in zip(coefficients(phase,order),local[shift:shift+order+1]):
            value+=float(c)*cp.asarray(d)
        return value
    def reset(v):
        for k,value in initial.items():
            if k not in ('ordered_history','step_index'):getattr(m,k)[:]=value
        m.step_index=index;coords.unpack(v,m)
    rows=[]
    for phase in (0.,.25,.5,.75):
        prior=None;prior_order=None
        for order in (3,5,7):
            xa=interpolate(phase,order);target=interpolate(phase,order,1)
            reset(xa);reset_error=float(cp.linalg.norm(coords.pack(m)-xa).get())
            m.advance_step();actual=coords.pack(m);defect=actual-target
            blocks={k:float(cp.linalg.norm(defect[s]).get()) for k,s in coords.slices.items()}
            difference=None if prior is None else float(cp.linalg.norm(xa-prior).get())
            row=dict(phase=phase,order=order,reset_error=reset_error,
                one_native_step_commutation_defect=float(cp.linalg.norm(defect).get()),block_defects=blocks,
                initial_difference_from_previous_order=difference,previous_order=prior_order,
                input_distance_from_corrected_section=float(cp.linalg.norm(xa-x).get()))
            rows.append(row);print(row,flush=True);prior=xa;prior_order=order
            del target,actual,defect
    write(folder/'result.json',dict(status='LOCAL_INTERPOLATION_COMMUTATION_MEASURED',rows=rows,
        corrected_source=str(a.corrected.resolve()),native_step_ms=DT,
        interpretation='Local interpolation/map commutation only; a full native invariant curve still requires a global invariance check'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--corrected',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--device',type=int,default=0);run(ap.parse_args())
