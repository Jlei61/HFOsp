"""Average the original Z equation over conditional cycles; do not alter dynamics."""
from native_path import *
from streaming_periodic import StreamPeriodic
from floquet_v3 import orbit_states
from dynamics_v3 import THRESHOLD_Z,TAU_Z
from scipy.special import ndtr
import argparse,gc


def main(a):
    s=model();rows=[]
    for family,name in [('native','seed_N512'),('native','native_near_N2048'),('rate','rate_seed_N1024')]:
        (attach_native_path if family=='native' else attach_rate_entry_path)(s)
        path=OUT/'periodic'/f'{name}.npz';z=np.load(path);D=float(z['D']);sol=dict(r=z['r'],T=float(z['T']),D=D)
        o=StreamPeriodic(s,len(sol['r']),a.device);o.cache_mean_operators=False
        Y,_=orbit_states(o,sol,len(sol['r']));Y=Y[:-1]
        sd=np.sqrt(np.maximum(s.tm*Y[:,9]/(2*s.tau[1]),1e-20))
        zi=ndtr((THRESHOLD_Z-Y[:,7])/sd)
        flow=(zi-Y[:,11])*s.E/TAU_Z*1000 # per second
        avg=flow.mean(0);weights=s.sizes*s.E;weights=weights/weights.sum()
        s.set_D(D+1e-5);zp=s.Z.copy();s.set_D(D-1e-5);zm=s.Z.copy();s.set_D(D)
        direction=(zp-zm)/2e-5;dot=float(np.sum(weights*avg*direction))
        cosine=dot/np.sqrt(np.sum(weights*avg**2)*np.sum(weights*direction**2))
        rows.append(dict(orbit=str(path),family=family,D=D,T_ms=sol['T'],
            mean_D_drift_per_s=float(-np.sum(weights*avg)),
            Z_drift_A_B_surround_per_s=(np.array(s.regional_rates(avg))/1000).tolist(),
            conditional_Z_A_B_surround=(np.array(s.regional_rates(s.Z))/1000).tolist(),
            drift_direction_cosine_with_parameter_path=float(cosine),
            tangent_projected_D_drift_per_s=dot/float(np.sum(weights*direction**2)),
            interpretation='Cycle-averaged original Z vector field at a held-Z cycle; direction diagnostic, not a scalar closed slow model.'))
        log('SLOW DRIFT',rows[-1]);del o,Y;gc.collect()
    write(OUT/'cycle_slow_drift.json',dict(status='COMPLETE',rows=rows,
        equation='dZ/dt = E*(Phi((theta_Z-I_G)/sd)-Z)/tau_Z; evaluated without modifying the frozen-Z orbit',tau_Z_ms=TAU_Z))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args())
