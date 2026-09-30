"""Seed the same conditional native-Z path from its physical D=0 low root.

The autonomous root is only a starting value; the conditional equations are
resolved with Z held and M dynamic. No stability or onset label is inherited.
"""
from native_high_equilibrium_branch import parameter_column
from native_path import *
import equilibria_v3 as arc


def main():
    s=model(); path=attach_native_path(s); arc.RS=.1; arc.DS=.1
    arc.param_derivative=parameter_column
    source=OUT/'full_ZM_equilibria/low.npz'
    r=np.load(source)['r']; s.set_D(0.)
    r,ok,tr=s.solve(r); assert ok and tr[-1]*1000<2e-8
    out=OUT/'equilibria/native_low_seed';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'endpoint.npz',r=r,D=s.D,Z=s.Z)
    s.set_D(1e-5);r,ok,tr=s.solve(r)
    assert ok and np.all(r>=0) and np.all(r<1/s.ref) and tr[-1]*1000<2e-8
    tangent=arc.tangent(s,r,s.D,direction=1)
    np.savez_compressed(out/'seed.npz',r=r,D=s.D,Z=s.Z,tangent=tangent)
    row=dict(status='CONDITIONAL_SEED_COMPLETE',source=str(source),path=path,
        D=s.D,global_E_hz=s.global_rate(r),residual_hz=float(abs(s.residual(r)).max()*1000),
        Z='held',M='dynamic',stability='NOT_INHERITED',
        scope='Low-rate conditional equilibrium on the existing native spatial path, including its explicit early endpoint extension. Not an observed resting attractor or an onset type.')
    write(out/'result.json',row);log('NATIVE LOW SEED',row)


if __name__=='__main__':main()
