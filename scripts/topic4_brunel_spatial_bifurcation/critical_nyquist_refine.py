"""Resolve remaining sharp determinant-phase turns near the two Hopf modes."""
from common import *
from model import SpatialBrunel
from response import characteristic
from nyquist import parity
from scipy.sparse.linalg import splu
BASE=OUT/'critical_revision'

def main():
    s=SpatialBrunel(response='calibrated_full');dest=BASE/'nyquist_refined';dest.mkdir(exist_ok=True);rows=[]
    for J in [.948,.955,.963]:
        source=read(BASE/'nyquist'/f'J{J:.6f}.json');r,ok,_=s.solve(J);assert ok
        cache={float(f):float(p) for f,p in zip(source['frequency_hz'],source['unwrapped_phase'])}
        for k in range(10):
            ff=np.array(sorted(cache));ph=np.unwrap(np.angle(np.exp(1j*np.array([cache[f] for f in ff]))))
            jumps=abs(np.diff(ph));count=-int(round((ph[-1]-ph[0])/np.pi))
            if jumps.max()<.25:break
            for i in np.flatnonzero(jumps>=.25):
                f=float((ff[i]+ff[i+1])/2);M=characteristic(s,r,J,2j*np.pi*f/1000).tocsc();lu=splu(M)
                cache[f]=float(np.angle(np.exp(1j*(np.angle(lu.U.diagonal()).sum()+np.pi*(parity(lu.perm_r)+parity(lu.perm_c))))))
        assert jumps.max()<.25
        data=dict(J_EE_core=J,unstable_root_count_candidate=count,frequency_hz=ff,unwrapped_phase=ph,maximum_phase_increment=float(jumps.max()),
            points=len(ff),additional_adaptive_refinements=k,source=str(BASE/'nyquist'/f'J{J:.6f}.json'),
            high_frequency_row_sum_bound=source['high_frequency_row_sum_bound'],meaning='Numerically refined full-determinant root count; sampled high-frequency endpoint has small loop norm, not an analytic bound on every unsampled tail frequency')
        write(dest/f'J{J:.6f}.json',data);rows.append({k:v for k,v in data.items() if k not in ['frequency_hz','unwrapped_phase']});print(rows[-1],flush=True)
    write(dest/'result.json',dict(status='COMPLETE',rows=rows))

if __name__=='__main__':main()
