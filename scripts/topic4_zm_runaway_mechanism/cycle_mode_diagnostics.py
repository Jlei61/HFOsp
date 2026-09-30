"""Spatial deformation at a periodic branch turn, modulo an overall phase shift."""
from common import *
import argparse


def main(a):
    s=model();z=np.load(a.orbit);r=z['r'];N=len(r);v=z['tangent'][:r.size].reshape(r.shape)/1000
    dr=np.fft.irfft(np.fft.rfft(r,axis=0)*(2j*np.pi*np.arange(N//2+1))[:,None],n=N,axis=0)
    weights=s.sizes*s.E;coef=float(np.sum(v*dr*weights)/np.sum(dr*dr*weights));shape=v-coef*dr
    e=weights*np.mean(shape**2,axis=0);e/=e.sum();cells=s.geo['group_cell']
    field=np.bincount(cells[s.E],weights=e[s.E],minlength=s.grid**2)
    population=np.array([weights[s.geo['group_region']==i].sum() for i in range(3)])
    population/=population.sum()
    energy=np.array([e[s.geo['group_region']==i].sum() for i in range(3)])
    g=r[:,s.E]@s.mean_weights*1000
    q=dict(source=a.orbit,D=float(z['D']),T_ms=float(z['T']),
        mode_kind='phase-removed tangent to the periodic branch; not yet a converged Floquet mode',
        mean_hz=float(g.mean()),max_hz=float(g.max()),min_hz=float(g.min()),
        removed_phase_coefficient=coef,
        mode_energy_A_B_surround=energy.tolist(),
        E_population_fraction_A_B_surround=population.tolist(),
        per_neuron_energy_relative_to_global_A_B_surround=(energy/population).tolist(),
        effective_E_groups=float(1/np.sum(e**2)),largest_cell=int(field.argmax()),largest_cell_fraction=float(field.max()))
    out=Path(a.orbit).parent
    stem=Path(a.orbit).stem+'_spatial_tangent'
    np.savez_compressed(out/(stem+'.npz'),energy=e,field=field,rate_shape=shape,reference_rate=r)
    write(out/(stem+'.json'),q);log('CYCLE SHAPE MODE',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');main(p.parse_args())
