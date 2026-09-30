"""Spatial waveform deformation along the accepted native periodic family.

This is an exact BVP branch tangent with one global phase direction removed;
it is not a Floquet eigenvector, a propagation speed, or a causal ablation.
"""
from common import *


def summarize(r,tangent,s):
    n,p=r.shape;weights=s.sizes*s.E;weights=weights/weights.sum()
    # Stored tangent is d(r/RS)/dlogT; RS=.001, so this is Hz per logT.
    dr=tangent[:n*p].reshape(n,p)
    phase=np.fft.irfft(2j*np.pi*np.arange(n//2+1)[:,None]*
                      np.fft.rfft(r*1000,axis=0),n=n,axis=0)
    den=float(np.sum(phase**2*weights))
    projection=float(np.sum(dr*phase*weights)/den)
    transverse=dr-projection*phase
    energy=np.mean(transverse**2,axis=0)*weights
    assert energy.sum()>0
    energy/=energy.sum();region=s.geo['group_region'];cells=s.geo['group_cell']
    field=np.bincount(cells[s.E],weights=energy[s.E],minlength=s.grid**2)
    return dict(region_energy_fraction={name:float(energy[region==i].sum())
                    for i,name in enumerate(['A','B','surround'])},
        phase_component_removed=projection,
        removed_energy_fraction=float(projection**2*den/np.sum(dr**2*weights)),
        effective_spatial_cell_count=float(1/np.sum(field**2)),
        highest_energy_cell_xy_mm=[float((field.argmax()%s.grid+.5)*20/s.grid),
                                   float((field.argmax()//s.grid+.5)*20/s.grid)]),field


def main():
    s=model();folder=OUT/'periodic/native_turn_G8505_M65536'
    evaluations=read(folder/'evaluations.json')
    # Prespecified local diagnostic: three accepted derivatives nearest zero.
    selected=sorted(enumerate(evaluations),key=lambda iq:abs(iq[1]['dD_dT']))[:3]
    rows=[];fields=[]
    for index,row in selected:
        path=folder/f'eval_{index:03d}.npz';z=np.load(path)
        assert float(z['residual'])<2e-8 and row['linear_residual']<1e-7
        assert float(z['D'])==row['D'] and float(z['T'])==row['T_ms']
        result,field=summarize(z['r'],z['tangent'],s)
        # Changing only the phase gauge must not alter the spatial readout.
        n,p=z['r'].shape
        phase=np.fft.irfft(2j*np.pi*np.arange(n//2+1)[:,None]*
                          np.fft.rfft(z['r']*1000,axis=0),n=n,axis=0)
        v=z['tangent'].copy();v[:n*p]+=0.37*phase.ravel()
        _,shifted=summarize(z['r'],v,s)
        assert np.max(abs(shifted-field))<1e-11
        rows.append(dict(source=str(path),D=row['D'],T_ms=row['T_ms'],
                         dD_dT=row['dD_dT'],**result))
        fields.append(field)
        log('NATIVE TURN SPATIAL TANGENT',rows[-1])
    q=dict(status='DESCRIPTIVE_COMPLETE',rows=rows,
        observable='E-neuron-count-weighted cycle-average squared waveform derivative along log-period, after projecting out one common phase direction',
        numerical_checks='Accepted BVP and exact tangent residual gates; adding a global phase derivative leaves the readout unchanged',
        selection='Three accepted local derivatives nearest zero, not selected by spatial appearance',
        limitation='BVP family deformation only. No Floquet mode, bifurcation certification, core causality, propagation speed or necessity inferred.')
    write(OUT/'periodic/native_turn_spatial_tangent.json',q)
    np.savez_compressed(OUT/'periodic/native_turn_spatial_tangent.npz',
                        fields=np.array(fields),centers_mm=s.geo['centers_mm'])


if __name__=='__main__':main()
