"""Form a targeted start from already computed original return derivatives.

The polynomial [G'(G'-a_M I)]^p reduces the almost-passive M cluster and near-zero
directions in the starting vector. The subsequent Arnoldi operator is still G'
itself; no eigenvalue or state variable is removed from the model. The broad
unfiltered spectrum remains part of the stability evidence.
"""
from pathlib import Path
import argparse,json,numpy as np
from summarize_generalized_modes import coefficients,OUT


def run(a):
    result=json.load(open(a.spectrum/'result.json'));cfg=json.load(open(a.spectrum/'config.json'))
    assert result['status']=='GENERALIZED_NORMAL_SPECTRUM_COMPLETE'
    storage=Path(cfg['storage']);H=np.load(a.spectrum/'arnoldi.npz')['H'];k=H.shape[1]
    assert 2*a.power<=k, 'All polynomial images must be contained in the saved Arnoldi relation'
    final=result['final'];order=cfg['interpolation_order'];n=cfg['integer_steps'];alpha=final['phase_fraction']
    decay=float(coefficients(alpha,order)@(1-.1/1000.)**(n+np.arange(order+1)))
    c=np.zeros(k+1);c[0]=1.;history=[]
    for _ in range(a.power):
        assert abs(c[-1])<1e-14
        once=H@c[:k];assert abs(once[-1])<1e-14
        c=H@once[:k]-decay*once
        norm=np.linalg.norm(c);assert norm>1e-14;c/=norm
        history.append(dict(coordinate_norm_before_rescale=float(norm),basis_coefficients=c.tolist()))
    base=OUT/'generalized_filters';disk=Path('/data/hfosp/topic4_sef_hfo/kinetic_population_bifurcation_20260916/generalized_filters')
    disk.mkdir(exist_ok=True,parents=True)
    if not base.exists():base.symlink_to(disk,target_is_directory=True)
    assert base.resolve()==disk.resolve()
    folder=base/a.label;folder.mkdir(exist_ok=False)
    first=np.load(storage/'q000.npy',mmap_mode='r');seed=np.zeros_like(first)
    for j,value in enumerate(c):
        if value:seed+=float(value)*np.load(storage/f'q{j:03d}.npy',mmap_mode='r')
    raw_norm=np.linalg.norm(seed);seed/=raw_norm
    root=Path(cfg['corrected_source']);normal=np.load(root/'section_normal.npy',mmap_mode='r')
    overlap=float(normal@seed);assert abs(overlap)<1e-10
    np.save(folder/'initial_vector.npy',seed)
    report=dict(status='FILTERED_INITIAL_VECTOR_PREPARED',source_spectrum=str(a.spectrum.resolve()),
        corrected_source=str(root.resolve()),D=cfg['D'],native_return_operator='Unchanged G derivative',
        starting_vector_polynomial='[Gprime * (Gprime - bare_M_decay * I)] ** power',power=a.power,
        bare_M_decay=decay,history=history,coordinate_reconstruction_norm=float(raw_norm),section_overlap=overlap,
        scope='Targeted Krylov start only. Background M directions remain in the model and unfiltered spectrum; no stability claim from filtering.')
    (folder/'config.json').write_text(json.dumps(report,indent=2)+'\n');print(folder)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spectrum',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--power',type=int,default=3);run(ap.parse_args())
