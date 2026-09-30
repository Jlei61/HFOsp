"""Characterize saved transverse Ritz modes without another physical replay.

Powers are in the fixed, documented StateCoordinates metric, not firing-rate
amplitudes. A nearly repeated eigenvalue permits mixtures of eigenvectors;
localization alone is not evidence that a core causes a bifurcation.
"""
from pathlib import Path
import argparse,csv,json,numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/kinetic_population_bifurcation_20260916'


def coefficients(alpha,order):
    return np.array([np.prod([(alpha-j)/(i-j) for j in range(order+1) if j!=i]) for i in range(order+1)])


def run(a):
    result=json.load(open(a.spectrum/'result.json'))
    assert result['status']=='GENERALIZED_NORMAL_SPECTRUM_COMPLETE'
    cfg=json.load(open(a.spectrum/'config.json'));root=Path(cfg['corrected_source'])
    rcfg=json.load(open(root/'config.json'));source=Path(rcfg['source'])
    geo=dict(np.load(OUT/'operators/selected_g40_theta0.25/geometry.npz'))
    names=('F','qa','ia','qg','ig','M','history');shapes={}
    with np.load(source/'checkpoint.npz') as z:
        for name in names:shapes[name]=z[name].shape
    slices={};offset=0
    for name in names:
        size=int(np.prod(shapes[name]));slices[name]=slice(offset,offset+size);offset+=size
    normal=np.load(root/'section_normal.npy',mmap_mode='r')
    final=result['final'];n=cfg['integer_steps'];alpha=final['phase_fraction'];order=cfg['interpolation_order']
    bare_M=float(coefficients(alpha,order)@(1-.1/1000.)**(n+np.arange(order+1)))
    additional=[]
    if a.reconstruct_indices:
        H=np.load(a.spectrum/'arnoldi.npz')['H'];dim=H.shape[1]
        values,coeff=np.linalg.eig(H[:dim]);ix=np.argsort(-abs(values));values=values[ix];coeff=coeff[:,ix]
        storage=Path(cfg['storage'])
        existing={(item['index'],item['part']) for item in result['mode_files']}
        for index in a.reconstruct_indices:
            assert 0<=index<dim
            target_value=complex(final['ritz_real'][index],final['ritz_imag'][index])
            assert abs(values[index]-target_value)<1e-12
            for part,c in [('real',coeff[:,index].real),('imag',coeff[:,index].imag)]:
                if np.linalg.norm(c)<1e-12 or (index,part) in existing:continue
                path=storage/f'mode_{index:02d}_{part}_cpu_reconstructed.npy'
                vector=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(offset,));vector[:]=0.
                for j,value in enumerate(c):
                    q=np.load(storage/f'q{j:03d}.npy',mmap_mode='r')
                    for start in range(0,offset,1000000):
                        vector[start:start+1000000]+=float(value)*q[start:start+1000000]
                vector.flush();del vector
                item=dict(index=index,part=part,path=str(path),eigenvalue=[target_value.real,target_value.imag],
                    origin='Reconstructed from the same saved Arnoldi basis and final Hessenberg eigenvector; no new dynamical product')
                additional.append(item);result['mode_files'].append(item)
        (a.spectrum/'reconstructed_mode_sources.json').write_text(json.dumps(additional,indent=2)+'\n')
    modes={}
    for item in result['mode_files']:modes.setdefault(item['index'],{})[item['part']]=Path(item['path'])
    rows=[];cells=[]
    for index,paths in sorted(modes.items()):
        vectors={part:np.load(path,mmap_mode='r') for part,path in paths.items()}
        assert all(len(v)==offset for v in vectors.values())
        group_power=np.zeros(len(geo['population']));block_power={}
        for name in names:
            shape=shapes[name];power=0.
            for v in vectors.values():
                block=v[slices[name]].reshape(shape)
                if name=='F':
                    for start in range(0,len(group_power),64):
                        per=np.sum(np.square(block[start:start+64]),axis=(1,2));group_power[start:start+64]+=per;power+=float(per.sum())
                elif name=='history':
                    per=np.sum(np.square(block),axis=0);group_power+=per;power+=float(per.sum())
                else:
                    per=np.square(block);group_power+=per;power+=float(per.sum())
            block_power[name]=power
        total=group_power.sum();ev=complex(final['ritz_real'][index],final['ritz_imag'][index])
        fractions={}
        for label,mask in [('coreA_E',(geo['population']==0)&(geo['group_region']==0)),
                           ('coreB_E',(geo['population']==0)&(geo['group_region']==1)),
                           ('other_E',(geo['population']==0)&(geo['group_region']==2)),
                           ('I',geo['population']==1)]:
            fractions[label]=float(group_power[mask].sum()/total)
        group_order=np.argsort(-group_power)[:12]
        phase_overlap=float(np.sqrt(sum(float(np.dot(normal,v))**2 for v in vectors.values())/total))
        row=dict(index=index,eigenvalue=[ev.real,ev.imag],modulus=abs(ev),
            ritz_absolute_residual=float(final['ritz_absolute_residual'][index]),
            bare_M_decay_multiplier=bare_M,distance_from_bare_M_multiplier=abs(ev-bare_M),
            coordinate_norm=float(np.sqrt(total)),section_normal_overlap=phase_overlap,
            coordinate_block_power_fraction={k:v/total for k,v in block_power.items()},
            population_region_power_fraction=fractions,
            largest_group_indices=group_order.tolist(),largest_group_power_fraction=(group_power[group_order]/total).tolist())
        rows.append(row)
        cells.append(np.bincount(geo['group_cell']+geo['population']*1600,weights=group_power/total,minlength=3200))
    report=dict(status='SAVED_RITZ_MODES_CHARACTERIZED',source=str(a.spectrum.resolve()),D=cfg['D'],rows=rows,
        metric='Fixed StateCoordinates metric of original source; includes density, currents, dynamic M and ordered delay memory',
        interpretation='Bare M decay is a reference scale, not a deflated or proven uncoupled mode. These Ritz vectors require spectral convergence before spatial causal interpretation.',
        limitation='Nearly repeated Ritz values admit mixed eigenvectors. These powers are not local firing-rate perturbation amplitudes.')
    (a.spectrum/'mode_characterization.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.spectrum/'mode_coordinate_spatial_power.npz',group_cell_power=np.asarray(cells),mode_index=[r['index'] for r in rows])
    with (a.spectrum/'mode_characterization.csv').open('w') as f:
        columns=['index','real','imag','modulus','ritz_residual','bare_M_distance','F','M','history','coreA_E','coreB_E','other_E','I']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in rows:
            writer.writerow(dict(index=row['index'],real=row['eigenvalue'][0],imag=row['eigenvalue'][1],modulus=row['modulus'],
                ritz_residual=row['ritz_absolute_residual'],bare_M_distance=row['distance_from_bare_M_multiplier'],
                **{k:row['coordinate_block_power_fraction'][k] for k in ('F','M','history')},**row['population_region_power_fraction']))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spectrum',type=Path,required=True)
    ap.add_argument('--reconstruct-indices',type=int,nargs='*',default=[],help='Also reconstruct these unsaved Ritz vectors from the existing Arnoldi relation; no new physical replay')
    run(ap.parse_args())
