"""Recompute the displayed observables of saved continuation solutions."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import csv, hashlib, json, sys
import numpy as np
from scipy.signal import resample, find_peaks
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/topic4_sef_hfo'
V2=BASE/'core_burst_bifurcation_v2_20260915'
V5=BASE/'core_branch_connections_v5_20260915'
V7=BASE/'core_network_bifurcation_v7_20260916'
OUT=BASE/'core_spatial_readout_v10_20260916'
OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
def write(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def identity(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def orbit(path):
    with np.load(path) as z:
        r=z['r'];g=float(z['g']);T=float(z['T']);err=float(z['residual'])
    # Fourier collocation has uniform phase j/N, no duplicated endpoint.
    dense=resample(r,32768,axis=0)*1000
    half=resample(r,16384,axis=0)*1000
    return dict(path=str(path),solution_id=identity(path),J_exact=g,T_full_ms=T,
        mesh=dict(kind='uniform periodic Fourier phase',points=len(r),dt_ms=T/len(r),endpoint=False),
        state_array='r',state_units='spikes/ms/cell',residual=err,
        mean=(r.mean(0)*1000).tolist(),lo=dense.min(0).tolist(),hi=dense.max(0).tolist(),
        extrema_grid_error_hz=float(max(abs(dense.min(0)-half.min(0)).max(),abs(dense.max(0)-half.max(0)).max())))

def audit():
    seq=read(V7/'displayed_curve_sequences.json');out=[];changes=[];cache={}
    for ib,rows in enumerate(seq):
        line=[]
        for j,row in enumerate(rows):
            path=row['path']
            if path not in cache:cache[path]=orbit(path)
            r=dict(cache[path],branch_id=f'mother_{ib:02d}',continuation_order=j,
                stable=row['stable'],family=row.get('family',f'saved_segment_{ib}'),period_multiple=1)
            assert abs(r['J_exact']-row['g'])<1e-12
            assert abs(r['T_full_ms']-row['T'])<1e-8
            changes.append(dict(solution_id=r['solution_id'],mean_error_hz=float(np.max(abs(np.array(r['mean'])-row['mean']))),
                extrema_change_hz=float(max(np.max(abs(np.array(r['lo'])-row['lo'])),np.max(abs(np.array(r['hi'])-row['hi']))))))
            line.append(r)
        out.append(line)
    daughters=[]
    for label,folder in [('PD1','mixed_lower_period2'),('PD2','mixed_period2'),('PD3','tonic_lower_period2')]:
        line=[]
        for j,amp in enumerate([.03,.06,.12]):
            path=V5/'periodic'/folder/f'amp{amp:g}_N4096.npz'
            r=orbit(path)
            sp=V5/'poincare'/folder/path.stem/'rk4_orthogonal_dt0.05.json'
            floquet=read(sp)
            r.update(branch_id=label+'_2T',continuation_order=j,stable=label!='PD2',family=folder,
                period_multiple=2,amplitude=amp,spectrum_source=str(sp),max_transverse=floquet['max_transverse'])
            assert (r['max_transverse']<1)==r['stable']
            line.append(r)
        daughters.append(line)
    write('periodic_branches.json',out+daughters)
    eq=read(V2/'equilibrium_spectrum.json');fold=read(V2/'fold.json')
    eqout=[]
    for direction in (-1,1):
        rows=[r for r in eq if r['direction']==direction]
        branch=[dict(solution_id='equilibrium_fold',branch_id=f'eq_{direction}',continuation_order=0,
            J_exact=fold['g'],mean=fold['r_hz'],stable=direction==-1,stability='critical',residual=fold['residual'])]
        for j,r in enumerate(rows):
            branch.append(dict(solution_id=f'eq_direction{direction}_step{r["step"]}',branch_id=f'eq_{direction}',
                continuation_order=j+1,J_exact=r['g'],mean=r['r_hz'],stable=r['unstable_count']==0,
                stability='stable' if r['unstable_count']==0 else 'unstable',leading_real_per_s=r['leading_real_per_s'],
                unstable_count=r['unstable_count'],residual=r['residual']))
        eqout.append(branch)
    write('equilibrium_branches.json',eqout)
    sys.path.insert(0,str(ROOT/'scripts/topic4_core_bifurcation_v2'))
    from model import System
    s=System();w0,q0=s.weights(1.);w,q=s.weights(1.23)
    mask=np.zeros((6,6),bool);mask[0,0]=mask[1,1]=True
    assert np.array_equal(w[:,~mask],w0[:,~mask]) and np.array_equal(q[~mask],q0[~mask])
    np.testing.assert_allclose(w[:,mask],w0[:,mask]*1.23)
    np.testing.assert_allclose(q[mask],q0[mask]*1.23**2)
    low=eqout[0][1];g=low['J_exact']
    near=[r for r in eq if r['direction']==1 and r['g']<g and r['r_hz'][0]<1]
    start=min(near,key=lambda r:abs(r['g']-g))
    high,err,ok=s.solve(g,np.array(start['r_hz'])/1000)
    assert ok and high[0]*1000>low['mean'][0]
    same_g=dict(J=g,stable_hz=low['mean'],unstable_hz=(high*1000).tolist(),
        difference_AB_hz=(high[:2]*1000-np.array(low['mean'])[:2]).tolist(),unstable_residual=err)
    write('parameter_mapping.json',dict(parameter='J_EE,core',units='dimensionless reference weight multiplier',
        populations=['A E','B E','Surround E','A I','B I','Surround I'],counts=s.count.tolist(),
        matrix_orientation='target rows, source columns',changed_blocks=['A E -> A E','B E -> B E'],
        mean_moment_scale='J',variance_moment_scale='J^2',other_blocks_exactly_unchanged=True,
        native_mask_source=str(ROOT/'.worktrees/topic4-continuous-core-state-r1/src/topic4_core_connectivity_v2.py'),
        reduced_mask_source=str(ROOT/'scripts/topic4_core_bifurcation_v2/model.py'),
        cross_core_direct_W=w0[:,[0,0,1,1],[1,4,0,3]].sum(0).tolist(),
        model='six-population deterministic delay-rate closure; not a spatial SNN state'))
    write('curve_audit.json',dict(status='PASS',mother_sequences=len(out),mother_samples=sum(map(len,out)),
        daughter_sequences=len(daughters),daughter_samples=sum(map(len,daughters)),
        AB_same_solution_ids_and_stability=True,continuation_order_preserved=True,J_sorting=False,
        time_mean='uniform-grid periodic quadrature; equal weight valid for this Fourier collocation',
        dense_extrema_points=32768,max_16384_to_32768_extrema_change_hz=max(r['extrema_grid_error_hz'] for seq in out for r in seq),
        max_original_mean_change_hz=max(r['mean_error_hz'] for r in changes),
        max_original_extrema_change_hz=max(r['extrema_change_hz'] for r in changes),
        B_low_rate_projection_check=same_g,source_sequences=str(V7/'displayed_curve_sequences.json')))

def states():
    source=read(V7/'reduced_condition_coordinates.json');rows=[]
    names=[('a','12a','Two-core self-limited bursts'),('b',None,'PD1: stable 2T'),
        ('c','15a','A bursts / B high background'),('d','15b','Both cores: high background')]
    for label,number,title in names:
        path=(V5/'periodic/mixed_lower_period2/amp0.12_N4096.npz' if number is None
              else Path(next(r['path'] for r in source if r['number']==number)))
        row=orbit(path);row.update(label=label,S='S'+str(len(rows)+1),title=title,stable=True,
            native_key='cd' if label in ('c','d') else label,source_condition=number)
        rows.append(row)
    assert rows[2]['J_exact']==rows[3]['J_exact']==1.38
    write('reduced_states.json',rows)
    # Same mother branch, equally close on the other side of PD1.
    from periodic import Orbit
    from model import System
    flip=np.load(V5/'flips/mixed_lower_flip_N2048.npz');jc=float(flip['g']);g=2*jc-rows[1]['J_exact']
    dest=OUT/'pd1_mother_T_control.npz'
    if not dest.exists():
        r,T,err,hist=Orbit(System(),g,4096).solve(resample(flip['r'],4096,axis=0),float(flip['T']))
        assert err<1e-9
        np.savez_compressed(dest,r=r,T=T,g=g,residual=err,history=hist)
    parent=orbit(dest)
    r=resample(np.load(rows[1]['path'])['r'],32768,axis=0)*1000
    diff=r[len(r)//2:]-r[:len(r)//2]
    peaks,_=find_peaks(r[:,0],prominence=50)
    intervals=np.diff(np.r_[peaks,peaks[:1]+len(r)])*rows[1]['T_full_ms']/len(r)
    write('pd1_comparison.json',dict(mother=parent,daughter=rows[1],critical_J=jc,
        mother_side='stable T side, inherited refined PD1 crossing; new orbit is not independently Floquet reclassified',
        max_half_difference_hz=np.max(abs(diff),axis=0).tolist(),
        rms_half_difference_hz=np.sqrt(np.mean(diff**2,axis=0)).tolist(),
        A_peak_intervals_ms=intervals.tolist(),same_parent=True,
        spatial_propagation_alternation='NOT_ESTABLISHED_BY_AGGREGATE_MODEL'))
    print('REDUCED_READOUTS_COMPLETE',flush=True)

if __name__=='__main__':audit();states()
