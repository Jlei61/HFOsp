"""Reselect the frozen H pool by three-component training J and redraw its KDE."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/honglab/leijiaxin/HFOsp')
ARCHIVE = ROOT/'results/paper-ready-figure/archive/2026-10-11_fig4_finalization/fig4/candidates/core_position_densification_complete_20261009/source/scores'
GROUPS = ('rank_pattern', 'local_and_interrod_timing', 'participation')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_function(filename, name):
    path = HERE/'h_density_producers'/filename
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name==name)
    env = {'np':np, 'SIGMA':.35}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env[name]


def main():
    snapshot = read(HERE/'h_pool_snapshot.json')
    old_selection = read(HERE/'D_position_selection.json')
    assert sha(HERE/'h_pool_snapshot.json')==old_selection['source_snapshot_sha256']
    rows = snapshot['records']
    valid = [r for r in rows if r.get('J') is not None and np.isfinite(r['J'])]
    assert len(rows)==418 and len(valid)==413
    spec = read(HERE/'i_j_frozen_spec.json')
    score_dir = HERE/'h_original_scores'; score_dir.mkdir(exist_ok=True)
    score_errors = []
    for row in valid:
        path = score_dir/(row['candidate']+'.json')
        if not path.exists():
            shutil.copy2(ARCHIVE/path.name, path)
        assert sha(path)==row['score_sha256'],row['candidate']
        score = read(path)
        assert score['objective_sha256']==row['objective_sha256']==spec['original_pickle_sha256']
        assert score['status']=='SCORABLE' and score['N']==row['N'] and row['N']>=16
        components = [score['groups'][g]['D_off']/spec['scales'][g] for g in GROUPS]
        np.testing.assert_allclose(components, [row['groups'][g]['scaled'] for g in GROUPS],rtol=0,atol=1e-12)
        np.testing.assert_allclose(np.mean(components),row['J'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(score['J'],row['J'],rtol=0,atol=1e-12)
        score_errors.append(abs(np.mean(components)-row['J']))
    lookup = {r['candidate']:r for r in valid}
    for filename, total_key in [('g_training_loss_profiles.csv','J'),('i_angle_response.csv','optimizer_J')]:
        for row in csv.DictReader((HERE/filename).open()):
            np.testing.assert_allclose(lookup[row['candidate']]['J'],float(row[total_key]),rtol=0,atol=1e-12)
    ordered = sorted(valid,key=lambda r:(r['J'],r['candidate']))
    nominal = math.ceil(.2*len(ordered)); cutoff = ordered[nominal-1]['J']
    selected = [r for r in ordered if r['J']<=cutoff]
    ids = {r['candidate'] for r in selected}
    old_ids = set(old_selection['selection']['candidate_ids'])
    assert len(selected)==83
    with np.load(HERE/'D_position_density.npz') as z:
        old_arrays = dict(z)
    all_xy = np.array([r['centers_mm'] for r in rows])
    np.testing.assert_array_equal(all_xy,old_arrays['all_positions_mm'])
    np.testing.assert_array_equal([r['candidate'] for r in rows],old_arrays['candidate_ids'])
    coords = old_arrays['x_mm']; yy_coords = old_arrays['y_mm']
    xx,yy = np.meshgrid(coords,yy_coords)
    cell_area = float(np.diff(coords)[0]*np.diff(yy_coords)[0]); sigma=.35
    density = frozen_function('build_fig4_joint_core_density.py','density')
    mass_levels = frozen_function('plot_fig4_core_position_search_context.py','mass_levels')
    # Verify the unchanged density algorithm against the previously displayed H.
    old_rebuilt = np.array([density(xx,yy,old_arrays['selected_positions_mm'][:,core],sigma=sigma) for core in range(2)])
    np.testing.assert_allclose(old_rebuilt,old_arrays['per_core_density_per_mm2'],rtol=0,atol=1e-12)
    mask = np.array([r['candidate'] in ids for r in rows])
    points = all_xy[mask]
    parts = np.array([density(xx,yy,points[:,core],sigma=sigma) for core in range(2)])
    combined = parts.mean(0)
    np.testing.assert_allclose(parts.sum((1,2))*cell_area,[1,1],rtol=0,atol=2e-6)
    np.savez_compressed(HERE/'H_position_density.npz',x_mm=coords,y_mm=yy_coords,
        density_per_mm2=combined,per_core_density_per_mm2=parts,all_positions_mm=all_xy,
        selected_positions_mm=points,selected_mask=mask,candidate_ids=old_arrays['candidate_ids'])
    contours={};peaks=[];old_peaks=[];isolated=[]
    for core in range(2):
        threshold=mass_levels(parts[core],cell_area)['0.9']
        fig,ax=plt.subplots();curve=ax.contour(xx,yy,parts[core],levels=[threshold])
        paths=[v.tolist() for v in curve.allsegs[0] if len(v)>2];plt.close(fig)
        enclosed=float(parts[core][parts[core]>=threshold].sum()*cell_area)
        assert abs(enclosed-.9)<.004
        contours['AB'[core]]={'0.9':dict(threshold_per_mm2=threshold,enclosed_mass=enclosed,paths_mm=paths)}
        iy,ix=np.unravel_index(np.argmax(parts[core]),parts[core].shape)
        peaks.append([float(coords[ix]),float(yy_coords[iy])])
        iy,ix=np.unravel_index(np.argmax(old_rebuilt[core]),old_rebuilt[core].shape)
        old_peaks.append([float(coords[ix]),float(yy_coords[iy])])
        at_points=density(points[:,core,0],points[:,core,1],points[:,core],sigma=sigma)
        isolated.extend(points[at_points<threshold,core].tolist())
    write(HERE/'H_position_contours.json',dict(method='Per-core highest-density mass contours; not confidence intervals.',
          sigma_mm=sigma,selection_metric='J',contours=contours))
    selection=dict(fraction=.2,nominal_count=nominal,count=len(selected),cutoff_J=cutoff,
                   candidate_ids=[r['candidate'] for r in selected],include_exact_cutoff_ties=True)
    meta=dict(selection=selection,pool_scorable=len(valid),pool_completed=len(rows),asof=snapshot['asof'],
        sigma_mm=sigma,density_integral=float(combined.sum()*cell_area),selection_metric='J',
        formula='J = (L_r + L_o + L_p) / 3',objective_sha256=spec['original_pickle_sha256'],
        source_pool='source/h_pool_snapshot.json',source_snapshot_sha256=sha(HERE/'h_pool_snapshot.json'),
        selection_method='Rank all 413 compatible finite J values; take lowest ceil(0.2*N), include exact cutoff ties; no position-based filtering.',
        unit='mm^-2',ground_peaks_mm=peaks,previous_ground_peaks_mm=old_peaks,
        retained_from_previous=len(ids&old_ids),added_candidates=sorted(ids-old_ids),removed_candidates=sorted(old_ids-ids),
        isolated_selected_positions_mm=np.unique(np.array(isolated).reshape(-1,2),axis=0).tolist(),
        interpretation='Descriptive position density of the lowest 20% training J within the same evaluated pool. Historical proposal objectives and deliberate densification are unchanged; not a posterior or independent recovery test.',
        geometry_and_kernel_unchanged=True,no_new_simulations=True)
    write(HERE/'H_position_selection.json',meta)
    with (HERE/'H_pool_training_loss.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['candidate','N','J','L_r','L_o','L_p','J_joint_historical','selected_by_J','selected_by_J_joint_historical','core_A_x_mm','core_A_y_mm','core_B_x_mm','core_B_y_mm'])
        writer.writeheader()
        for row in rows:
            groups=row.get('groups',{})
            writer.writerow(dict(candidate=row['candidate'],N=row['N'],J=row['J'],
                **{'L_'+k:groups.get(g,{}).get('scaled') for k,g in zip(('r','o','p'),GROUPS)},
                J_joint_historical=row.get('J_joint'),selected_by_J=row['candidate'] in ids,
                selected_by_J_joint_historical=row['candidate'] in old_ids,
                **dict(zip(('core_A_x_mm','core_A_y_mm','core_B_x_mm','core_B_y_mm'),np.ravel(row['centers_mm'])))))
    report=dict(status='PASS',same_418_configuration_pool=True,scorable=413,selected=83,
        frozen_score_files_verified=413,same_objective_as_C_G_I=True,all_G_and_I_J_values_match=True,
        maximum_J_reconstruction_error=float(max(score_errors)),unchanged_KDE_reproduces_previous_density=True,
        core_density_integrals=(parts.sum((1,2))*cell_area).tolist(),combined_density_integral=meta['density_integral'],
        contour_masses={k:v['0.9']['enclosed_mass'] for k,v in contours.items()},
        new_ground_peaks_mm=peaks,old_ground_peaks_mm=old_peaks,retained=meta['retained_from_previous'],
        replaced=len(ids-old_ids),selection_cutoff_J=cutoff,source_producers={p.name:sha(p) for p in (HERE/'h_density_producers').glob('*.py')})
    write(HERE/'h_loss_density_validation.json',report)
    print(report)


if __name__=='__main__':
    main()
