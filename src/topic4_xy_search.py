"""Auditable XY-only proposal geometry and explicitly corrected graph bindings."""
from pathlib import Path
import hashlib
import json
import pickle

import numpy as np
from scipy.stats import qmc
from src.topic4_manual_dual_core import budget_matched_dual_core_h
from src.topic4_core_field_runner import connectivity_config as _legacy_connectivity_config
# This module reads already-built corrected caches; it never rebuilds a graph.
# Keep the historical main sampler untouched when validating this study's cache.
PARTNER_SAMPLER_VERSION = "positive_weight_no_autapse_v2"


def connectivity_config(p, theta_deg, ar, *, git_commit=None):
    cfg = _legacy_connectivity_config(p, theta_deg, ar, git_commit=git_commit)
    cfg["partner_sampler_version"] = PARTNER_SAMPLER_VERSION
    for field in ("dt", "w_EE", "w_IE", "tau_m_E", "tau_m_I", "tau_r_AMPA", "tau_r_GABA"):
        cfg[field] = getattr(p, field)
    return cfg



def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def verify_runtime_snapshot(path, root, *, required_paths, config_path, candidate_manifest_path):
    path=Path(path);root=Path(root)
    record=json.loads(path.read_text())
    sources=record['source_hashes']
    required=set(required_paths)|{'src/topic4_xy_search.py','scripts/run_topic4_rev12_node_worker.py'}
    if not required.issubset(sources):
        raise RuntimeError('runtime snapshot omits loaded source modules')
    for relative, expected in sources.items():
        if sha(root/relative)!=expected:
            raise RuntimeError(f'runtime source changed: {relative}')
    for p in (Path(config_path),Path(candidate_manifest_path)):
        if record['input_hashes'].get(str(p.resolve()))!=sha(p):
            raise RuntimeError(f'runtime input changed: {p}')
    return {'path':str(path.resolve()),'sha256':sha(path),'verified':True,
            'identity_kind':'source_hash_snapshot','git_clean_claim':False}


def load_corrected_network(record, params, theta_deg, ar):
    path=Path(record['path'])
    if sha(path)!=record['sha256']:
        raise RuntimeError('corrected graph hash changed')
    with open(path,'rb') as stream:
        payload=pickle.load(stream)
    cfg=payload['config']
    if cfg.get('partner_sampler_version')!=PARTNER_SAMPLER_VERSION:
        raise RuntimeError('graph does not use the corrected partner sampler')
    expected=connectivity_config(params,theta_deg,ar,git_commit=cfg['git_commit'])
    if cfg!=expected:
        raise RuntimeError('corrected graph parameters do not match the simulation')
    return payload['net'],payload['NE'],payload['NI'],True,{
        'path':str(path),'sha256':record['sha256'],'sampler':PARTNER_SAMPLER_VERSION}


def canonical_centers(centers):
    centers=np.asarray(centers,float)
    if centers.shape!=(2,2) or not np.isfinite(centers).all():
        raise ValueError('centers must be finite (2,2)')
    return centers[np.lexsort((centers[:,1],centers[:,0]))]


def field_descriptor(centers, target_count=1499):
    payload={'field_type':'manual_dual_core_budget_matched',
             'centers_mm':canonical_centers(centers).tolist(),'target_count':int(target_count)}
    payload['field_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return payload


def audit_geometry(positions,centers,target_count):
    centers=canonical_centers(centers)
    _,audit=budget_matched_dual_core_h(positions,centers,target_count=target_count)
    radius=audit['distance_cutoff_mm']
    delta=centers[1]-centers[0]
    return {**audit,'minimum_clearance_mm':float(np.minimum(centers,20-centers).min()-radius),
            'center_separation_mm':float(np.linalg.norm(delta)),
            'center_line_angle_deg':float(np.degrees(np.arctan2(delta[1],delta[0]))),
            'full_disks_disjoint':bool(np.linalg.norm(delta)>2*radius)}


def geometry_allowed(centers,positions,*,domain,target_count=1499):
    centers=canonical_centers(centers)
    if np.any((centers<.75)|(centers>19.25)):
        return False
    audit=audit_geometry(positions,centers,target_count)
    if audit['center_separation_mm']<4 or not audit['full_disks_disjoint']:
        return False
    if min(audit['selected_per_core'])<.25*target_count:
        return False
    if domain=='interior' and audit['minimum_clearance_mm']<1.65:
        # Proposal margin includes 0.15 mm tolerance for different sampled positions.
        return False
    if domain not in ('whole_sheet','interior'):
        raise ValueError('unknown geometry domain')
    return True


def sobol_xy_candidates(positions,*,n_per_domain=64,seed=20260905,target_count=1499):
    rows=[];seen=set()
    for offset,domain in enumerate(('whole_sheet','interior')):
        sampler=qmc.Sobol(d=4,scramble=True,seed=seed+offset)
        # Fixed-size draw pool keeps the Sobol sequence balanced and bounds rejection.
        draws=sampler.random_base2(14)
        accepted=0
        for i,draw in enumerate(draws):
            centers=canonical_centers(.75+18.5*draw.reshape(2,2))
            field=field_descriptor(centers,target_count)
            if field['field_sha256'] in seen or not geometry_allowed(centers,positions,domain=domain,target_count=target_count):
                continue
            rows.append({'candidate_id':f'xy_{domain}_{accepted:03d}','domain':domain,
                         'proposal':'sobol_4d','sobol_draw_index':i,'node_field':field,
                         'geometry':audit_geometry(positions,centers,target_count)})
            seen.add(field['field_sha256']);accepted+=1
            if accepted==n_per_domain:break
        if accepted!=n_per_domain:
            raise RuntimeError('not enough valid Sobol proposals')
    return rows


def pareto_indices(values):
    values=np.asarray(values,float)
    return [i for i,v in enumerate(values) if not np.any(np.all(values<=v,axis=1)&np.any(values<v,axis=1))]
