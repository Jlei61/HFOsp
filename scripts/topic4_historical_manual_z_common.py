"""Historical manually placed field on the current audited fast SNN carrier."""
from validate_topic4_fixed_rate_base import ROOT, SOURCE, setup as carrier_setup, read, write
from src.topic4_core_field_runner import _placement
from src.sef_hfo_heterogeneity import sample_core_field
from src.topic4_core_field import manual_mask, axis_coords, arm_h, sample_core_quantiles, core_thresholds, signed_depth, build_vth
from src.topic4_graph_edge_flow import array_sha256
import numpy as np
import copy
import os

ARM = os.environ.get('TOPIC4_MANUAL_ARM', 'manual_hard')
assert ARM in ('manual_hard', 'manual_smooth')
OUT = ROOT / 'results/topic4_sef_hfo' / f'historical_{ARM}_native_z_v1'
TIMES_MS = [8000, 9400, 9800, 10180, 10680]


def setup(seed):
    s, tr, frozen, identity = carrier_setup(seed)
    c = read(ROOT / 'results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json')
    reg = _placement(c); e = c['engine']; ne = s.net['NE']; n = ne + s.net['NI']
    centers = np.array([reg['source_centroid'], reg['sink_centroid']])
    is_e = np.arange(n) < ne
    mask = manual_mask(s.positions_e, *centers, e['core_r'])
    if ARM == 'manual_hard':
        fields = [sample_core_field(s.net['pos'], is_e, xy, e['core_r'], np.random.default_rng(6101 + 7 + k),
                  core_mean=e['core_mean'], core_std=e['core_std'], base_mean=e['v_base'])['vth'] for k, xy in enumerate(centers)]
        vth = np.minimum(*fields); h = mask.astype(float)
        definition = 'Exact historical manual_hard field producer: separate seed+7/+8 draws, minimum of two core fields; above-background draws clipped by this historical construction.'
    else:
        axis = centers[1] - centers[0]; axis /= np.linalg.norm(axis)
        u, v = axis_coords(s.positions_e, reg['center'], axis)
        geom = dict(sep=float(np.linalg.norm(centers[1] - centers[0])),
                    s_support=(float(u.min()) + c['field']['AXIAL_MARGIN'], float(u.max()) - c['field']['AXIAL_MARGIN']),
                    M=c['field']['M'], sigma_perp=e['core_r'], shift_mm=c['field']['SHIFT_MM'])
        h = arm_h(ARM, u, v, geom, float(c['N_core_manual']), manual_mask_E=mask)
        d = signed_depth(core_thresholds(sample_core_quantiles(ne, c['quantile_seed']), e['core_mean'], e['core_std']), e['v_base'])
        vth = build_vth(h, d, n_total=n, n_E=ne, v_base=e['v_base'])
        definition = 'Exact historical manual_smooth h and signed-depth threshold mapping from the stage configuration.'
    assert vth.shape == (n,) and np.isfinite(vth).all()
    s.vtheta = vth; s.h_e = h; s.delta_vtheta = vth[:ne] - e['v_base']
    changed = copy.deepcopy(frozen)
    changed['candidate']['candidate_id'] = ARM + '_field_on_C_fast_carrier'
    changed['candidate']['node_field'] = {'field_type': ARM, 'centers_mm': centers.tolist(), 'core_radius_mm': e['core_r']}
    for name, val in [('h_sha256', h), ('delta_vtheta_sha256', s.delta_vtheta), ('vtheta_sha256', vth)]:
        identity[name] = array_sha256(np.asarray(val, np.float32))
    record = {'arm': ARM, 'centers_mm': centers.tolist(), 'historical_core_radius_mm': e['core_r'],
              'stage_config': str(ROOT / 'results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json'),
              'field_definition': definition, 'field_seed': 6101, 'dynamics_seed': seed,
              'scope': 'Historical manual node field replaces searched field. Current C fast connectivity, synaptic parameters and native OU remain. This is not a full reproduction of every historical engine setting.',
              'old_node_mapping_applied_to_manual': False, 'geometric_core_E_count': int(mask.sum()),
              'threshold_range_E_mv': [float(vth[:ne].min()), float(vth[:ne].max())],
              'current_dynamic_parameters': frozen['candidate']['dynamic_parameters'], 'identity': identity}
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / 'substrate.json').exists():
        write(OUT / 'substrate.json', record)
        np.savez_compressed(OUT / 'substrate.npz', positions_e=s.positions_e, h_e=h, vtheta=vth, centers_mm=centers)
        dist = np.linalg.norm(s.positions_e[:, None] - centers[None], axis=2)
        groups = np.full(ne, 2); groups[dist[:, 0] < 1.75] = 0
        groups[(dist[:, 1] < 1.75) & (dist[:, 1] < dist[:, 0])] = 1
        rng = np.random.default_rng(202609091)
        samples = np.r_[*[rng.choice(np.flatnonzero(groups == k), size=size, replace=False) for k, size in enumerate((60, 60, 120))], rng.choice(np.arange(ne, n), size=60, replace=False)]
        folder = OUT / 'reference_samples'; (folder / 'runs').mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'runs/z_current_e_seed9108401.npz', sample_ids=samples,
                            sample_groups=np.repeat(np.arange(4), [60, 60, 120, 60]))
        qa = read(ROOT / 'results/topic4_sef_hfo/snn_raster_inhibition_transition_v1/engine_qa.json')
        assert qa['status'] == 'PASS'; write(folder / 'engine_qa.json', qa)
    return s, tr, changed, identity
