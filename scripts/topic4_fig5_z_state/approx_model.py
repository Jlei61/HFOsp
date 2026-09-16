#!/usr/bin/env python3
"""Build the coarse (20x20 main, 10x10 control) reduction of THIS network (manual_hard field on the C fast carrier).

The pathway first/second moments, per-delay-bin operators and per-cell empirical threshold support are projected
from the realized SNN graph and threshold field that the reference runs used (identity checked).  Nothing here
touches SNN biology; it is the mathematical projection defined in reduction_definition.md.
"""
import argparse
import resource
from scipy import sparse
from common import *  # noqa: F401,F403
from src.topic4_patient_zm_meanfield import build_patient_coarse_model, save_patient_coarse_model, spatial_cell_index
from src.topic4_dual_core_spatial_z_delay import build_coarse_delay_operators

APPROX = OUT / 'approx'


def build(grid):
    started = time.time()
    s, tr, frozen, identity = old.setup(MAIN_SEED)
    assert identity == reference_protocol()['identity'], 'coarse model must be built from the reference network'
    folder = APPROX / f'coarse_{grid}'; folder.mkdir(parents=True, exist_ok=True)
    model = build_patient_coarse_model(s, n_grid=grid, threshold_groups=8)
    save_patient_coarse_model(folder / 'model.npz', model)
    ops = build_coarse_delay_operators(s, model)
    for key in ('ee', 'ei', 'ie', 'ii'):
        sparse.save_npz(folder / f'delay_{key}.npz', getattr(ops, f'w_{key}_history'))
    cell_e = spatial_cell_index(s.positions_e, n_grid=grid, sheet_l_mm=s.params.L)
    cell_i = spatial_cell_index(s.positions_i, n_grid=grid, sheet_l_mm=s.params.L)
    centers = np.asarray(read(SUBSTRATE / 'substrate.json')['centers_mm'])
    d = np.linalg.norm(s.positions_e[:, None] - centers[None], axis=2)
    g15 = np.full(s.n_e, 2); g15[d[:, 0] < 1.5] = 0; g15[(d[:, 1] < 1.5) & (d[:, 1] < d[:, 0])] = 1
    g175 = np.full(s.n_e, 2); g175[d[:, 0] < 1.75] = 0; g175[(d[:, 1] < 1.75) & (d[:, 1] < d[:, 0])] = 1
    np.savez_compressed(folder / 'geometry.npz', cell_e=cell_e, cell_i=cell_i, positions_e=s.positions_e, positions_i=s.positions_i,
                        vtheta_e=s.vtheta[:s.n_e], vtheta_i=s.vtheta[s.n_e:], h_e=s.h_e, centers_mm=centers, g15=g15, g175=g175,
                        count_e=model.count_e, count_i=model.count_i)
    params = {k: v for k, v in vars(s.params).items() if isinstance(v, (int, float, str, bool))}
    from params import compute_nu_theta
    write(folder / 'prepared.json', dict(dt_ms=ops.dt_ms, max_delay_steps=ops.max_delay_steps, grid=grid, threshold_groups=8,
          graph_identity=identity, tau_r_ampa_ms=s.params.tau_r_AMPA, tau_r_gaba_ms=s.params.tau_r_GABA,
          tau_d_ampa_ms=s.params.tau_d_AMPA, tau_d_gaba_ms=s.params.tau_d_GABA, params=params,
          nu_theta_per_ms=float(compute_nu_theta(s.params)[0]), nu_ext_per_ms=float(model.nu_ext_per_ms),
          spatial_ou=tr['spatial_ou'], seconds=time.time() - started,
          peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
          scope='Projection of the realized reference graph; pathway means/variances per target cell, exact 0.1-ms delay bins, 8-chunk threshold support per cell.'))
    print('built', grid, time.time() - started, flush=True)




def build_variance_delay(grid):
    """Per-delay-bin second-moment operators (sum of squared weights per target neuron), for the kernel-filtered variance."""
    from src.topic4_patient_zm_meanfield import _aggregate_pathway
    started = time.time()
    s, tr, frozen, identity = old.setup(MAIN_SEED)
    assert identity == reference_protocol()['identity']
    folder = APPROX / f'coarse_{grid}'
    m = np.load(folder / 'model.npz'); count_e = m['count_e'].astype(float); count_i = m['count_i'].astype(float)
    n_e, n_i = s.n_e, s.n_i; n = grid * grid
    cell_e = spatial_cell_index(s.positions_e, n_grid=grid, sheet_l_mm=s.params.L)
    cell_i = spatial_cell_index(s.positions_i, n_grid=grid, sheet_l_mm=s.params.L)
    all_cells = np.r_[cell_e, cell_i]; maximum = int(s.net['max_delay_steps'])
    ampa = s.net['ampa_by_delay']; gaba = s.net['gaba_by_delay']
    spec = dict(ee=(ampa, cell_e, lambda rows: rows < n_e, count_e, s.params.tau_r_AMPA / s.params.tau_m_E),
                ie=(ampa, cell_e, lambda rows: rows >= n_e, count_i, s.params.tau_r_AMPA / s.params.tau_m_I),
                ei=(gaba, cell_i, lambda rows: rows < n_e, count_e, s.params.tau_r_GABA / s.params.tau_m_E),
                ii=(gaba, cell_i, lambda rows: rows >= n_e, count_i, s.params.tau_r_GABA / s.params.tau_m_I))
    for name, (mats, src, mask, counts, factor) in spec.items():
        blocks = []
        for step in range(1, maximum + 1):
            _, sq = _aggregate_pathway((mats[step],), target_cells=all_cells, source_cells=src, target_mask=mask,
                                       n_cells=n, target_counts=counts, physical_factor=factor)
            blocks.append(sparse.csr_matrix(sq))
        op = sparse.hstack(blocks, format='csr')
        total = np.zeros((n, n))
        for d in range(maximum):
            total += op[:, d * n:(d + 1) * n].toarray()
        assert np.allclose(total, m[f'v_{name}'], rtol=1e-11, atol=1e-13), name
        sparse.save_npz(folder / f'vdelay_{name}.npz', op)
    write(folder / 'variance_delay_prepared.json', dict(status='COMPLETE', seconds=time.time() - started,
          note='Squared-weight operators per exact delay bin; block sums equal model.v_* (checked).'))
    print('variance delay built', grid, time.time() - started, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--grid', type=int, nargs='+', default=[20, 10])
    ap.add_argument('--variance', action='store_true'); a = ap.parse_args()
    for g in a.grid:
        if a.variance:
            build_variance_delay(g)
        else:
            build(g)
