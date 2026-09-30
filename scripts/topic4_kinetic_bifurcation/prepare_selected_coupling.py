"""Lift the selected g40 communication without changing its target averaging.

Density subgroups resolve thresholds inside a spatial cell. They must all
receive that cell's original g40 mean input; reprojecting the native graph
separately for each threshold group defines a different communication model.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/topic4_sef_hfo/kinetic_population_bifurcation_20260916'
SOURCE = ROOT/'results/topic4_sef_hfo/fig5_spatial_kinetic_equivalence_20260916/coarse_40'
GROUPS = ROOT/'results/topic4_sef_hfo/interictal_spatial_population_density_6101_20260916/operators/g40_theta0.25'
DEST = OUT/'operators/selected_g40_theta0.25'


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    g = dict(np.load(GROUPS/'geometry.npz'))
    P = len(g['population']); n = 1600
    pop = g['population']; cell = g['group_cell']; ids = np.arange(P)
    coarse = cell+pop*n
    sizes = np.bincount(coarse, weights=g['group_size'], minlength=2*n)
    lift = sparse.coo_matrix((np.ones(P), (ids, coarse)), shape=(P, 2*n)).tocsr()
    weight_matrices = {}; checks = {}
    prep = json.loads((GROUPS/'prepared.json').read_text())
    depth = prep['max_delay_steps']
    rng = np.random.default_rng(6119)
    history = rng.uniform(0., .01, (depth, P))
    for s, (name, paths) in enumerate((('ampa', ('ee', 'ie')), ('gaba', ('ei', 'ii')))):
        take = pop == s
        restrict = sparse.coo_matrix((g['group_size'][take]/sizes[coarse[take]],
                                      (cell[take], ids[take])), shape=(n, P)).tocsr()
        original = sparse.vstack([sparse.load_npz(SOURCE/f'delay_{key}.npz') for key in paths]).tocsr()
        rows = []; cols = []; values = []
        for d in range(depth):
            part = (lift@original[:, d*n:(d+1)*n]@restrict).tocoo()
            rows.append(part.row); cols.append(part.col+d*P); values.append(part.data)
        W = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(P, P*depth)).tocsr()
        expected = lift@(original@(restrict@history.T).T.reshape(-1))
        actual = W@history.reshape(-1)
        error = float(np.max(abs(expected-actual)))
        assert error < 1e-10
        # Every density subgroup in the same original target cell receives the
        # same current, even when its threshold/core label differs.
        reference = np.zeros(2*n); reference[coarse] = actual
        same_target_error = float(np.max(abs(actual-reference[coarse])))
        assert same_target_error < 1e-11
        checks[name] = dict(arbitrary_delayed_activity_error=error,
                           same_target_cell_input_error=same_target_error, terms=W.nnz)
        sparse.save_npz(DEST/f'delay_{name}.npz', W)
        print(name, checks[name], flush=True)
    (DEST/'geometry.npz').write_bytes((GROUPS/'geometry.npz').read_bytes())
    prep.update(scope='Selected g40 mean communication, exactly lifted to threshold density groups',
                communication_source=str(SOURCE), communication_qa=checks,
                candidate_approximations=['threshold quadrature inside cells',
                    'group-mean frozen Z', 'group-mean dynamic M',
                    'deterministic conditional population limit and density discretization'],
                discarded_subgroup_graph_projection=str(GROUPS))
    (DEST/'prepared.json').write_text(json.dumps(prep, indent=2)+'\n')
    print(DEST, flush=True)


if __name__ == '__main__':
    main()
