#!/usr/bin/env python3
"""Preserve physical source identity and full delay for future dynamic closure."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import shutil
import time
import numpy as np
from scipy import sparse
from campaign import ROOT, read, write, sha
from conditional_density_inputs import OPS
import run_topic4_loop_zk_conditional as native

OUT = ROOT/'dynamic_individual_source_operators'


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json', dict(status='REGISTERED_OPERATOR_PREPARATION_ONLY', created_epoch=time.time(),
        question='Can a future physical-time closure preserve individual source activity and original delays without prescribing the observed22step phase period?',
        motivation='Local and full-target phase-history tests show source identity and relative phase matter. A fixed-period statistical map does not determine autonomous frequency or physical stability; actual exit also needs dynamicG and the originalR5 Kdecay switch.',
        design='One original graph reconstruction, cache exact source/target/delay first and squared-weight operators. Collapse sources to the old3479groups and require exact agreement with the existing target-resolved operators. No simulation, parameter change, new seed or root iteration.',
        next_gate='A future dynamic model must retain source activity in physical time, individual native thresholds and complete initial histories; independently test spatial, phase and transient correspondence before continuation. These operators alone do not validate any white/Gaussian fluctuation closure.',
        no_fixed_phase_period=True, simulations_launched=False, formal_bifurcation_allowed=False,
        producer_sha256=sha(__file__)))
    started = time.time();geo = dict(np.load(OPS/'geometry.npz'));prep = read(OPS/'prepared.json')
    groups = geo['cell_group'];N, P = len(groups), len(geo['group_size']);D = prep['max_delay_steps']
    assert N == 40000 and P == 3479
    write(OUT/'progress.json', dict(status='BUILDING_ORIGINAL_GRAPH', pid=os.getpid(), updated_epoch=time.time()))
    sim, _, _, identity = native.base.old.setup(9108405)
    assert identity == prep['graph_identity']
    assert np.array_equal(sim.net['pos'], geo['original_positions'])
    checks = []
    for kind, offset in [('ampa', 0), ('gaba', 32000)]:
        rr, cc, ww = [], [], []
        rise = getattr(sim.params, 'tau_r_'+kind.upper())
        for delay, matrix in enumerate(sim.net[kind+'_by_delay']):
            if not matrix.nnz:continue
            assert 1 <= delay <= D
            a = matrix.tocoo()
            rr.append(a.row);cc.append(a.col+offset+(delay-1)*N)
            ww.append(a.data/(np.where(a.row < 32000, sim.params.tau_m_E, sim.params.tau_m_I)/rise))
        row, col, weight = np.concatenate(rr), np.concatenate(cc), np.concatenate(ww)
        del rr, cc, ww
        for moment, value in [('mean', weight), ('variance', weight**2)]:
            operator = sparse.coo_matrix((value, (row, col)), shape=(N, N*D)).tocsr()
            assert operator.nnz == len(value), 'Physical edge identities unexpectedly collided'
            projected_col = groups[col % N]+(col//N)*P
            collapsed = sparse.coo_matrix((value, (row, projected_col)), shape=(N, P*D)).tocsr()
            old = sparse.load_npz(ROOT/'target_density_exit'/f'{moment}_{kind}.npz').tocsr()
            difference = collapsed-old
            error = float(abs(difference.data).max()) if difference.nnz else 0.
            assert error < 1e-10, (kind, moment, error)
            sparse.save_npz(OUT/f'{moment}_{kind}.npz', operator)
            checks.append(dict(pathway=kind, moment=moment, physical_edges=operator.nnz,
                source_group_projection_max_abs_error=error))
            write(OUT/'progress.json', dict(status='SAVING_EXACT_SOURCE_DELAY_OPERATORS', pid=os.getpid(),
                completed=checks, updated_epoch=time.time()))
            del operator, collapsed, old, difference, projected_col
        del row, col, weight
    result = dict(status='COMPLETE_OPERATORS_ONLY', graph_identity=identity, physical_sources=N,
        physical_targets=N, original_source_groups=P, max_delay_steps=D, dt_ms=.1, checks=checks,
        source_index='Flattened column=(delay_steps-1)*40000+original_cell_id; GABA source IDs offset by32000.',
        units='Source rate is spikes/ms; weights are native synaptic jump divided by target tau_m/tau_r, matching the frozen density membrane update.',
        variance_limit='Squared physical weights are cached exactly. Treating source fluctuations as independent whiteGaussian processes would remain an additional approximation requiring validation.',
        fixed_phase_period_imposed=False, physical_model_simulated=False, formal_bifurcation_allowed=False,
        elapsed_s=time.time()-started, producer_sha256=sha(__file__))
    write(OUT/'result.json', result);shutil.copy2(__file__, OUT/'producer.py')
    write(OUT/'progress.json', dict(status=result['status'], updated_epoch=time.time()))
    print(result, flush=True)


if __name__ == '__main__':main()
