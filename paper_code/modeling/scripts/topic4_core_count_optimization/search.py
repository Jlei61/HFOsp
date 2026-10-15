"""Delayed feedback BO for variable core count. No confirmation inputs."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import argparse
import pickle
from scipy.stats import qmc
from modeling.scripts.topic4_core_count_optimization.common import *
pass
from modeling.scripts.topic4_core_count_optimization.feedback_schedule import proposal_state

def propose(k, restart, batch):
    verify_sources()
    plan = read(OUT / 'plan.json')
    d = run_dir(k, restart)
    dest = d / 'proposals' / f'batch_{batch:02d}.json'
    if dest.exists():
        return
    dim = 2 * k + 2
    lo, hi = bounds(k)
    seed = plan['random_seed'] + 10000 * k + 1000 * restart + batch
    inp = d / 'proposals' / f'input_{batch:02d}.json'
    if not inp.exists():
        rows = sorted([read(p) for p in (d / 'scores').glob('*.json')], key=lambda row: row['epoch'])
        write(inp, dict(**proposal_state(d, plan, batch, rows), created_unix=time.time()))
    snapshot = read(inp)
    rows = snapshot['rows']
    expected = snapshot['reserved_count']
    batch_size = snapshot['batch_size']
    good = [r for r in rows if r['J'] is not None and np.isfinite(r['J'])]
    pool = qmc.Sobol(dim, scramble=True, seed=seed).random_base2(13)
    pool = np.array([z for z in pool if valid(k, lo + z * (hi - lo))])
    used = (np.asarray([r['vector'] for r in snapshot['reserved_candidates']]) - lo) / (hi - lo)
    selected = []
    roles = []
    detail = {}

    def novel(z):
        all_used = np.vstack([used, np.asarray(selected).reshape(-1, dim)])
        return np.min(np.linalg.norm(all_used - z, axis=1)) > 1e-05 and valid(k, lo + z * (hi - lo))
    if len(good) >= plan.get('feedback_min_scorable', 8):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition.logei import qLogExpectedImprovement
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.optim import optimize_acqf
        from gpytorch.kernels import ScaleKernel, MaternKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        X = torch.tensor((np.asarray([r['vector'] for r in good]) - lo) / (hi - lo), dtype=torch.float64)
        Y = torch.tensor([[-r['J']] for r in good], dtype=torch.float64)
        Yvar = torch.full_like(Y, max(float(Y.var()), 1e-12) * 1e-05)
        model = SingleTaskGP(X, Y, train_Yvar=Yvar, covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dim)), outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model), optimizer_kwargs={'options': {'maxiter': 160}})
        center = X[int(Y.argmax())]
        half = 0.35
        box = torch.stack([(center - half).clamp_min(0), (center + half).clamp_max(1)])
        acq = qLogExpectedImprovement(model, best_f=Y.max(), sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=seed))
        pending = snapshot['pending_candidates']
        if pending:
            acq.set_X_pending(torch.tensor((np.asarray([r['vector'] for r in pending]) - lo) / (hi - lo), dtype=torch.float64))
        z, _ = optimize_acqf(acq, bounds=box, q=min(6, batch_size), num_restarts=8, raw_samples=128, sequential=True, options={'maxiter': 80, 'batch_limit': 2})
        for x in z.detach().numpy():
            if novel(x):
                selected.append(x)
                roles.append('qLogEI')
        with torch.no_grad():
            std = np.concatenate([model.posterior(torch.tensor(p, dtype=torch.float64)).variance[:, 0].sqrt().numpy() for p in np.array_split(pool, 32)])
        for i in np.argsort(-std):
            if len(selected) == batch_size:
                break
            if novel(pool[i]):
                selected.append(pool[i])
                roles.append('global_posterior_uncertainty')
        method = 'Matern52_qLogEI_plus_global_uncertainty'
        detail = dict(numerical_nugget=float(Yvar[0]), note='Fixed-noise numerical regularization, not empirical between-noise variance.')
        torch.save(dict(state_dict=model.state_dict(), X=X, Y=Y, Yvar=Yvar), d / 'proposals' / f'gp_{batch:02d}.pt')
    else:
        method = 'feasibility_space_filling_insufficient_scorable_points'
        for z in pool:
            if len(selected) == batch_size:
                break
            if novel(z):
                selected.append(z)
                roles.append('space_filling')
    if len(selected) != batch_size:
        raise RuntimeError('Insufficient feasible novel proposals')
    ids = []
    for i, z in enumerate(selected):
        cid = f'bo{batch:02d}_{i:02d}'
        ids.append(cid)
        write(d / 'candidates' / f'{cid}.json', candidate(k, restart, lo + z * (hi - lo), cid, expected + i + 1, batch, roles[i]))
    write(dest, dict(ids=ids, batch=batch, method=method, roles=roles, detail=detail, seed=seed, input_sha256=sha(inp), n_scorable=len(good), created_unix=time.time(), batch_size=batch_size, reserved_before=expected, observed_before=len(rows), pending_before=[c['id'] for c in snapshot['pending_candidates']]))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cores', type=int, required=True)
    p.add_argument('--restart', type=int, required=True)
    p.add_argument('--batch', type=int, required=True)
    a = p.parse_args()
    propose(a.cores, a.restart, a.batch)
