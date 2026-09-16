"""Explain event-sampling bias at frozen joint-model parameter points.

For slowly varying latent intensity, the effective shifted-renewal rate is
lambda/(1+deadtime*lambda). This quasi-static approximation is compared with
completed 0.1-second-step stochastic trajectories, not asserted to be exact
for a dynamic point process or the original event-packing algorithm.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN, write_json

OUT = RUN/'stationary_event_sampling_v1_37'

def quadrature(config, order):
    nodes, w = hermgauss(order); w /= np.sqrt(np.pi)
    s = np.sqrt(2)*config['sd_s']*nodes
    rate = np.exp(config['a']+np.sqrt(2)*config['sd_r']*nodes[:, None]+config['c']*s[None, :])
    effective = rate/(1+rate*config['deadtime']/3600)
    weights = w[:, None]*w[None, :]
    total = np.sum(weights*effective)
    probability = expit(config['b']+s)
    return dict(rate_per_hour=float(total), uniform_time_tb_probability=float(np.sum(w*probability)), event_weighted_tb_share=float(np.sum(weights*effective*probability[None, :])/total))

def main():
    OUT.mkdir(exist_ok=True); rows = []; configs = {}; checks = []
    for name in ['independent', 'coupled']:
        source = RUN/f'joint_two_state_generation_v1_20/runs/joint_two_state_{name}_000.json'
        config = json.loads(source.read_text())['job']['config']; configs[name] = config
        coarse, fine = quadrature(config, 96), quadrature(config, 192)
        error = max(abs(coarse[k]-fine[k])/max(abs(fine[k]), 1e-8) for k in fine)
        assert error < 2e-4
        runs = [json.loads(p.read_text()) for p in sorted((RUN/'joint_generation_resolution_v1_31/runs').glob(f'simulation_step_{name}_*.json'))]
        assert len(runs) == 64 and all(r['status'] == 'COMPLETE' for r in runs)
        n = np.array([r['n_events'] for r in runs]); tb = np.rint(n*np.array([r['tb_fraction'] for r in runs])).astype(np.int64)
        rates = np.array([r['rate_per_hour'] for r in runs]); rng = np.random.default_rng(1037001); index = rng.integers(0, len(n), (5000, len(n)))
        share_boot = tb[index].sum(1)/n[index].sum(1); rate_boot = rates[index].mean(1)
        row = dict(model=name, **fine, simulation_event_weighted_tb_share=float(tb.sum()/n.sum()), simulation_mean_rate=float(rates.mean()), simulation_share_mean_lower=float(np.quantile(share_boot, .025)), simulation_share_mean_upper=float(np.quantile(share_boot, .975)), simulation_rate_mean_lower=float(np.quantile(rate_boot, .025)), simulation_rate_mean_upper=float(np.quantile(rate_boot, .975)), quadrature_relative_error=error, n_existing_sequences=64)
        rows.append(row); checks.append(dict(model=name, relative_error_96_vs192=error))
    independent = rows[0]
    assert abs(independent['uniform_time_tb_probability']-independent['event_weighted_tb_share']) < 1e-12
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv', index=False)
    config = configs['coupled']; theory = rows[1]
    x = np.linspace(-6*config['sd_s'], 6*config['sd_s'], 801)
    nodes, weights = hermgauss(192); weights /= np.sqrt(np.pi)
    rates = np.exp(config['a']+np.sqrt(2)*config['sd_r']*nodes[:, None]+config['c']*x[None, :])
    mean_rate = np.sum(weights[:, None]*rates/(1+rates*config['deadtime']/3600), axis=0)
    density_time = norm.pdf(x, scale=config['sd_s']); density_event = density_time*mean_rate/theory['rate_per_hour']
    assert abs(np.trapz(density_event, x)-1) < 1e-6
    pd.DataFrame(dict(state_residual=x, density_uniform_time=density_time, density_at_event=density_event, conditional_tb_probability=expit(config['b']+x), effective_rate_per_hour=mean_rate)).to_csv(OUT/'state_sampling_curve.csv', index=False)
    patient = json.loads((RUN/'data_audit.json').read_text()); patient_share = patient['n_tb']/patient['n_interictal_events']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(x, density_time, label='Uniform time', color='#688baa'); axes[0].plot(x, density_event, label='At an event', color='#c87542')
    axes[0].set(xlabel='State residual x (log-odds)', ylabel='Density', title='State sampled by events'); axes[0].legend(fontsize=8)
    axes[1].plot(x, expit(config['b']+x), color='#8553a0'); axes[1].set(xlabel='State residual x (log-odds)', ylabel='TB probability', title='Same mode readout', ylim=(0, 1))
    values = [theory['uniform_time_tb_probability'], theory['event_weighted_tb_share'], theory['simulation_event_weighted_tb_share'], patient_share]
    axes[2].bar(range(4), values, color=['#688baa', '#c87542', '#dbab87', '#444444'])
    for i, value in enumerate(values): axes[2].text(i, value+.018, f'{value:.3f}', ha='center', fontsize=8)
    axes[2].set(xticks=range(4), xticklabels=['Time\nmodel', 'Event\ntheory', 'Event\nsimulated', 'Event\npatient'], ylim=(0, .65), ylabel='TB probability / share', title='Why the generated mixture shifts')
    for ax in axes: ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('State-dependent event rates change the sampled mode mixture\nFrozen coupled-model parameters; quasi-static calculation checked against existing 0.1-s simulations', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .89])
    for extension in ['png', 'pdf']: fig.savefig(RUN/'figures'/f'state_dependent_event_sampling.{extension}', dpi=180)
    plt.close(fig)
    write_json(OUT/'scientific_audit.json', dict(status='COMPLETE', results=rows, numerical_checks=checks, independent_observation_factorization_pass=True, n_new_fits=0, n_new_simulations=0, formula='P(TB|event)=E[lambda_eff(s,r)*sigmoid(b+s)]/E[lambda_eff(s,r)], lambda_eff=lambda/(1+delta*lambda); time probability=E[sigmoid(b+s)]', assumptions='Independent stationary OU marginals at frozen parameters; state approximately constant over the0.25s effective deadtime. This is not an exact dynamic-renewal theorem or a biological refractory claim.', interpretation='Rate-based event sampling can tilt the mode mixture even if state occupancy and logistic readout are unchanged. This diagnoses the existing parameter point; it does not prove the whole joint model family cannot fit the patient, nor identify a physiological force.', actual_patient_uniform_time_probability='NOT_OBSERVED', scope='Analytical readout audit plus already completed simulations, not another fitted model or independent patient validation'))
    readme = RUN/'figures/README.md'
    if '### state_dependent_event_sampling.png' not in readme.read_text():
        with readme.open('a') as handle:
            handle.write('\n### state_dependent_event_sampling.png\n在已冻结的联合模型参数处，计算均匀时刻与事件时刻采样到的状态分布，以及同一logistic读出如何产生不同TB份额。使用慢状态相对0.25秒有效间隔的准静态近似，并与已完成的64条0.1秒步长轨迹比较；患者仅有事件份额，不赋予其未观测的均匀时刻概率。\n**关注点**：模式状态到事件率的映射会改变事件混合比例；这是现有参数点的观察层解释，不是已经识别的生理drift或模型族不可能拟合的证明。\n')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == '__main__':
    main()
