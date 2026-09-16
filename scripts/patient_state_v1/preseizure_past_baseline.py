"""Can earlier interval marks account for pre15 counts without a new mark drift?

Uses only completed labels before each pre15 window to estimate one constant
TB fraction. Actual future event counts and clinical types stay conditioned on;
this is not a forecast of total activity, seizure occurrence or seizure type.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.stats import betabinom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scripts.patient_state_v1.common import RUN, write_json
from scripts.patient_state_v1.preseizure_mark_reallocation import statistics

OUT = RUN/'preseizure_past_baseline_v1_39'

def main():
    started = time.time(); OUT.mkdir(exist_ok=True)
    events = pd.read_csv(RUN/'events.csv'); windows = pd.read_csv(RUN/'frozen_seizure_windows.csv'); windows = windows[windows.window == 'pre15'].reset_index(drop=True)
    offsets = np.array([s['offset'] for s in json.loads((RUN/'seizures.json').read_text())]); baseline = []
    for row in windows.itertuples():
        epoch = int(np.searchsorted(offsets, row.start_epoch, side='right'))
        earlier = events[(events.interictal_epoch == epoch) & (events.end_epoch <= row.start_epoch)]
        assert len(earlier) == 0 or earlier.end_epoch.max() <= row.start_epoch
        baseline.append(dict(sz=row.sz, label=row.label, n_earlier=len(earlier), tb_earlier=int(earlier.label_tb.sum()), last_earlier_end=float(earlier.end_epoch.max()) if len(earlier) else None, window_start=row.start_epoch, n_pre=row.n_events, tb_pre=row.n_tb, observed_hours=row.observed_hours))
    frame = pd.DataFrame(baseline); frame.to_csv(OUT/'earlier_only_baselines.csv', index=False)
    n = windows.n_events.to_numpy(); hours = windows.observed_hours.to_numpy(); is_tb = windows.label.to_numpy() == 'TB'; observed = {k:float(v[0]) for k,v in statistics(windows.n_tb.to_numpy()[None, :], n, hours, is_tb).items()}
    summaries = []; predictions = []; rng = np.random.default_rng(1039001)
    for prior in [.5, 1.]:
        alpha = frame.tb_earlier.to_numpy()+prior; beta = frame.n_earlier.to_numpy()-frame.tb_earlier.to_numpy()+prior
        p = rng.beta(alpha, beta, size=(20000, len(frame))); counts = rng.binomial(n, p)
        assert np.all((counts >= 0) & (counts <= n))
        stats = statistics(counts, n, hours, is_tb)
        np.savez_compressed(OUT/f'prior_{prior}.npz', pre15_tb_counts=counts, **stats)
        for measure, values in stats.items():
            lo, median, hi = np.quantile(values, [.025, .5, .975]); extreme = int(np.sum(abs(values) >= abs(observed[measure])-1e-12)) if measure in ['rank_contrast', 'median_rate_difference', 'median_tb_share_difference'] else None
            summaries.append(dict(prior=prior, measure=measure, patient=observed[measure], lower=float(lo), median=float(median), upper=float(hi), n_predictive_sequences=20000, n_abs_at_least_patient=extreme, add_one_abs_tail=(extreme+1)/20001 if extreme is not None else None))
        for i, row in enumerate(frame.itertuples()):
            expected = float(n[i]*alpha[i]/(alpha[i]+beta[i])); lo = int(betabinom.ppf(.025, n[i], alpha[i], beta[i])) if n[i] else 0; hi = int(betabinom.ppf(.975, n[i], alpha[i], beta[i])) if n[i] else 0
            predictions.append(dict(prior=prior, sz=row.sz, label=row.label, n_earlier=row.n_earlier, tb_earlier=row.tb_earlier, earlier_posterior_mean_tb=alpha[i]/(alpha[i]+beta[i]), n_pre=row.n_pre, patient_tb_count=row.tb_pre, predictive_mean_tb_count=expected, predictive_count_lower=lo, predictive_count_upper=hi, patient_tb_rate=row.tb_pre/row.observed_hours, predictive_mean_tb_rate=expected/row.observed_hours, predictive_rate_lower=lo/row.observed_hours, predictive_rate_upper=hi/row.observed_hours))
    table = pd.DataFrame(summaries); table.to_csv(OUT/'group_comparison.csv', index=False); detail = pd.DataFrame(predictions); detail.to_csv(OUT/'individual_predictions.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    chosen = detail[detail.prior == .5].reset_index(drop=True)
    for i, row in enumerate(chosen.itertuples()):
        color = '#296dac' if row.label == 'TB' else '#bb4050'
        axes[0].errorbar(i, row.predictive_mean_tb_rate, yerr=[[row.predictive_mean_tb_rate-row.predictive_rate_lower], [row.predictive_rate_upper-row.predictive_mean_tb_rate]], fmt='o', color=color, capsize=3)
        axes[0].scatter(i, row.patient_tb_rate, marker='x', color='black', s=28, zorder=4)
    axes[0].set(xticks=range(12), xticklabels=[f'SZ{s}' for s in chosen.sz], ylabel='Pre15 TB events / observed hour', title='Earlier-only composition; actual pre15 total counts fixed')
    axes[0].tick_params(axis='x', rotation=50)
    axes[1].bar(range(12), frame.n_earlier, color=['#296dac' if x == 'TB' else '#bb4050' for x in frame.label])
    axes[1].set(yscale='symlog', xticks=range(12), xticklabels=[f'SZ{s}' for s in frame.sz], ylabel='Completed earlier marks in same interval', title='How much earlier information is available?')
    axes[1].tick_params(axis='x', rotation=50)
    axes[1].legend(handles=[Patch(color='#bb4050', label='TA-type seizure'), Patch(color='#296dac', label='TB-type seizure')], loc='upper left', fontsize=8)
    for ax in axes: ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Do pre15 marks require a new change from the earlier interval baseline?\nBeta(0.5,0.5) prior; dots and 95% ranges: conditional prediction; black crosses: patient', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .88])
    for extension in ['png', 'pdf']: fig.savefig(RUN/'figures'/f'preseizure_earlier_mode_baseline.{extension}', dpi=180)
    plt.close(fig)
    write_json(OUT/'scientific_audit.json', dict(status='COMPLETE', n_seizures=12, n_tb_seizures=2, n_without_earlier_marks=int((frame.n_earlier == 0).sum()), n_with_fewer_than10_earlier_marks=int((frame.n_earlier < 10).sum()), prior='Symmetric Beta(0.5,0.5) primary, Beta(1,1) sensitivity; no hyperparameter fitting', conditioned='Actual future total event count and coverage in each original pre15 window; clinical type unused in baseline estimation', temporal_audit='Only completed event windows ending at or before pre15 start in the same clinical interval; no pre15 labels enter the baseline', uncertainty='Exact beta-binomial per-window predictive count intervals;20000replicates for group summaries under each prior', interpretation='Checks whether pre15 mark counts require a departure from a simple earlier composition once their actual activity is given. It cannot establish a correct dynamical model, nor forecast the supplied future total activity or clinical outcome.', limits='Several intervals have no or few earlier marks. Fixed full-record discovery labels still make this development evidence. Symmetric priors and independent per-interval constant probabilities are explicit assumptions; short-time mark ordering is not modeled.', elapsed=time.time()-started))
    readme = RUN/'figures/README.md'
    if '### preseizure_earlier_mode_baseline.png' not in readme.read_text():
        with readme.open('a') as handle:
            handle.write('\n### preseizure_earlier_mode_baseline.png\n只用每个pre15窗口开始前、同一临床间期段内已经完成的事件标签估计固定TB比例，再给定真实pre15总事件数得到精确Beta-binomial计数区间。左侧黑叉为真实TB率、圆点及范围为条件预测；右侧列出各例实际可用的更早事件数，显示信息不足的窗口。\n**关注点**：发作前标记数是否必须偏离较早模式背景；已给定未来总活动，不能称为完整事件率、发作时间或类型的前瞻预测。\n')
    print(detail[detail.label == 'TB'].to_string(index=False)); print(table[table.measure.isin(['rank_contrast', 'median_rate_difference', 'median_tb_share_difference'])].to_string(index=False))

if __name__ == '__main__':
    main()
