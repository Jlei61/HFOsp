"""Separate total activity, interval preferences and local ordering in the quoted association.

Exact conditional count reallocation holds every event time, clinical type,
window and exposure fixed. It is a specificity diagnostic for the existing
exploratory association, not another clinical validation or a dynamical model.
"""
import sys, json, itertools, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN, write_json
from scripts.explore_e1146_seizure_interictal_association import exact_rank_test

OUT = RUN/'preseizure_mark_reallocation_v1_38'

def statistics(counts, n, hours, is_tb):
    rates = counts/hours
    ranks = rankdata(rates, axis=1)
    assert is_tb.sum() == 2 and (~is_tb).sum() == 10
    contrast = (ranks[:, is_tb].sum(1)-13)/10
    pairs = np.array(list(itertools.combinations(range(12), 2)))
    null = (ranks[:, pairs].sum(2)-13)/10
    p_original_test = np.mean(np.abs(null) >= np.abs(contrast[:, None])-1e-12, axis=1)
    eligible = n > 0
    shares = counts[:, eligible]/n[eligible]
    return dict(rank_contrast=contrast, median_tb_rate_tb_seizures=np.median(rates[:, is_tb], axis=1), median_tb_rate_ta_seizures=np.median(rates[:, ~is_tb], axis=1), median_rate_difference=np.median(rates[:, is_tb], axis=1)-np.median(rates[:, ~is_tb], axis=1), median_tb_share_difference=np.median(shares[:, is_tb[eligible]], axis=1)-np.median(shares[:, ~is_tb[eligible]], axis=1), original_unstratified_p=p_original_test)

def main():
    started = time.time(); OUT.mkdir(exist_ok=True)
    events = pd.read_csv(RUN/'events.csv')
    windows = pd.read_csv(RUN/'frozen_seizure_windows.csv'); windows = windows[windows.window == 'pre15'].reset_index(drop=True)
    assert len(windows) == 12
    membership = np.full(len(events), 12, dtype=np.int64)
    for i, row in enumerate(windows.itertuples()):
        keep = (events.start_epoch >= row.start_epoch) & (events.end_epoch <= row.end_epoch) & (events.start_epoch < row.end_epoch)
        assert np.all(membership[keep] == 12)
        assert int(keep.sum()) == row.n_events and int(events.loc[keep, 'label_tb'].sum()) == row.n_tb
        membership[keep] = i
    n = windows.n_events.to_numpy(); hours = windows.observed_hours.to_numpy(); is_tb = windows.label.to_numpy() == 'TB'
    observed = {k:float(v[0]) for k,v in statistics(windows.n_tb.to_numpy()[None, :], n, hours, is_tb).items()}
    direct = exact_rank_test(windows.n_tb/hours, is_tb)
    assert np.isclose(observed['original_unstratified_p'], direct['p']) and np.isclose(direct['p'], 1/66)
    strata = dict(global_counts=np.zeros(len(events), dtype=int), interval_counts=events.interictal_epoch.to_numpy(), local_counts=pd.factorize(list(zip(events.interictal_epoch, events.coverage_segment)))[0])
    replicas = 20000; rows = []; all_statistics = {}
    for offset, (condition, groups) in enumerate(strata.items()):
        rng = np.random.default_rng(1038001+offset); counts = np.zeros((replicas, 13), dtype=np.int64); group_records = []
        for group in np.unique(groups):
            selected = groups == group; sizes = np.bincount(membership[selected], minlength=13); k = int(events.loc[selected, 'label_tb'].sum())
            samples = rng.multivariate_hypergeometric(sizes, k, size=replicas)
            assert np.all(samples.sum(1) == k) and np.all(samples <= sizes)
            counts += samples
            group_records.append(dict(group=int(group), n_events=int(sizes.sum()), n_tb=k, window_counts=sizes))
        assert np.all(counts.sum(1) == int(events.label_tb.sum())) and np.all(counts[:, :12] <= n)
        stats = statistics(counts[:, :12], n, hours, is_tb); all_statistics[condition] = stats
        np.savez_compressed(OUT/f'{condition}.npz', pre15_tb_counts=counts[:, :12], **stats)
        for measure, values in stats.items():
            lo, median, hi = np.quantile(values, [.025, .5, .975])
            extreme = int(np.sum(abs(values) >= abs(observed[measure])-1e-12)) if measure in ['rank_contrast', 'median_rate_difference', 'median_tb_share_difference'] else None
            rows.append(dict(condition=condition, measure=measure, patient=observed[measure], lower=float(lo), median=float(median), upper=float(hi), n_reallocations=replicas, n_abs_at_least_patient=extreme, add_one_abs_tail=(extreme+1)/(replicas+1) if extreme is not None else None))
        write_json(OUT/f'{condition}_strata.json', group_records)
    table = pd.DataFrame(rows); table.to_csv(OUT/'comparison.csv', index=False)
    windows.to_csv(OUT/'frozen_pre15_windows.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    conditions = list(strata); labels = ['Global counts', 'Interval counts', 'Local counts']
    for ax, measure, title, unit in zip(axes, ['rank_contrast', 'median_rate_difference', 'median_tb_share_difference'], ['Rank separation of TB rates', 'Difference of median TB rates', 'Difference of median TB shares'], ['Rank-biserial contrast', 'TB events / observed hour', 'TB fraction']):
        for i, condition in enumerate(conditions):
            row = table[(table.condition == condition) & (table.measure == measure)].iloc[0]
            ax.errorbar(row['median'], i, xerr=[[row['median']-row.lower], [row.upper-row['median']]], fmt='o', color='#6688a8', capsize=4)
        ax.axvline(observed[measure], color='#b52038', linestyle='--', label='Patient')
        ax.set(yticks=range(3), yticklabels=labels, xlabel=unit, title=title, ylim=(-.5, 2.5))
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('What remains when event times and total activity are held fixed?\n20,000 exact-count mark reallocations per condition; original seizure types and windows unchanged', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .88])
    for extension in ['png', 'pdf']: fig.savefig(RUN/'figures'/f'preseizure_activity_and_mode_controls.{extension}', dpi=180)
    plt.close(fig)
    write_json(OUT/'scientific_audit.json', dict(status='COMPLETE', n_conditions=3, n_reallocations_per_condition=replicas, n_seizures=12, n_tb_seizures=2, observed=observed, strata='Global; interictal interval; clinical interval x continuous recording coverage. Every condition retains exact stratum TB totals and all event times/exposure.', numerical_checks='Disjoint original windows and exact counts; every draw preserves total and stratum counts; original tied-rank test exactly reproduced as1/66', interpretation='Global-count reallocation checks what total activity alone plus a common mark fraction can produce. Interval/local reallocation additionally preserves measured mode preferences at those coarser scales while removing within-stratum temporal mark ordering.', limits='Exchangeability is a specified diagnostic null, not an accepted model of clustered event marks. Conditional tails are exploratory specificity checks using the same development data; no correction for the full exploratory search is supplied here. Local counts are measured with the pre15 windows included, so this is not a forecast. These conditions do not preserve short-time order, do not test first passage and do not identify a physical drift.', elapsed=time.time()-started))
    readme = RUN/'figures/README.md'
    if '### preseizure_activity_and_mode_controls.png' not in readme.read_text():
        with readme.open('a') as handle:
            handle.write('\n### preseizure_activity_and_mode_controls.png\n固定全部事件时刻、12次发作类型与原预发作窗口，按全记录、临床间期段、临床段×连续覆盖段分别保留精确TA/TB数量，重分配标签各20000次。图中点与范围为重分配统计的中位数和中央95%，红线为真实统计；同时展示rank分离、率的中位数差和份额差，避免混用这些读出。\n**关注点**：原关联有多少来自总活动、粗时间尺度模式偏好或更细时序；这是同一开发数据的条件随机化诊断，既非前瞻预测，也不是保留短时聚集的动力学生成模型。\n')
    print(table[table.measure.isin(['rank_contrast', 'median_rate_difference', 'median_tb_share_difference'])].to_string(index=False))

if __name__ == '__main__':
    main()
