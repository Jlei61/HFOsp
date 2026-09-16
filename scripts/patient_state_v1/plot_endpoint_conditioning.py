"""Reproduce the completed endpoint-selection calibration figure."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial', action='store_true')
    args = parser.parse_args()
    folder = RUN/('initial_conditioning_calibration_v1_35' if args.initial else 'endpoint_conditioning_calibration_v1_34')
    samples = pd.read_csv(folder/'prefix_statistics.csv')
    comparison = pd.read_csv(folder/'calibration_comparison.csv')
    assert len(samples) == 384 and len(comparison) == 6
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    kind = 'initial' if args.initial else 'endpoint'
    display = 'Initial state' if args.initial else 'Endpoint'
    labels = [f'Stationary Gaussian {kind}', f'{display} selected at ±2 SD', f'{display} selected at ±3 SD']
    colors = ['#6d96ba', '#d48b4e', '#895ba4']
    for ax, history in zip(axes, [False, True]):
        for condition, label, color in zip([f'gaussian_{kind}', f'selected_{kind}_2sd', f'selected_{kind}_3sd'], labels, colors):
            part = samples[(samples.history == history) & (samples.condition == condition)]
            assert len(part) == 64
            ax.hist(part.delta_log_tau_prefix1_minus_full, bins=np.linspace(-.8, 1.35, 30), histtype='step', color=color, linewidth=1.6, label=label)
        patient = comparison[comparison.history == history].patient.iloc[0]
        ax.axvline(patient, color='#b51c35', linewidth=2, label='Patient')
        ax.set(title='OU + short memory fit' if history else 'OU fit', xlabel='First-prefix minus full log time constant', ylabel='Synthetic sequences')
        ax.spines[['top', 'right']].set_visible(False)
    question = 'Can nonstationary initial states explain the prefix shift?' if args.initial else 'Can informative interval endpoints explain the prefix shift?'
    subtitle = '64 sequences per initial law; actual delay from clinical offset to first event retained' if args.initial else '64 sequences per endpoint law; fixed patient event times and clinical interval ends'
    fig.suptitle(question+'\n'+subtitle)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2)
    fig.tight_layout(rect=[0, .15, 1, .84])
    stem = 'initial_conditioning_calibration' if args.initial else 'endpoint_conditioning_calibration'
    for extension in ['png', 'pdf']:
        fig.savefig(RUN/'figures'/f'{stem}.{extension}', dpi=180)
    plt.close(fig)
    if args.initial:
        readme = RUN/'figures/README.md'
        if f'### {stem}.png' not in readme.read_text():
            with readme.open('a') as handle:
                handle.write(f'\n### {stem}.png\n固定同一OU动力学，在真实临床offset假设平稳高斯或±2/3SD初态，各生成64条新标签序列，再按患者相同方法拟合前缀与全数据。首个观察事件前的真实延迟被保留；红线是患者的前缀减全数据log时间常数差。\n**关注点**：初态瞬态是否能被误读为时间尺度变化；这是给定初态分布的合成诊断，并未识别患者生理重置。\n')

if __name__ == '__main__':
    main()
