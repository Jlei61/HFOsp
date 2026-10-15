"""Three-color absolute loss trajectories with equal-epoch support, not rescaled starts."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import csv
from modeling.scripts.topic4_core_count_optimization.common import *
COLORS = {1: '#397DAA', 2: '#8C4B20', 3: '#8073AC'}

def trajectory(rows, epochs):
    by_epoch = {r['epoch']: r for r in rows}
    best = None
    out = np.full((epochs, 4), np.nan)
    support = 0
    for i in range(1, epochs + 1):
        if i not in by_epoch:
            break
        row = by_epoch[i]
        support = i
        if row['J'] is not None and (best is None or row['J'] < best['J']):
            best = row
        if best is not None:
            out[i - 1] = [best['J']] + [best['groups'][g]['scaled'] for g in GROUPS]
    return (out, support)

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan = read(OUT / 'plan.json')
    epochs = plan['epochs']
    repeats = plan['restarts']
    data = {}
    progress = []
    allrows = []
    confirmations = []
    for k in plan['core_counts']:
        data[k] = []
        for restart in range(repeats):
            d = run_dir(k, restart)
            rows = [read(p) for p in sorted((d / 'scores').glob('*.json'))]
            for row in rows:
                allrows.append({name: row.get(name) for name in ('core_count', 'restart', 'epoch', 'candidate', 'status', 'N', 'J')} | {g: row['groups'].get(g, {}).get('scaled') for g in GROUPS})
            arr, support = trajectory(rows, epochs)
            data[k].append(arr)
            progress.append(dict(core_count=k, restart=restart, evaluated=len(rows), contiguous_epochs=support, scorable=sum((r['J'] is not None for r in rows)), final_training_J=arr[-1, 0] if support == epochs else None))
            for path in sorted((d / 'confirmation').glob('*/score.json')):
                confirmations.append(read(path))
        data[k] = np.asarray(data[k])
    summary = dict(status='TRAINING_COMPLETE' if len(allrows) == plan['train_simulations'] else 'RUNNING_PARTIAL', subject=plan['subject'], training_complete=len(allrows), training_planned=plan['train_simulations'], progress=progress, confirmation_complete=len(confirmations), confirmation_planned=plan['confirmation_simulations'], confirmations=confirmations, objective=plan['objective'], epoch_definition=plan['epoch_definition'], colors=COLORS, limitations=plan['limitations'], updated_unix=time.time())
    if summary['status'] == 'TRAINING_COMPLETE':
        endpoints = {k: data[k][:, -1, 0] for k in data}
        summary['training_endpoints'] = endpoints
        summary['two_minus_one_per_restart'] = endpoints[2] - endpoints[1]
        summary['three_minus_two_per_restart'] = endpoints[3] - endpoints[2]
    write(OUT / 'review.json', summary)
    if not allrows:
        return
    fields = ['core_count', 'restart', 'epoch', 'candidate', 'status', 'N', 'J', *GROUPS]
    temp = OUT / 'loss_by_epoch.csv.tmp'
    with temp.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(allrows)
    temp.replace(OUT / 'loss_by_epoch.csv')
    figdir = OUT / 'figures'
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})

    def draw(ax, component):
        for k, values in data.items():
            for curve in values[:, :, component]:
                ax.plot(np.arange(1, epochs + 1), curve, color=COLORS[k], alpha=0.25, lw=0.75)
            block = values[:, :, component]
            shared = np.isfinite(block).all(axis=0)
            x = np.flatnonzero(shared) + 1
            if len(x):
                q = np.quantile(block[:, shared], [0.25, 0.5, 0.75], axis=0)
                ax.fill_between(x, q[0], q[2], color=COLORS[k], alpha=0.12, linewidth=0)
                ax.plot(x, q[1], color=COLORS[k], lw=2.1, label=f'{k} core' if k == 1 else f'{k} cores')
            else:
                ax.plot([], [], color=COLORS[k], lw=2.1, label=f'{k} core' if k == 1 else f'{k} cores')
        ax.set_xlim(1, epochs)
        ax.set_xticks([1, 24, 48, 72, 96])
        ax.set_xlabel('Epochs')
        ax.tick_params(direction='out', width=0.8)
        ax.grid(False)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    draw(ax, 0)
    ax.set_ylabel('Loss')
    ax.legend(frameon=False, loc='upper right')
    ax.set_title('E1146', fontsize=11)
    if not any((np.isfinite(a[:, :, 0]).any() for a in data.values())):
        ax.text(0.5, 0.5, 'No scorable configurations yet', transform=ax.transAxes, ha='center', fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(figdir / f'e1146_core_count_loss.{suffix}', dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for j, (ax, label) in enumerate(zip(axes, ['Rank distribution', 'Order / timing', 'Participation'])):
        draw(ax, j + 1)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('Scaled error')
    axes[-1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(figdir / f'e1146_core_count_components.{suffix}', dpi=220)
    plt.close(fig)
    (figdir / 'README.md').write_text('### e1146_core_count_loss.png\nE1146 单、双、三 core 的同预算优化曲线，另有同源 PDF/SVG。细线为四次随机优化各自已评估最优 loss，粗线及阴影为四次均具同一 epoch 有效值时的中位数及四分位范围；每个 epoch 是一条新配置的 60 秒仿真，按提案顺序计数。三组使用同一冻结 loss 尺度，不按各自起点重新归一化；运行中曲线只画已完成前缀。\n**关注点**：曲线按定义单调不增；随机优化不是患者或网络重复，优劣须结合最终噪声复测，尚待作者目视检查。\n\n### e1146_core_count_components.png\n三项分量对应同一总 loss 最优候选，另有同源 PDF/SVG。分别为 rank 分布、局部及跨杆顺序/时差、触点参与，沿用原 E1146 评分器及 CAL 尺度；不是逐项各挑最低值拼接。\n**关注点**：某分量可能随总 loss 下降而上升；core 数同时改变总易激区域与随机输入支持，本实验未保持总剂量相等。\n')
if __name__ == '__main__':
    main()
