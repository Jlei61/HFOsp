"""Show the actual matched Z intervention and its spatial states."""
from common import OUT, np, read, write, model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    base = OUT / 'core_a_bifurcation_type_20260924/reference_stability_gap/matched_natural_short_entry_step'
    audit = read(base / 'independent_audit.json')
    assert audit['status'] == 'PASS'
    s = model(40)
    times = [2600, 3450, 8500]
    colors = ['#237C88', '#C87522']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(9.0, 7.0), layout='constrained')
    gs = fig.add_gridspec(3, 4, height_ratios=[.75, 1, 1],
                          width_ratios=[1, 1, 1, .055], hspace=.08)
    ax = fig.add_subplot(gs[0, :3])
    exported = []
    for j, row in enumerate(audit['rows']):
        y = 1-j
        for event in row['regions'][1]['activities']:
            start, duration = event['start_ms']/1000, event['duration_ms']/1000
            ax.broken_barh([(start, duration)], (y-.13, .26), facecolors=colors[j])
            if event['right_censored']:
                ax.plot(start+duration, y, marker='>', color=colors[j], clip_on=False)
        data = np.load(base / row['label'] / 'trajectory.npz')
        fields = []
        for k, tm in enumerate(times):
            use = (data['time_ms'] > tm-25) & (data['time_ms'] <= tm+25)
            assert use.sum() == 50
            field = data['field_E_hz'][use].astype(float).mean(0)
            fields.append(field.tolist())
            panel = fig.add_subplot(gs[j+1, k])
            im = panel.imshow(field.reshape(20, 20), origin='lower', extent=[0, 20, 0, 20],
                              vmin=0, vmax=500, cmap='magma', interpolation='nearest')
            panel.set_xticks([0, 10, 20]); panel.set_yticks([0, 10, 20])
            if j == 1: panel.set_xlabel('x (mm)')
            if k == 0:
                panel.set_ylabel('y (mm)')
                panel.text(-.20, 1.05, 'BC'[j], transform=panel.transAxes,
                           fontsize=16, fontweight='bold')
                panel.text(.02, .97, f'$D_A={1-row["Z_A"]:.4f}$',
                           transform=panel.transAxes, va='top', color='white', fontsize=11)
            else: panel.set_yticklabels([])
            if j == 0: panel.text(.5, 1.04, f'{tm/1000:.2f} s', transform=panel.transAxes, ha='center')
            for name, center in zip('AB', s.geo['centers_mm']):
                panel.add_patch(Circle(center, 1.5, fill=False, color='#20CCD0', lw=1.1))
                panel.text(center[0], center[1]+1.9, name, ha='center', color='#20CCD0', fontsize=10)
        exported.append(dict(label=row['label'], Z_A=row['Z_A'], fields_E_hz=fields))
    ax.set_xlim(0, 10); ax.set_ylim(-.5, 1.55)
    ax.set_yticks([1, 0], [r'$D_A=0.2985$', r'$D_A=0.3000$'])
    ax.set_xlabel('Time after Z intervention (s)')
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.text(-.13, 1.03, 'A', transform=ax.transAxes, fontweight='bold', fontsize=16)
    ax.text(0, 1.03, 'Core A activity', transform=ax.transAxes)
    for t in times: ax.plot(t/1000, 1.44, marker='v', color='black', markersize=5)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1:, 3]))
    cb.set_ticks([0, 250, 500]); cb.set_label('E rate (Hz / neuron)')
    folder = base / 'figures'; folder.mkdir(exist_ok=True)
    name = 'fig_matched_local_Z_entry'
    for ext in ['png', 'pdf', 'svg']: fig.savefig(folder / f'{name}.{ext}', dpi=220)
    plt.close(fig)
    write(folder / f'{name}.json', dict(source=str(base), independent_audit='PASS',
          dt_ms=.05, times_ms=times, window_ms=50, fields=exported,
          fixed_Z_dynamic_M=True, initial_full_state_matched=True,
          scope='Finite10s matched spatial-rate intervention. Bars show CoreA activity; right arrows mark censoring. No stable/unstable branch or critical type is claimed.',
          human_visual_acceptance='PENDING', model_promoted=False))
    (folder / 'README.md').write_text(
        '### fig_matched_local_Z_entry.png / .pdf / .svg\n\n'
        '同一完整短事件末态出发，仅将核 A 的原生资源场由 Z_A≈0.7015 改为0.700；核外 Z 保持，全部兴奋性 M 动态演化。'
        '上排是10秒内核 A 活动段，右端箭头表示记录结束时尚未结束；下两排比较同一三个时刻的50ms空间活动场，使用相同0–500Hz色标。'
        '该对照证明当前0.05ms数值步长下的小幅局部耗减可招募长活动，但不能单独命名分岔或等同于全局发作。\n\n'
        '**关注点**：降低核内Z后出现长活动而保持场仍反复自行终止；也要检查核B和外围没有被误称为全局持续。细步控制另行报告，PNG/PDF待用户人工检查。\n')


if __name__ == '__main__': main()
