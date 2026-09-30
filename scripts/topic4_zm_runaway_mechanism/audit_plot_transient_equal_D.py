"""Independent saved-data checks and fixed-D spatial intervention display."""
from common import OUT, model, np, read, write, log
from transient_equal_D_fields import DEST
from refractory_spatial_resolution import mapping, projections
from native_readouts import readouts, window_stats
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    assert read(DEST/'jobs.json')['status'] == 'COMPLETE'
    s, coarse = model(40), model(20)
    parent, _ = mapping(coarse, s)
    P, count = projections(s, coarse, parent)[20]
    weights = count/count.sum()
    report = read(DEST/'result.json')
    fields = np.load(DEST/'fields.npz')
    qa, traces = [], []
    for row in report['rows']:
        label = row['arm']
        z = np.load(DEST/label/'trajectory.npz')
        t, r, field = z['time_ms'], z['group_rate_hz'].astype(float), z['field_E_hz'].astype(float)
        assert np.array_equal(z['Z'], fields[label])
        assert abs(1-z['Z'][s.E]@s.mean_weights-.20) < 1e-12
        assert np.array_equal(t, np.arange(9001, 12501.))
        spatial_error = float(np.max(abs((P@r.T).T-field)))
        weighted_error = float(np.max(abs(field@weights-z['global_E_hz'])))
        assert max(spatial_error, weighted_error) < 1e-4
        counts = r*s.sizes/1000
        assert np.max(abs(counts-np.rint(counts))) < 1e-4
        assert np.max(counts[:-1, s.E]+counts[1:, s.E]-s.sizes[s.E]) < 1e-4
        assert np.max(counts[:, ~s.E]-s.sizes[~s.E]) < 1e-4
        events, summary, whole, sm = readouts(t, field, count, label)
        complete = []
        for ev in events:
            a = int(np.searchsorted(t, ev['start_ms']))
            b = a+int(ev['duration_ms'])
            if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():
                complete.append(ev)
        assert summary['high_onset_ms'] == row['high_entry_ms']
        stored_metrics = window_stats(complete, 9000, 12500)
        # Onset-centroid extent can change at exact threshold ties after the
        # recorded float32 field quantization. Require every discrete event,
        # duration, direction, participation and state readout to reproduce;
        # retain and report the extent difference, without changing any gate.
        original_metrics = row['complete_events']
        for key, value in stored_metrics.items():
            if key != 'median_extent_mm':
                assert value == original_metrics[key], (label, key, value, original_metrics[key])
        extent_difference = (stored_metrics.get('median_extent_mm', 0.)-
            original_metrics.get('median_extent_mm', 0.))
        assert abs(summary['quiet_fraction']-row['quiet_fraction']) < 1e-12
        persistence = float(weights[(field[-1000:]>50).mean(0)>=.9].sum())
        assert abs(persistence-row['tail_persistent_fraction']) < 1e-12
        assert abs(whole[-1000:].mean()-row['tail_global_hz']) < 1e-4
        assert np.isfinite(z['M_current']).all() and z['M_current'].min() >= 0
        qa.append(dict(arm=label, spatial_error_hz=spatial_error, weighted_error_hz=weighted_error,
            D=float(1-z['Z'][s.E]@s.mean_weights), counts_physical=True, state_and_discrete_event_readouts_reproduced=True,
            saved_precision_median_extent_difference_mm=extent_difference,
            stored_field_complete_events=stored_metrics,
            runtime_float64_complete_events=original_metrics))
        traces.append((t, sm, (P@z['Z']).reshape(20, 20), field[-1000:].mean(0).reshape(20, 20)))
    write(DEST/'independent_audit.json', dict(status='PASS', rows=qa,
        scope=report['scope'], model_promoted=False))
    plt.rcParams.update({'font.size':10, 'pdf.fonttype':42, 'svg.fonttype':'none',
        'axes.spines.top':False, 'axes.spines.right':False})
    fig = plt.figure(figsize=(7.8, 8.6))
    gs = fig.add_gridspec(3, 2, left=.1, right=.80, bottom=.08, top=.93,
        height_ratios=[1, .65, 1], hspace=.48, wspace=.28)
    for col, ((t, rate, Z, field), name) in enumerate(zip(traces, ['Native Z pattern', 'Free-rate Z pattern'])):
        for row, values, cmap, vmax, label in [(0, Z, 'viridis', 1, 'A'), (2, field, 'magma', 500, 'C')]:
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(values, origin='lower', extent=[0, 20, 0, 20], cmap=cmap,
                vmin=0, vmax=vmax, interpolation='nearest')
            ax.set(xticks=[0, 10, 20], yticks=[0, 10, 20], xlabel='x (mm)')
            if col == 0:
                ax.set_ylabel('y (mm)')
                ax.text(-.27, 1.08, label, fontsize=16, fontweight='bold', transform=ax.transAxes)
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.text(.5, 1.1, name, ha='center', transform=ax.transAxes)
            for center, letter in zip(s.geo['centers_mm'], 'AB'):
                ax.add_patch(Circle(center, 1.75, fill=False, color='#20c4cf' if row==2 else 'white', lw=1))
                ax.text(center[0], center[1]+2.2, letter, color='#20c4cf' if row==2 else 'white', ha='center')
            if col == 1:
                bar = fig.add_axes([.855, .70 if row==0 else .08, .018, .23])
                fig.colorbar(im, cax=bar, label='Z' if row==0 else 'Mean E rate (Hz)')
        ax = fig.add_subplot(gs[1, col])
        ax.plot(t/1000, rate, color='#202020' if col==0 else '#a35f36', lw=.8)
        ax.set(xlim=(9, 12.5), ylim=(0, 500), xlabel='Time (s)', yticks=[0, 250, 500])
        if col == 0:
            ax.set_ylabel('Global E rate (Hz)')
            ax.text(-.27, 1.08, 'B', fontsize=16, fontweight='bold', transform=ax.transAxes)
        else:
            ax.tick_params(labelleft=False)
    folder = DEST/'figures'
    folder.mkdir(exist_ok=True)
    stem = 'fig_equal_D_spatial_feedback'
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(folder/f'{stem}.{ext}', dpi=190)
    plt.close(fig)
    write(folder/f'{stem}.json', dict(source=str(DEST/'result.json'), D=.20,
        top='Two complete prescribed Z fields with identical cell-weighted meanD. Projected to original20x20 display grid.',
        middle='Same full9000ms history, M dynamic, same future exogenous input and count innovation keys; original10ms rate smoothing.',
        bottom='Mean spatial E rate over final11.5-12.5s; not instantaneous snapshots or equilibrium branches.',
        Z_held=True, M_dynamic=True, bifurcation_type='NOT_ESTABLISHED',
        agent_PNG_PDF_check='PENDING', human_visual_acceptance='PENDING'))
    (folder/'README.md').write_text(f'### {stem}.png / .pdf / .svg\n'
        '两列的全E平均耗减严格同为D=0.20，但左列沿原生Z空间分布，右列沿自由率模型的Z空间分布。'
        'A为固定Z场，B为相同9秒完整快历史、相同M初态及未来输入下的全局率，C为末1秒二维平均放电率；两臂M都动态。'
        '**关注点**：相同平均资源下空间耗减差异是否改变自限和持续活动；这是单个配对历史的有限时间干预，不是吸引子、separatrix或分岔类型认证。\n')
    log('EQUAL D AUDIT AND FIGURE COMPLETE', qa)


if __name__ == '__main__':
    main()
