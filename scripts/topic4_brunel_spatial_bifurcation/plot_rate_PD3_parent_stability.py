"""Paired full-history spectrum and three growing modes above PD3."""
from plot_rate_branch_completion import *
from complete_rate_positive_stability import paired_modes


def main():
    source = DATA/'PD3_parent_spectra/current_evidence.json'
    merged = read(source)
    parent = next(q for q in merged['rows'] if q['side'] == 'above')
    assert parent['status'] == 'UNSTABLE' and parent['numerical_unstable_dimension'] == 3
    witness = next(q for q in read(parent['physical_source'])['rows'] if q['side'] == 'above')
    assert Path(witness['orbit']).resolve() == Path(parent['orbit']).resolve()
    assert witness['physical']['filter_state_check']['positive']
    assert witness['physical']['maximum_group_defect_Hz'] < 1e-6
    attempt = next(a for a in reversed(parent['attempts'])
                   if a['classification']['numerical_unstable_dimension'] == 3)
    spectra = [read(p) for p in attempt['sources']]
    assert all(Path(q['orbit']).resolve() == Path(parent['orbit']).resolve() for q in spectra)
    for q in spectra:
        assert q['locked_subspace_invariance_defect'] < 1e-6
        assert max(q['complement_filter_relative_changes']) < 1e-5
        assert max(q['lift_condition_numbers']) < 1e10
    verdict = paired_modes(*spectra)
    assert verdict['numerical_unstable_dimension'] == 3
    assert verdict['filter_coverage'] and verdict['section_projection_checked']
    modes = [np.load(Path(p).with_suffix('.npz')) for p in attempt['sources']]
    for q, z in zip(spectra, modes):
        assert np.max(abs(values(q)-z['multipliers'])) < 1e-10
    s = RateField()
    mass, E, region, cell = s.geo['group_size'], s.E, s.geo['group_region'], s.geo['group_cell']
    weights = mass*E
    population_fraction = np.array([mass[E&(region == k)].sum() for k in range(3)])/mass[E].sum()
    assert abs(population_fraction.sum()-1) < 1e-12 and np.all(population_fraction > 0)
    counts = np.bincount(cell[E], weights=mass[E], minlength=400)
    assert s.P == 935 and len(counts) == 400
    fine = values(spectra[-1])
    growing = np.flatnonzero(verdict['outside_unit_disk_mask'])
    flip_index = int(np.argmin(abs(fine-witness['negative_multiplier'])))
    assert abs(fine[flip_index]-witness['negative_multiplier']) < 1e-4
    assert verdict['reliable_mode_mask'][flip_index]
    assert abs(fine[flip_index]) < 1-verdict['per_mode_margin'][flip_index]
    records = []
    for i in growing:
        mu = fine[i]
        assert abs(mu.imag) < 1e-10
        output_modes, selected = [], []
        for q, z in zip(spectra, modes):
            k = int(np.argmin(abs(values(q)-mu)))
            vector = z['local_vectors'][:, k]
            assert vector.shape == (9*935,)
            assert np.linalg.norm(vector.imag) < 1e-8*np.linalg.norm(vector.real)
            local = vector.real.reshape(9, 935)
            output_modes.append(s.alpha*local[0]+(1-s.alpha)*local[1])
            selected.append(k)
        x, y = output_modes
        cosine = float((weights*x)@y/np.sqrt((weights*x)@x*((weights*y)@y)))
        assert abs(cosine) > .995
        sign = np.sign(y[np.flatnonzero(E)[np.argmax(abs(y[E]))]])
        y = y*sign/np.max(abs(y[E]))
        field = np.bincount(cell[E], weights=mass[E]*y[E], minlength=400)/np.maximum(counts, 1)
        energy = np.array([np.sum(mass[E&(region == k)]*y[E&(region == k)]**2) for k in range(3)])
        energy /= energy.sum()
        records.append(dict(fine_mode_index=int(i), multiplier=mu,
            paired_mode_indices=selected, reference_phase_E_output_cosine=abs(cosine),
            reference_phase_E_mode_energy_A_B_surround=energy,
            per_E_cell_mode_energy_relative_to_network=energy/population_fraction, field_E=field))
    plt.rcParams.update({'font.size':11, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, grid = plt.subplots(2, 2, figsize=(10.8, 9.0))
    fig.subplots_adjust(left=.09, right=.96, bottom=.08, top=.86, hspace=.43, wspace=.40)
    ax = grid.ravel()[0]
    for q, color, marker in zip(spectra, ['#d2691e', '#2166ac'], ['+', 'o']):
        modulus = np.sort(abs(values(q)))[::-1]
        ax.plot(np.arange(1, len(modulus)+1), modulus, marker, mfc='none', color=color,
                ms=7, label=f'dt = {q["dt_ms"]:.5g} ms')
    ax.axhline(1, color='black', ls='--', lw=1)
    rank = int(np.flatnonzero(np.argsort(abs(fine))[::-1] == flip_index)[0])+1
    ax.plot(rank, abs(fine[flip_index]), 'v', color='black', ms=6)
    ax.annotate(rf'PD3 mode: $\mu={fine[flip_index].real:.4f}$',
                (rank, abs(fine[flip_index])), xytext=(7, 3), textcoords='data',
                fontsize=9, arrowprops=dict(arrowstyle='-', lw=.8))
    ax.set(yscale='log', xlabel='Returned mode (ordered by modulus)',
           ylabel=r'$|\mu|$', title='A   Three unstable directions', xlim=(.5, len(fine)+.5))
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    style(ax)
    geo = np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    for ax, row, letter in zip(grid.ravel()[1:], records, 'BCD'):
        im = ax.imshow(row['field_E'].reshape(20, 20), origin='lower', extent=(0, 20, 0, 20),
                       cmap='RdBu_r', vmin=-1, vmax=1)
        for k, center in enumerate(s.geo['centers_mm']):
            ax.add_patch(plt.Circle(center, 1.5, fill=False, color='black', lw=.9))
            ax.text(*center, 'AB'[k], ha='center', va='center', fontsize=9)
        ax.scatter(geo['contact_xy'][:, 0], geo['contact_xy'][:, 1], s=12,
                   facecolors='none', edgecolors='#00a6b2', linewidths=.8)
        ax.set(xlabel='x (mm)', ylabel='y (mm)',
               title=rf'{letter}   $\mu={row["multiplier"].real:.6g}$: E-rate mode')
        fig.colorbar(im, ax=ax, pad=.04, shrink=.85, label='Relative E-rate perturbation')
    fig.suptitle('Higher-J parent near PD3: full-delay spectrum and spatial modes\n'+
        rf'$J_{{\mathrm{{EE,core}}}}={parent["J_EE_core"]:.8f}$'+
        f' | T = {spectra[-1]["T_ms"]:.3f} ms | Modes at reference phase', fontsize=13, y=.98)
    name = 'PD3_parent_full_history_stability'
    save_new(fig, name)
    write(DATA/(name+'.json'), dict(status='PAIRED_PARENT_DIMENSION_AND_MODES_CHECKED',
        source=str(source), physical_source=parent['physical_source'], orbit=parent['orbit'],
        J_EE_core=parent['J_EE_core'], spectrum_sources=attempt['sources'],
        paired_classification=verdict, growing_modes=records,
        PD3_crossing_mode_at_this_sample=dict(multiplier=fine[flip_index],
            fine_mode_index=flip_index, inside_unit_disk=True,
            matched_physical_parent_witness_multiplier=witness['negative_multiplier']),
        E_population_fraction_A_B_surround=population_fraction,
        spatial_model=dict(cells=400, populations=935, local_states=8415),
        scope='Exactly three numerically verified unstable directions at the higher-J parent sample. Every displayed rate mode is from the full-state/history Poincare eigenvector with the phase direction removed. Spatial maps are reference-phase linear perturbations, not spontaneous propagation snapshots or causal contributions. The lower-J total dimension and the complete intervening interval remain unresolved.'))
    path = OUTPUT/'figures/README.md'
    body = path.read_text()
    body = re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)', '', body, flags=re.M|re.S).rstrip()
    body += '\n\n### '+name+'.png\n展示 PD3 高参数侧一个物理周期解的两步长完整延迟谱，以及三个增长方向在参考相位的二维 E 率特征向量。原 400 空间格、935 群体及触点几何保持不变，模态分别按最大群体 E 率扰动归一化。**关注点**：三个不稳定方向只在这个精确参数样本上完成数值验证；模态图不是自发传播快照，低参数侧总维数与整个区间仍待核验。\n'
    path.write_text(body)
    print('PD3 ABOVE DIMENSION', verdict['numerical_unstable_dimension'],
          'MODE COSINES', [q['reference_phase_E_output_cosine'] for q in records], flush=True)


if __name__ == '__main__':
    main()
