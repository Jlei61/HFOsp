"""Show why a locally verified cycle fold need not create a stable burst."""
from plot_rate_branch_completion import *


def main():
    source = DATA / 'primary_folds/LPC_B2_root_spectrum_assessment.json'
    q = read(source)
    assert q['status'] == 'FOLD_WITH_VERIFIED_UNSTABLE_MODES'
    assert all(v['identity_checked'] for v in q['critical_mode_identity'])
    mu = values(q['paired_spectrum'])
    reliable = np.asarray(q['paired_spectrum']['reliable_mode_mask'])
    outside = np.asarray(q['paired_spectrum']['outside_unit_disk_mask'])
    critical = q['critical_mode_identity'][-1]['critical_mode_index']
    plt.rcParams.update({'font.size':11, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.1), gridspec_kw={'width_ratios':[1,1.2]})
    fig.subplots_adjust(left=.08, right=.98, bottom=.24, top=.79, wspace=.32)
    theta = np.linspace(0, 2*np.pi, 500)
    ax = axes[0]
    ax.plot(np.cos(theta), np.sin(theta), '--', color='black', lw=.9)
    ax.axhline(0, color='#dddddd', lw=.6)
    ax.axvline(0, color='#dddddd', lw=.6)
    for j, z in enumerate(mu):
        if abs(z.real)>1.6 or abs(z.imag)>1.2:
            continue
        color = '#c74e39' if outside[j] else '#2878b5'
        marker = 's' if j==critical else 'o' if reliable[j] else 'x'
        if j==critical:color='black'
        ax.plot(z.real, z.imag, marker, ms=7, color=color, zorder=4)
    ax.annotate(r'Fold mode: $\mu\to+1$', (mu[critical].real, 0),
                xytext=(-1.35,-.65), arrowprops=dict(arrowstyle='-', lw=.8), fontsize=10)
    ax.set(xlim=(-1.55,1.6), ylim=(-1.2,1.2), aspect='equal',
           xlabel=r'Re $\mu$', ylabel=r'Im $\mu$', title='A  Spectrum near the unit circle')
    ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1]); style(ax)
    ax = axes[1]
    ax.axhline(1, color='black', ls='--', lw=.9)
    for k, identity in enumerate(q['critical_mode_identity']):
        spectrum = read(identity['source'])
        vals = values(spectrum)
        # The paired source ordering is verified instead of assuming that
        # sorting by modulus preserves mode identity across time steps.
        from scipy.optimize import linear_sum_assignment
        ii, jj = linear_sum_assignment(abs(vals[:,None]-mu[None,:]))
        aligned = np.empty_like(mu); aligned[jj] = vals[ii]
        for j,z in enumerate(aligned):
            color = '#c74e39' if outside[j] else '#2878b5'
            marker = 's' if j==critical else 'o' if reliable[j] else 'x'
            if j==critical:color='black'
            ax.plot(j+1+(-.08 if k==0 else .08), abs(z), marker,
                ms=7 if k==0 else 5, color=color,
                mfc='white' if k==0 else color, mew=1., zorder=4)
    ax.annotate(rf'$\mu={mu[0].real:.3f}$', (1,abs(mu[0])), xytext=(1.4,40),
                fontsize=10, arrowprops=dict(arrowstyle='-',lw=.7))
    ax.text(3.75,2.6,'3 verified unstable directions',fontsize=10,ha='center')
    ax.set(xlim=(.55,6.6), ylim=(.002,70), yscale='log', xticks=np.arange(1,7),
           xlabel='Returned mode index', ylabel=r'Multiplier modulus $|\mu|$',
           title='B  All six returned modes')
    ax.set_yticks([.01,.1,1,10]); ax.set_yticklabels(['0.01','0.1','1','10']); style(ax)
    fig.suptitle('LPC13: a fold of unstable cycles\n'+
        rf'$J_{{\mathrm{{EE,core}}}}={q["J_EE_core"]:.8f}$; $T={q["T_ms"]:.3f}$ ms',
        fontsize=14, y=.98)
    legend=[
        Line2D([0],[0],marker='o',mfc='white',mec='#2878b5',ls='',label=r'$\Delta t\approx0.05$ ms'),
        Line2D([0],[0],marker='o',color='#2878b5',ls='',label=r'$\Delta t\approx0.025$ ms'),
        Line2D([0],[0],marker='o',color='#c74e39',ls='',label='Verified unstable mode'),
        Line2D([0],[0],marker='s',color='black',ls='',label='Matched fold mode')]
    if not np.all(reliable):
        legend.append(Line2D([0],[0],marker='x',color='#2878b5',ls='',label='Small-mode residual pending'))
    fig.legend(handles=legend,
        loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,frameon=False,fontsize=9)
    name='H2_unstable_fold_spectrum'
    save_new(fig,name)
    write(DATA/(name+'.json'),dict(source=str(source),
        verified_unstable_real_dimension_lower_bound=int(outside.sum()),
        full_unstable_dimension=q['numerical_unstable_dimension_excluding_fold'],
        complex_plane_xlim=[-1.55,1.6],all_returned_modes_in_right_panel=True,
        scope='The fold mode is independently matched to the full-history BVP tangent. '
              'A real growing mode and a conjugate growing pair are verified. '
              +('All remaining returned modes pass the paired-step residual and modulus checks. '
                if q['remaining_spectrum_classified'] else
                'The smallest returned mode misses the residual gate; the total unstable dimension is not asserted. ')+
              'No branch-wide or global completeness claim.'))
    path=OUTPUT/'figures/README.md'
    body=path.read_text()
    body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    detail=('完整返回子空间通过配对时间步及独立传播检查，除折叠方向外有 3 个数值不稳定方向。'
            if q['remaining_spectrum_classified'] else
            '最小模态尚未通过残差门槛，因此不报告完整不稳定维数。')
    body+='\n\n### '+name+'.png\n显示 H2 家族 LPC13 的非平凡 Floquet 谱，左侧放大单位圆附近，右侧保留全部六个返回模态，包括左侧范围外约为 23 的实乘子。折叠模态通过完整状态及延迟历史的特征向量匹配确认；另有一个实乘子和一对复乘子位于单位圆外。**关注点**：这是已不稳定周期解上的折叠，不能解释成稳定 burst 的产生；'+detail+'\n'
    path.write_text(body)


if __name__=='__main__':
    main()
