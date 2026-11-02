"""Validate sampling, rank ordering, loss arithmetic and requested display semantics."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    fig=plt.figure(figsize=(320/25.4,245/25.4));ab=load('ab_panel')
    a=ab.draw_a(fig,[5/320,167/245,76/320,70/245]);b=ab.draw_b(fig,[93/320,173/245,59/320,59/245])
    c=load('c_panel').render_c(fig,[164/320,164/245,151/320,73/245])
    ia=load('i_panel').render_i(fig,[253/320,15/245,53/320,51/245])
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    bax=next(ax for ax in fig.axes if ax.get_gid()=='fig4_b_sheet');inset=bax.child_axes[0]
    labels=[t for t in inset.texts if t.get_text() in b['sampling']['contacts']]
    bounds=[]
    for t in labels:
        box=t.get_window_extent(renderer)
        bounds.append(inset.bbox.contains(*box.p0) and inset.bbox.contains(*box.p1))
    checks=dict(B_sampling_restored=not b['readout_waveforms'] and b['local_sampling_inset'],
        B_three_exact_contacts=b['sampling']['contacts']==['ICL3','ICL4','ICL5'],
        B_sigma_unchanged=b['sampling']['sigma_mm']==.25,
        B_local_FOV_unchanged=b['sampling']['field_of_view_mm']==[3.8,1.9],
        B_three_Gaussian_images=len(inset.images)==3,
        B_all_contact_labels_inside=bool(all(bounds)),
        B_no_burst_waveforms=len(inset.lines)==2,
        A_connection_fonts_enlarged=min(t.get_fontsize() for t in fig.axes[0].texts)>=9.5,
        B_EI_legend_enlarged=all(t.get_fontsize()>=8.2 for t in bax.get_legend().get_texts()))
    axes=c['axes'];model_rank=axes[1].get_position();data_rank=axes[31].get_position();data_heat=axes[30].get_position();bar=axes[-1].get_position()
    checks.update(C_rank_columns_between_heatmaps=model_rank.x1<data_rank.x0 and data_rank.x1<data_heat.x0,
        C_colorbar_right_of_heatmap=bar.x0>data_heat.x1,C_colorbar_vertical=bar.height>bar.width,
        C_no_signal_frequency_subtitles=not any(t.get_text() in ['Firing density','SEEG · 80–250 Hz'] for t in c['artists'] if hasattr(t,'get_text')),
        C_three_normalized_loss_terms=sum(t.get_gid() is not None and t.get_gid().startswith('loss_term_') for t in c['artists'])==3,C_three_underbraces=sum(t.get_gid() is not None and t.get_gid().startswith('loss_underbrace_') for t in c['artists'])==3)
    checks['C_explicit_Loss_equals_and_common_third']=any(t.get_gid()=='loss_equation_lhs' and t.get_text()==r'$\mathrm{Loss}=\frac{1}{3}$' for t in c['artists'] if hasattr(t,'get_text'))
    checks['C_two_addition_signs']=sum(getattr(t,'get_text',lambda: '')()=='+' for t in c['artists'])==2
    checks['C_compact_model_data_terms']=[t.get_text() for t in c['artists'] if t.get_gid() and t.get_gid().startswith('loss_term_')]==[rf'$\ell(\hat{{{g}}},{g})$' for g in 'rop']
    checks['C_term_colors_match_H']=[t.get_color() for t in c['artists'] if t.get_gid() and t.get_gid().startswith('loss_term_')]==['#3D5C6F','#E47159','#6B7546']
    score=json.loads((HERE/'c_provenance/xy_left_20.json').read_text());frozen=json.loads((HERE/'i_j_frozen_spec.json').read_text());assert hashlib.sha256(Path(frozen['original_pickle']).read_bytes()).hexdigest()==score['objective_sha256'];expanded=sum((g['A']-g['B'])/frozen['scales'][key] for key,g in score['groups'].items())/3
    checks['C_displayed_formula_reproduces_frozen_J']=bool(np.isclose(expanded,score['J'],atol=1e-12))
    legend=ia._i_loss_axis.get_legend()
    checks.update(I_legend_upper_right=legend._loc==1,I_legend_one_column=legend._ncols==1,
        I_no_minimum_callout=not any('Min' in t.get_text() for t in ia.texts),I_four_legend_items=len(legend.get_texts())==4)
    checks={key:bool(value) for key,value in checks.items()}
    report=dict(checks=checks,all_pass=all(checks.values()),frozen_J=score['J'],expanded_J=expanded)
    (HERE/'sampling_revision_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['all_pass'];plt.close(fig)

if __name__=='__main__':main()
