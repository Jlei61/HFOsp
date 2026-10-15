#!/usr/bin/env python3
"""Render and publish the author-designated A-J Fig4 with J's upper-right legend."""
from pathlib import Path
import argparse
import csv
import json
import shutil
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from matplotlib.transforms import Bbox
from scripts.paper_figures import build_fig4_core_count_candidate as base

plate=base.plate
OUT=ROOT/'results/paper-ready-figure/fig4/candidates/complete_aj_legend_20260929'
DEST=ROOT/'results/paper-ready-figure/fig4'
PARENT=base.OUT
ARCHIVE=ROOT/'results/paper-ready-figure/archive/2026-09-29_pre_aj_legend_fig4/fig4'
PRODUCER=str(Path(__file__).relative_to(ROOT))
CORE_PREVIEW=None
PARAMETER_PREVIEW=None
DENSITY_PREVIEW=None
ALIGN_EFH_WITH_B=False
MATCH_DG_WIDTH_TO_A=False
FINISH_D_RIGHT_COLUMN=False
CURRENT_D_VERSION='three_error_responses_density_v1'
B_MIN_STAGE=6
B_VERSION='post_prior_stages_6_16_v1'
B_PRIOR_NOTE=('Stages 1-5 belong to the preceding label-informed development phase and are omitted from B. '
              'Subsequent searches inherit its candidate pool and patient geometry as development priors; '
              'their scoring objectives omit TA/TB labels. This display restriction does not establish '
              'independence from development data or remove historical information use.')


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def align_efh_with_b(fig):
    """Align E/F/H to the unchanged B, making room beside D at fixed canvas size."""
    target=plate.GROUPS['B']['axes'][0].get_position().x0*plate.W
    np.testing.assert_allclose(target,148,atol=1e-7)
    def move(ax,x,width=None):
        _,y,w,h=np.asarray(ax.get_position().bounds)*[plate.W,plate.H,plate.W,plate.H]
        plate.pos(ax,x,y,w if width is None else width,h)
    audited=[]
    for key in ('D','E','G','H','I'):
        for ax in plate.GROUPS[key]['axes']:
            audited.extend((artist,np.array(artist.get_array()).copy()) for artist in ax.images)
            audited.extend((artist,np.array(artist.get_ydata()).copy()) for artist in ax.lines)
    for key in ('D','E'):
        group=plate.GROUPS[key]
        envelopes=[a for a in group['axes'] if a.images and a.get_xlim()[0]<0]
        envelopes.sort(key=lambda a:a.get_position().x0)
        for ax,x in zip(envelopes,[target,175.]):move(ax,x,18.)
        scale=next(a for a in group['axes'] if a.get_visible() and not a.images and
                   abs(a.get_position().x0*plate.W-192)<1e-5)
        move(scale,195.)
        for t in group['texts']:
            if t.get_rotation()==0 and t.get_text() in ('MTA','MTB'):
                _,y=t.get_position();t.set_position(((157 if t.get_text()=='MTA' else 184)/plate.W,y))
    move(plate.GROUPS['G']['axes'][0],target,23.)
    for key in ('H','I'):
        for ax in plate.GROUPS[key]['axes']:move(ax,ax.get_position().x0*plate.W+4)
    for t in fig.texts:
        if t.get_text() in ('E','F','H','I','J') and t.get_fontsize()==plate.FONT['letter']:
            _,y=t.get_position();t.set_position((dict(E=137,F=137,H=137,I=174,J=234)[t.get_text()]/plate.W,y))
    for artist,values in audited:
        actual=artist.get_array() if hasattr(artist,'get_array') else artist.get_ydata()
        np.testing.assert_array_equal(actual,values)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    np.testing.assert_allclose([plate.GROUPS[k]['axes'][0].get_position().x0*plate.W
                               for k in ('B','D','E','G')],target,atol=1e-7,rtol=0)
    # All four panels now have identical left data edges and letter anchors.
    positions={t.get_text():t.get_position()[0]*plate.W for t in fig.texts
               if t.get_text() in ('B','E','F','H') and t.get_fontsize()==plate.FONT['letter']}
    np.testing.assert_allclose(list(positions.values()),137,atol=1e-7,rtol=0)
    bar=plate.GROUPS['C']['axes'][-1]
    contact_left=min(t.get_window_extent(renderer).x0 for t in plate.GROUPS['E']['axes'][0].get_yticklabels() if t.get_visible())
    gap=(contact_left-bar.get_tightbbox(renderer).x1)/fig.dpi*25.4
    assert gap>3,gap
    plate.CHECKS['BEFH_alignment']=dict(version='efh_aligned_to_b_v1',left_axis_x_mm=target,
        letter_x_mm=137,added_gap_beside_D_mm=18,envelope_width_mm=18,H_width_mm=23,
        I_translation_mm=4,J_translation_mm=4,canvas_size_unchanged=True,fonts_unchanged=True,
        image_and_curve_values_unchanged=True,D_colourbar_to_F_contacts_gap_mm=gap)


def require_current_d(record, *, publication=False):
    """Reject missing/legacy D before rendering or touching published files."""
    panel=record.get('panel_d',{})
    if panel.get('version')!=CURRENT_D_VERSION or not panel.get('source_package'):
        raise ValueError('Fig4D version is missing or obsolete. Read docs/current_figure4.md and '
                         'restore panel_d from the current registry; never fall back to D_candidate_right_axis.')
    if publication:
        checks=record.get('original_panel_checks',{})
        valid=(checks.get('D_parameter_layout',{}).get('single_left_error_axes')==3
               and checks.get('D_recurrent_EE',{}).get('metrics_recomputed_from_primary_event_times') is True
               and checks.get('D_position_density',{}).get('selection_metric')=='J_joint'
               and checks.get('D_EF_workpoint_references',{}).get('verified_against_executed_design_and_applied_physics') is True)
        if not valid:
            raise ValueError('Cannot publish Fig4: latest D curves, recurrent EE, density or E/F checks are missing.')
        if panel.get('density_source_package'):
            density=checks['D_position_density']
            if (density.get('source_package')!=panel['density_source_package'] or
                density.get('source_snapshot_sha256')!=panel.get('density_snapshot_sha256')):
                raise ValueError('Cannot publish Fig4: D density does not match the current frozen density source.')
    return panel


def match_dg_width_to_a(fig):
    """Match the visible left and right edges, including labels and D's scale."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    def edges(key):
        group=plate.GROUPS[key]
        boxes=[a.get_tightbbox(renderer) for a in group['axes'] if a.get_visible()]
        boxes += [t.get_window_extent(renderer) for t in group['texts'] if t.get_visible()]
        box=Bbox.union(boxes)
        return np.array([box.x0,box.x1])/fig.dpi*25.4
    def move(ax,dx=0,dw=0):
        x,y,w,h=np.asarray(ax.get_position().bounds)*[plate.W,plate.H,plate.W,plate.H]
        plate.pos(ax,x+dx,y,w+dw,h)
    target=edges('A');before={key:edges(key).tolist() for key in ('C','F')}
    arrays=[]
    for key in ('C','F'):
        for ax in plate.GROUPS[key]['axes']:
            arrays.extend((a,np.array(a.get_array()).copy()) for a in ax.images)
            arrays.extend((a,np.array(a.get_ydata()).copy()) for a in ax.lines)
    gain,angle,error,position,bar=plate.GROUPS['C']['axes']
    for ax in plate.GROUPS['C']['axes']:move(ax,dx=target[0]-before['C'][0])
    extra=float(np.diff(target)[0]-np.diff(before['C'])[0])
    assert 0<extra<25,extra
    # Allocate the extra room to wider response axes and inter-column spacing;
    # retain square, undistorted XY coordinates and the equal-height colourbar.
    for ax in (gain,error):move(ax,dw=extra/2)
    move(angle,dx=extra-extra/4,dw=extra/2)
    for ax in (position,bar):move(ax,dx=extra)
    waveform=plate.GROUPS['F']['axes'][0]
    move(waveform,dx=target[0]-before['F'][0],dw=float(np.diff(target)[0]-np.diff(before['F'])[0]))
    fig.canvas.draw()
    after={key:edges(key) for key in ('A','C','F')}
    for key in ('C','F'):np.testing.assert_allclose(after[key],target,rtol=0,atol=1e-7)
    for artist,values in arrays:
        actual=artist.get_array() if hasattr(artist,'get_array') else artist.get_ydata()
        np.testing.assert_array_equal(actual,values)
    contact_left=min(t.get_window_extent(renderer).x0 for t in plate.GROUPS['E']['axes'][0].get_yticklabels() if t.get_visible())
    gap=(contact_left-bar.get_tightbbox(renderer).x1)/fig.dpi*25.4
    assert gap>3,gap
    plate.CHECKS['BEFH_alignment']['D_colourbar_to_F_contacts_gap_mm']=gap
    plate.CHECKS['ADG_equal_width']=dict(version='adg_equal_visible_width_v1',
        visible_edges_mm={label:after[key].tolist() for label,key in [('A','A'),('D','C'),('G','F')]},
        common_visible_width_mm=float(np.diff(target)[0]),
        previous_DG_visible_edges_mm={'D':before['C'],'G':before['F']},
        includes_axis_labels_and_D_colourbar=True,fonts_and_heights_unchanged=True,
        density_xy_aspect_unchanged=True,image_and_curve_values_unchanged=True,
        D_response_axis_width_mm=gain.get_position().width*plate.W,
        G_axis_width_mm=waveform.get_position().width*plate.W)


def finish_d_right_column(fig):
    """Align the angle response with the density plus scale; contain its legend."""
    _,angle,_,position,bar=plate.GROUPS['C']['axes']
    original=[(line,np.array(line.get_xdata()).copy(),np.array(line.get_ydata()).copy()) for line in angle.lines]
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    left=position.get_position().x0*plate.W
    right=bar.get_tightbbox(renderer).x1/fig.dpi*25.4
    _,y,_,height=np.asarray(angle.get_position().bounds)*[plate.W,plate.H,plate.W,plate.H]
    plate.pos(angle,left,y,right-left,height)
    handles=position.legend_.legend_handles
    labels=['Search','Outlier','90% mass']
    legend=position.legend(handles,labels,loc='upper right',bbox_to_anchor=(1,1),
        fontsize=plate.FONT['legend'],frameon=False,borderpad=.2,labelspacing=.1,handlelength=.7,
        handletextpad=.2,borderaxespad=.25)
    assert not legend.get_frame_on()
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    lb=legend.get_window_extent(renderer);pb=position.get_window_extent(renderer)
    assert pb.contains(lb.x0,lb.y0) and pb.contains(lb.x1,lb.y1)
    assert not any(lb.overlaps(t.get_window_extent(renderer)) for t in position.texts)
    # Ground-plane XY coordinates are orthographic and axis-aligned. Allow for
    # raster crop rounding and marker radius when checking the fitted contacts.
    rods=plate.read(plate.SOURCE/'D_position_straight_electrodes.json')
    geometry=plate.read(plate.SOURCE/'D_position_electrodes.json')
    xy=np.asarray(rods['rod_sheet_xy'])[[rods['names'].index(n) for n in geometry['contact_names']]]
    margin=.15/25.4*fig.dpi
    padded=lb.padded(margin)
    assert not any(padded.contains(*point) for point in position.transData.transform(xy))
    for line,x,y in original:
        np.testing.assert_array_equal(line.get_xdata(),x);np.testing.assert_array_equal(line.get_ydata(),y)
    axes=plate.GROUPS['C']['axes']
    box=Bbox.union([a.get_tightbbox(renderer) for a in axes])
    np.testing.assert_allclose(np.array([box.x0,box.x1])/fig.dpi*25.4,
        plate.CHECKS['ADG_equal_width']['visible_edges_mm']['A'],atol=1e-7,rtol=0)
    np.testing.assert_allclose(angle.get_position().x1*plate.W,right,atol=1e-7,rtol=0)
    plate.CHECKS['D_position_density'].update(legend_order=labels,legend_inside=True,legend_frame=False,
        legend_line_meaning='Boundary containing 90% of the per-core KDE mass',legend_does_not_cover_contacts_or_core_labels=True)
    plate.CHECKS['D_right_column_layout']=dict(version='angle_matches_density_with_colorbar_v1',
        angle_axis_left_mm=left,angle_axis_right_mm=right,angle_axis_width_mm=right-left,
        width_includes_density_colourbar_and_label=True,density_legend_inside=True,density_legend_frame=False,
        legend_labels=labels,legend_does_not_cover_contacts_or_core_labels=True,
        data_fonts_density_scale_and_other_panels_unchanged=True)


def render():
    current_record=plate.read(DEST/'figure4_panel_registry.json')
    require_current_d(current_record,publication=True)
    if PARAMETER_PREVIEW is None:
        raise ValueError('Resolve PARAMETER_PREVIEW from current_version.json.panel_d before rendering Fig4.')
    baseline_path=DEST/'figures/fig4-complete-layout.png'
    baseline_sha=plate.sha(baseline_path)
    before=np.asarray(Image.open(baseline_path).convert('RGB'))
    draw=base.recovery.draw_modes
    validate=base.validate_new_panel
    frozen=base.frozen_case
    def selected_case():
        if CORE_PREVIEW is None:return frozen()
        return frozen(CORE_PREVIEW/'source/core_count_input_snapshot.json',
                      core_counts=(1,2,3,4),summary_style='mean_std')
    draw_d=base.panel_d
    verify_alignment=plate.verify_alignment
    def verify_with_alignment(fig,**kwargs):
        if ALIGN_EFH_WITH_B:align_efh_with_b(fig)
        if MATCH_DG_WIDTH_TO_A:match_dg_width_to_a(fig)
        if FINISH_D_RIGHT_COLUMN:finish_d_right_column(fig)
        return verify_alignment(fig,**kwargs)
    draw_b=plate.panel_b
    def post_prior_b(fig,rows,cases,**kwargs):
        visible=[r for r in rows if r['stage']>=B_MIN_STAGE]
        excluded=[r for r in rows if r['stage']<B_MIN_STAGE]
        assert len(rows)==226 and len(visible)==194 and len(excluded)==32
        assert {r['stage'] for r in visible}==set(range(6,17))
        assert all(c['candidate'] in {r['candidate'] for r in visible} for c in cases)
        ax=draw_b(fig,visible,cases,**kwargs)
        bar=fig.axes[-1]
        colors=bar._colorbar.cmap.colors[B_MIN_STAGE-1:]
        cmap=plate.matplotlib.colors.ListedColormap(colors,name='post_prior_stage_purples')
        norm=plate.matplotlib.colors.BoundaryNorm(np.arange(5.5,17),len(colors))
        bar._colorbar.update_normal(plate.plt.cm.ScalarMappable(norm=norm,cmap=cmap))
        bar._colorbar.set_ticks([6,10,16])
        ordered=sorted(visible,key=lambda r:(r['stage'],r['candidate']))
        assert len(ax.lines)==len(visible)
        for line,row in zip(ax.lines,ordered):
            np.testing.assert_array_equal(line.get_ydata(),[row[k] for k in plate.base.KEYS])
            np.testing.assert_allclose(line.get_color(),cmap(norm(row['stage'])))
        for name,table in [('B_visible_workpoints.csv',visible),('B_prior_workpoints.csv',excluded)]:
            with (OUT/'source'/name).open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(table)
        panel=dict(version=B_VERSION,included_original_stages=list(range(6,17)),
            excluded_original_stages=list(range(1,6)),visible_workpoints=len(visible),
            prior_workpoints=len(excluded),original_workpoints=len(rows),
            stage_numbers_renumbered=False,prior_interpretation=B_PRIOR_NOTE,
            visible_source='source/B_visible_workpoints.csv',prior_source='source/B_prior_workpoints.csv',
            requested_on='2026-10-09',human_visual_acceptance='PENDING')
        write(OUT/'source/B_display_contract.json',panel)
        plate.CHECKS.update(B_display_contract=panel,B_prior_lines_removed=True,
            B_rendered_values_and_colors_exact=True,B_colorbar_ticks=[6,10,16],
            B_iteration_palette='Original purple stage colors retained for stages 6-16; stages 1-5 omitted.')
        return ax
    def with_legend(ax,summary,**kwargs):
        return draw(ax,summary,legend_loc='upper right')
    def validate_legend(fig,case):
        checks=validate(fig,case)
        ax=plate.GROUPS['I']['axes'][0]  # Source group I is displayed as J.
        legend=ax.get_legend();box=legend.get_window_extent()
        maximum=max(float(np.max(c.get_offsets()[:,1])) for c in ax.collections)
        top=ax.transData.transform((0,maximum))[1]
        assert box.y0>top+3, (box.y0,top)
        assert legend._loc==1
        subject,median=[t.get_window_extent() for t in legend.get_texts()]
        assert subject.y0>median.y1 and abs(subject.x0-median.x0)<1e-7
        checks.update(J_legend_labels=[t.get_text() for t in legend.get_texts()],
                      J_legend_location='upper right',J_legend_clear_of_points=True,
                      J_legend_columns=1,J_legend_rows=2)
        return checks
    titles=dict(base.TITLES,J='逐患者 TA–MTA 与 TB–MTB 模板相似度（右上角 Subject / Median 上下排列图例）')
    titles.update({key:value for key,value in current_record['titles'].items() if key!='J'})
    titles['B']='先验开发后的参数搜索三类误差：194个工作点，原第6–16阶段'
    if CORE_PREVIEW is not None:
        titles['C']='E1146 一至四 core 的最佳训练 loss：均值±标准差'
    if PARAMETER_PREVIEW is not None:
        titles['D']=plate.read(PARAMETER_PREVIEW/'figure4_candidate_registry.json')['titles']['D']
    if DENSITY_PREVIEW is not None:
        titles['D']='三误差参数响应与低J_joint前20%（83/413配置）的核位置密度'
    with patch.object(base,'OUT',OUT),patch.object(base,'TITLES',titles),\
         patch.object(base,'frozen_case',selected_case),\
         patch.object(base,'REFLOW_COLUMNS','column_reflow' in current_record.get('original_panel_checks',{})),\
         patch.object(base.recovery,'draw_modes',with_legend),\
         patch.object(plate,'panel_b',post_prior_b),\
         patch.object(plate,'verify_alignment',verify_with_alignment),\
         patch.object(base.parameters,'DENSITY',DENSITY_PREVIEW or base.parameters.DENSITY),\
         patch.object(base,'panel_d',draw_d),\
         patch.object(base,'validate_new_panel',validate_legend):
        base.main()
    after=np.asarray(Image.open(OUT/'figures/fig4-complete-layout.png').convert('RGB'))
    assert before.shape==after.shape
    h,w=before.shape[:2];outside=np.ones((h,w),bool)
    # B's profile, labels and colourbar; the remaining panel checks below stay exact.
    if DENSITY_PREVIEW is not None:
        # Only the lower-right D map, its axes, labels and colourbar may change.
        outside[int((plate.H-134)/plate.H*h):int((plate.H-68)/plate.H*h)+1,
                int(55/plate.W*w):int(123/plate.W*w)+1]=False
    else:
        outside[int((plate.H-260)/plate.H*h):int((plate.H-185)/plate.H*h)+1,
                int(130/plate.W*w):int(210/plate.W*w)+1]=False
    if ALIGN_EFH_WITH_B:
        outside[int((plate.H-190)/plate.H*h):int((plate.H-68)/plate.H*h)+1,int(120/plate.W*w):]=False
        outside[int((plate.H-70)/plate.H*h):int((plate.H-7)/plate.H*h)+1,
                int(115/plate.W*w):]=False
    if MATCH_DG_WIDTH_TO_A:
        outside[:]=True
        for low,high in ((68,191),(5,70)):
            outside[int((plate.H-high)/plate.H*h):int((plate.H-low)/plate.H*h)+1,
                    :int(135/plate.W*w)+1]=False
    if FINISH_D_RIGHT_COLUMN:
        outside[:]=True
        outside[int((plate.H-191)/plate.H*h):int((plate.H-68)/plate.H*h)+1,
                :int(135/plate.W*w)+1]=False
    changed=int(np.count_nonzero(np.any(before!=after,axis=2)&outside))
    assert changed==0,changed
    record=plate.read(OUT/'figure4_candidate_registry.json')
    parent=plate.read(PARENT/'figure4_candidate_registry.json')
    assert record['candidate_J']['groups']==parent['candidate_J']['groups']
    if CORE_PREVIEW is None:
        assert record['core_count_summary']==parent['core_count_summary']
    else:
        # Preserve every other standalone asset byte-for-byte after checking the
        # native rebuild reproduces their rendered pixels.
        unchanged='aefghij' if PARAMETER_PREVIEW is not None else 'adefghij'
        if ALIGN_EFH_WITH_B:unchanged='ag'
        if MATCH_DG_WIDTH_TO_A:unchanged='aefhij'
        if FINISH_D_RIGHT_COLUMN:unchanged='aefghij'
        if PARAMETER_PREVIEW is not None and current_record.get('panel_c',{}).get('snapshot_sha256')==plate.sha(CORE_PREVIEW/'source/core_count_input_snapshot.json'):
            unchanged+='c'
        for letter in unchanged:
            stem=f'fig4-panel{letter}'
            np.testing.assert_array_equal(np.asarray(Image.open(DEST/'figures'/f'{stem}.png')),
                                          np.asarray(Image.open(OUT/'figures'/f'{stem}.png')))
            for ext in ('png','pdf','svg'):
                shutil.copy2(DEST/'figures'/f'{stem}.{ext}',OUT/'figures'/f'{stem}.{ext}')
        record['outputs']={str(p.relative_to(OUT)):plate.sha(p) for p in (OUT/'figures').iterdir()
                           if p.suffix in ('.png','.pdf','.svg')}
    record.update(status='CURRENT_AUTHOR_DESIGNATED',author_designated_on='2026-09-29',
                  producer=PRODUCER,changes='Use the A-J layout; J has Subject above Median in one legend column at upper right.',
                  parent_package=str(PARENT.relative_to(ROOT)),outside_panel_j_changed_pixels=changed,
                  panel_j=record.pop('candidate_J'))
    record['base_published_complete_sha256']=baseline_sha
    record['source_files'][str(Path(__file__))]=plate.sha(__file__)
    record['checks']['matched_mode_values_unchanged']=True
    recovery_record=plate.read(base.recovery.OUT/'figure4_candidate_registry.json')
    record['panel_j_interpretation']=recovery_record['interpretation']
    record['source_metadata_correction']=recovery_record['source_metadata_correction']
    if CORE_PREVIEW is not None:
        preview=plate.read(CORE_PREVIEW/'preview_registry.json')
        row=next(r for r in preview['comparison'] if r['core_counts']=='1,2,3,4')
        record.update(changes='Update C to four core counts with thin mean lines, small points and sample SD bands.',
                      updated_on='2026-10-08',outside_panel_c_changed_pixels=changed,
                      other_standalone_panels_pixel_identical=True,
                      scientific_scope='E1146 best-so-far training loss over 31 shared parameter evaluations; four optimization restarts per arm; independent-noise confirmation deferred.',
                      panel_c=dict(preview_package=str(CORE_PREVIEW.relative_to(ROOT)),
                          core_counts=[1,2,3,4],summary_style='mean_std',std_ddof=1,
                          statistical_unit='optimization_restart_within_E1146',restarts=4,common_epoch=31,
                          comparison=row,author_accepted_on='2026-10-08',
                          snapshot_sha256=plate.sha(CORE_PREVIEW/'source/core_count_input_snapshot.json')))
        record['checks'].update(C_summary_style='mean_std',C_sample_std_ddof=1,C_restarts=4,
                                C_shading_artist_values_exact=True,C_individual_trajectories=False)
        record.pop('outside_panel_j_changed_pixels',None)
        for name in ('comparison.csv','mean_std_curves.csv'):
            shutil.copy2(CORE_PREVIEW/name,OUT/'source'/f'core_count_{name}')
        record['packaged_rendering_sources']={}
        for source in (Path(__file__),Path(base.__file__),Path(plate.__file__),Path(base.progress.__file__),Path(base.parameters.__file__)):
            target=OUT/'source'/source.name
            shutil.copy2(source,target)
            record['packaged_rendering_sources'][str(target.relative_to(OUT))]=plate.sha(target)
    if PARAMETER_PREVIEW is not None:
        expected_d=plate.read(PARAMETER_PREVIEW/'figure4_candidate_registry.json')
        for name in ('D_connection_response.csv','D_recurrent_EE_response.csv','D_position_selection.json','D_EF_workpoint_references.json'):
            if DENSITY_PREVIEW is not None and name=='D_position_selection.json':continue
            assert plate.sha(OUT/'source'/name)==plate.sha(PARAMETER_PREVIEW/'source'/name),name
        if DENSITY_PREVIEW is None:
            np.testing.assert_array_equal(np.asarray(Image.open(OUT/'figures/fig4-paneld.png')),
                                          np.asarray(Image.open(PARAMETER_PREVIEW/'figures/fig4-paneld.png')))
        for key in ('D_connection_response','D_recurrent_EE','D_position_density','D_EF_workpoint_references','D_parameter_layout'):
            if DENSITY_PREVIEW is not None and key in ('D_position_density','D_parameter_layout'):continue
            assert record['original_panel_checks'][key]==expected_d['original_panel_checks'][key],key
        record.pop('outside_panel_c_changed_pixels',None)
        record.update(changes='Restore the latest D: three-error parameter curves, recurrent EE sweep, position density and executed E/F references; retain the current four-core mean/SD C.',
                      outside_updated_panels_changed_pixels=changed,unchanged_standalone_panels=list(unchanged),
                      panel_d=dict(version=CURRENT_D_VERSION,
                          source_package=str(PARAMETER_PREVIEW.relative_to(ROOT)),requested_on='2026-10-08',
                          source_registry_sha256=plate.sha(PARAMETER_PREVIEW/'figure4_candidate_registry.json'),
                          source_panel_png_sha256=plate.sha(PARAMETER_PREVIEW/'figures/fig4-paneld.png'),
                          source_data_and_panel_pixels_verified=True,column_reflow_promoted=False))
    # This revision changes only B. Verify and preserve all other current panels,
    # including the explicitly routed C/D versions, byte-for-byte.
    preserved='abcefghij' if DENSITY_PREVIEW is not None else 'acdefghij'
    if ALIGN_EFH_WITH_B:preserved='abcg'
    if MATCH_DG_WIDTH_TO_A:preserved='abcefhij'
    if FINISH_D_RIGHT_COLUMN:preserved='abcefghij'
    for letter in preserved:
        for ext in ('png','pdf','svg'):
            source=DEST/'figures'/f'fig4-panel{letter}.{ext}'
            target=OUT/'figures'/source.name
            if ext=='png':
                np.testing.assert_array_equal(np.asarray(Image.open(source)),np.asarray(Image.open(target)))
            shutil.copy2(source,target)
    record['panel_c']=current_record['panel_c']
    record['panel_d']=current_record['panel_d']
    record.update(panel_b=plate.CHECKS['B_display_contract'],updated_on='2026-10-09',
        changes='Omit the 32 lines from original stages 1-5 in B; show 194 configurations from stages 6-16 with their original colors and stage numbers. Preserve prior-development provenance and every other current panel.',
        scientific_scope=B_PRIOR_NOTE,outside_panel_b_changed_pixels=changed,
        unchanged_standalone_panels=list('acdefghij'))
    if DENSITY_PREVIEW is not None:
        record['panel_b']=current_record['panel_b']
        record['panel_d']=dict(current_record['panel_d'],
            density_version='completed_joint_densification_top20_relief_v1',
            density_source_package=str(DENSITY_PREVIEW.relative_to(ROOT)),
            density_snapshot_sha256=plate.sha(DENSITY_PREVIEW/'source/snapshot.json'),
            density_source_png_sha256=plate.sha(DENSITY_PREVIEW/'figures/core-position-search-context.png'),
            density_selected=83,density_scorable=413,density_completed=418,
            density_author_accepted_on='2026-10-09',source_data_and_panel_pixels_verified=False,
            response_curve_sources_verified_unchanged=True)
        record.pop('outside_panel_b_changed_pixels',None)
        record.update(changes='Update only D position density to the completed 128-configuration densification snapshot: 83/413 lowest-J_joint configurations; preserve all other current panels and D response curves.',
            scientific_scope=record['original_panel_checks']['D_position_density']['interpretation'],
            outside_d_density_changed_pixels=changed,unchanged_standalone_panels=list(preserved))
    if ALIGN_EFH_WITH_B:
        record.pop('outside_d_density_changed_pixels',None)
        record['layout_revision']=plate.CHECKS['BEFH_alignment']
        record.update(changes='Promote the completed 83/413 core-position density into D and align E/F/H left data edges and panel letters to B. Compact E/F envelopes and H horizontally; shift I/J right by 4 mm without changing data, fonts or canvas size. Preserve A/B/C/G and D response curves.',
            outside_updated_panels_changed_pixels=changed,other_standalone_panels_pixel_identical=list(preserved))
    if MATCH_DG_WIDTH_TO_A:
        record.update(left_column_layout=plate.CHECKS['ADG_equal_width'],
            changes='Match the complete visible widths and both horizontal edges of D/G to the current A. Widen D response axes and G waveform; preserve square XY density, fonts, data, heights and the current B/E/F/H alignment.',
            outside_dg_changed_pixels=changed)
    if FINISH_D_RIGHT_COLUMN:
        record.update(d_layout_revision=plate.CHECKS['D_right_column_layout'],outside_panel_d_changed_pixels=changed,
            changes='Remove the density legend frame in D while keeping its in-map position, labels and font. Preserve the right-column alignment, A/D/G widths, all data and the other nine panels.')
    record['outputs']={str(p.relative_to(OUT)):plate.sha(p) for p in (OUT/'figures').iterdir()
                       if p.suffix in ('.png','.pdf','.svg')}
    write(OUT/'figure4_candidate_registry.json',record)
    qa=plate.read(OUT/'visual_qa.json')
    qa.update(status='LAYOUT_AND_VALUE_CHECKS_PASS_VISUAL_PENDING',outside_panel_j_changed_pixels=changed,
              matched_mode_values_unchanged=True,author_designated_on='2026-09-29')
    if CORE_PREVIEW is not None:
        qa.pop('outside_panel_j_changed_pixels',None)
        qa.update(outside_panel_c_changed_pixels=changed,other_standalone_panels_pixel_identical=True,
                  panel_c_author_accepted_on='2026-10-08',panel_c_summary='mean +/- sample SD; n=4; ddof=1')
        qa['checks']=record['checks']
    if PARAMETER_PREVIEW is not None:
        qa.pop('outside_panel_c_changed_pixels',None)
        qa.update(outside_updated_panels_changed_pixels=changed,unchanged_standalone_panels=list(unchanged),
                  D_matches_latest_candidate_pixels=True,D_source_tables_and_checks_match=True)
    qa.update(outside_panel_b_changed_pixels=changed,unchanged_standalone_panels=list('acdefghij'),
              panel_b=record['panel_b'],human_visual_acceptance='PENDING')
    if DENSITY_PREVIEW is not None:
        qa.pop('outside_panel_b_changed_pixels',None)
        qa.pop('D_matches_latest_candidate_pixels',None)
        qa.update(outside_d_density_changed_pixels=changed,unchanged_standalone_panels=list(preserved),
                  D_source_tables_and_checks_match='Three response curves and E/F references match the previous formal D; density matches the author-selected completed snapshot.',
                  density_source_author_accepted_on='2026-10-09')
    if ALIGN_EFH_WITH_B:
        qa.pop('outside_d_density_changed_pixels',None)
        qa.update(layout_revision=record['layout_revision'],outside_updated_panels_changed_pixels=changed,
                  other_standalone_panels_pixel_identical=list(preserved))
    if MATCH_DG_WIDTH_TO_A:
        qa.update(left_column_layout=record['left_column_layout'],outside_dg_changed_pixels=changed)
    if FINISH_D_RIGHT_COLUMN:
        qa.update(d_layout_revision=record['d_layout_revision'],outside_panel_d_changed_pixels=changed)
    write(OUT/'visual_qa.json',qa)
    mapping=(OUT/'panel_map.md').read_text().replace('# Figure 4：新 C 候选（A–J）','# 当前 Figure 4：A–J')
    mapping=mapping.replace('状态：候选，待作者目视检查；当前指定 A–I 的文件与机器入口保持不变。',
        '状态：CURRENT_AUTHOR_DESIGNATED。用户确认当前编号为A–J；J在图内右上角以一列两行显示Subject / Median图例。')
    mapping=mapping.replace('候选编号','当前编号').replace('build_fig4_core_count_candidate.py','build_fig4_current_aj.py')
    if (DEST/'panel_map.md').exists():
        mapping=(DEST/'panel_map.md').read_text().replace('右上角加入Subject / Median图例','右上角以一列两行显示Subject / Median图例')
        mapping='\n'.join(f'| J | {titles["J"]} |' if line.startswith('| J |') else line for line in mapping.split('\n'))
    mapping+='\nJ每个点代表一位患者，横线为中位数，竖段及端帽为四分位区间；25位患者的两项相似度及工作点均未改变。I为原触点分折交叉匹配，J为完整模板描述性相似度。\n'
    if CORE_PREVIEW is not None:
        mapping='\n'.join(f'| C | {titles["C"]} |' if line.startswith('| C |') else
            'C 的轴框为51 × 45 mm，与B纵轴上下边界一致；使用Epochs／Loss标签、四种颜色及小点细线，无标题或图注。' if line.startswith('C 的轴框') else
            'C 使用2026-10-08作者认可的数据快照：每组4次优化重复，共同前31个epoch。线为各次最佳训练loss的算术均值，阴影为均值±样本标准差（ddof=1）。3／4 core的两轮反馈优化阶段已完成；独立噪声复测暂缓。' if line.startswith('数据冻结于') else line
            for line in mapping.split('\n'))
    if PARAMETER_PREVIEW is not None:
        mapping='\n'.join(f'| D | {titles["D"]} |' if line.startswith('| D |') else line
                          for line in mapping.split('\n') if not line.startswith('D沿用当前正式图：'))
        mapping=mapping.replace('原 A 及原 C–I 的内容、数据和排版保持','A、B及E–J的内容、数据和排版保持')
        details=[p for p in (PARAMETER_PREVIEW/'panel_map.md').read_text().split('\n\n') if p.startswith('D ')]
        for paragraph in details:
            if paragraph not in mapping:mapping+='\n\n'+paragraph
    mapping='\n'.join(f'| B | {titles["B"]} |' if line.startswith('| B |') else line
                      for line in mapping.split('\n'))
    mapping+='\n\nB仅展示原第6–16阶段的194个工作点；第1–5阶段32个配置作为此前标签辅助的开发先验保留在来源表。后续目标不使用TA/TB标签，但继承的候选池与患者几何属于开发先验，本图不表示与开发数据独立。原阶段编号和各条保留曲线的颜色、误差数值及E/F示例不变。\n'
    if DENSITY_PREVIEW is not None:
        mapping='\n'.join(line for line in mapping.split('\n') if not line.startswith('D 右下'))
        mapping+='\nD 右下使用2026-10-09完成补充实验的冻结快照：418个完成配置、413个可评分，选取J_joint最低20%的83个配置（59个历史、24个新增），高斯核宽0.35 mm。40 × 40 mm地图与竖向色条等高；直杆和触点叠加于曲面之上，灰点为搜索位置、空心点为90%密度轮廓外的入选位置。密度体现已评估搜索池的集中性，受局部加密设计影响；存在其他低损失区域，不代表独立优化唯一收敛或参数后验。\n'
    (OUT/'panel_map.md').write_text(mapping)
    if ALIGN_EFH_WITH_B:
        mapping+='\n按用户要求，B/E/F/H绘图区左边界统一148 mm，角标统一137 mm；E/F的接触时程各宽18 mm，原生空间快照大小不变。H宽23 mm，I/J右移4 mm，字体、数据与画布尺寸不变，为D右侧新增18 mm间距；A/B/C/G保持原样。\n'
        (OUT/'panel_map.md').write_text(mapping)
    if MATCH_DG_WIDTH_TO_A:
        width=record['left_column_layout']['common_visible_width_mm']
        mapping+=f'\n2026-10-09继续按用户要求将A/D/G的整体可见左右边界对齐，统一宽度为{width:.3f} mm（计入轴标签和D色条，不计panel角标）。D响应曲线与G时程横向展开，位置图保留40 × 40 mm和等高色条；字体、数据、行高及B/E/F/H纵向对齐保留。\n'
        (OUT/'panel_map.md').write_text(mapping)
    if FINISH_D_RIGHT_COLUMN:
        mapping+='\nD右上E→E角度响应轴的左边界与位置地图对齐，右边界与下方色条及其标签的整体右边界对齐。位置图图例全部移回图内右上角，保持Search、Outlier两点在前、90% mass轮廓线在后；90% mass表示逐core KDE的90%概率质量轮廓。图例避开15个触点与Core A/B标签，字号、数据、曲面及A/D/G整体宽度不变。\n'
        (OUT/'panel_map.md').write_text(mapping)
    p=OUT/'figures/README.md'
    notes=p.read_text().replace('候选待作者目视检查。','本版按用户指定作为当前A–J版；J图例在右上角。').replace('候选编号','当前编号')
    if (DEST/'figures/README.md').exists():
        baseline_notes=(DEST/'figures/README.md').read_text()
        j_section=next(block for block in notes.split('### ') if block.startswith('fig4-panelj.png'))
        notes='### '.join(j_section if block.startswith('fig4-panelj.png') else block
                         for block in baseline_notes.split('### '))
    p.write_text(notes)

    if CORE_PREVIEW is not None:
        block=('fig4-panelc.png\nE1146一至四core的最佳训练loss；每组4次优化重复，展示共同前31个epoch。'
               '细线和小点为均值，色带为均值±样本标准差（ddof=1），另有同源PDF/SVG。\n'
               '**关注点**：用户已确认该四组预览并要求放入C；训练loss比较不等于独立噪声确认。\n\n')
        notes='### '.join(block if part.startswith('fig4-panelc.png') else part for part in notes.split('### '))
        notes=notes.replace('新 C 单核／双核 loss 曲线','C 一至四 core 均值±标准差 loss 曲线')
        p.write_text(notes)
    if PARAMETER_PREVIEW is not None:
        block=('fig4-paneld.png\n向外E→E、E→E方向和核内E→E的三种误差响应，共用单一Error轴。'
               '左下为核内E→E七点扫描，右下为低J_joint前20%配置的核位置密度；E/F虚线对应传播示例实际参数。'
               '另有同源PDF/SVG。\n**关注点**：位置密度不是误差或参数后验；扫描背景不同，虚线交点不代表E/F的实际误差。\n\n')
        notes='### '.join(block if part.startswith('fig4-paneld.png') else part for part in notes.split('### '))
        p.write_text(notes)
    block=('fig4-panelb.png\n展示先验开发后的194个参数配置在Mean rank、Order和Participation上的误差；'
           '保留原第6–16阶段编号和对应紫色，移除第1–5阶段32条线。E/F仍指向相同传播示例，另有同源PDF/SVG。\n'
           '**关注点**：早期标签辅助开发结果作为先验来源保留；后续评分不使用TA/TB标签，但继承的候选与几何不构成独立验证。\n\n')
    notes='### '.join(block if part.startswith('fig4-panelb.png') else part for part in notes.split('### '))
    if DENSITY_PREVIEW is not None:
        block=('fig4-paneld.png\n三张参数响应曲线及E/F参考线保持不变，右下更新为413个可评分配置中低J_joint前20%的83个配置的位置密度。'
               '使用0.35 mm高斯核、单色立体曲面、90%密度轮廓和直电极杆，触点叠加显示；另有同源PDF/SVG。\n'
               '**关注点**：灰点为已搜索位置，Outlier为空心的轮廓外入选位置；存在其他低损失位置，密度受局部加密影响，不代表唯一收敛或参数后验。主图排版待作者目视检查。\n\n')
        notes='### '.join(block if part.startswith('fig4-paneld.png') else part for part in notes.split('### '))
    if MATCH_DG_WIDTH_TO_A:
        width=record['left_column_layout']['common_visible_width_mm']
        for name in ('fig4-paneld.png','fig4-panelg.png'):
            sentence=f'整体可见宽度与A对齐至{width:.3f} mm，字体、数据和轴高保持不变。'
            notes='### '.join(part.replace('\n**关注点**：',sentence+'\n**关注点**：',1)
                if part.startswith(name) and sentence not in part else part for part in notes.split('### '))
    if FINISH_D_RIGHT_COLUMN:
        sentence='右上角度响应图与下方位置图连同色条等宽，位置图图例完全位于图内、无边框和底框，且避开触点；90% mass为逐core密度的90%质量轮廓。'
        notes='### '.join(part.replace('\n**关注点**：',sentence+'\n**关注点**：',1)
            if part.startswith('fig4-paneld.png') and sentence not in part else part for part in notes.split('### '))
    p.write_text(notes)


def publish():
    record=plate.read(OUT/'figure4_candidate_registry.json')
    current=plate.read(DEST/'current_version.json')
    require_current_d(current)
    require_current_d(record,publication=True)
    assert current['producer'] in [PRODUCER,'scripts/paper_figures/build_fig4_template_recovery_by_mode.py']
    expected=record.get('base_published_complete_sha256')
    current_image=plate.sha(DEST/'figures/fig4-complete-layout.png')
    assert expected is None or current_image in [expected,record['outputs']['figures/fig4-complete-layout.png']], 'Published figure changed during rendering; rebuild against the new version.'
    for name,digest in record['outputs'].items():assert plate.sha(OUT/name)==digest,name
    if not ARCHIVE.exists():
        ARCHIVE.mkdir(parents=True)
        for directory in ['figures','source']:
            shutil.copytree(DEST/directory,ARCHIVE/directory)
        for path in DEST.iterdir():
            if path.is_file():shutil.copy2(path,ARCHIVE/path.name)
    shutil.copytree(OUT/'figures',DEST/'figures',dirs_exist_ok=True)
    shutil.copytree(OUT/'source',DEST/'source',dirs_exist_ok=True)
    shutil.copy2(OUT/'panel_map.md',DEST/'panel_map.md')
    record.update(published_from=str(OUT.relative_to(ROOT)),historical_package=str(ARCHIVE.relative_to(ROOT)),
        current_package_and_pointer_unchanged=False,
        panels={s:[str((DEST/'figures'/f'fig4-panel{s}.{ext}').relative_to(ROOT))
                   for ext in ['png','pdf','svg']] for s in 'abcdefghij'})
    for name in ['figure4_panel_registry.json','figure4_candidate_registry.json']:write(DEST/name,record)
    qa=plate.read(OUT/'visual_qa.json')
    hashes={str((DEST/name).relative_to(ROOT)):digest for name,digest in record['outputs'].items()}
    for name,digest in hashes.items():assert plate.sha(ROOT/name)==digest
    qa.update(publication_bytes_identical=True,outputs_sha256=hashes,current_package_and_pointer_unchanged=False)
    write(DEST/'visual_qa.json',qa)
    (DEST/'README.md').write_text(
        '# 当前 Figure 4：A–J\n\n当前图已正式放入`figures/`。C为E1146单核／双核训练loss曲线；'
        'I为交叉匹配矩阵，J为25位患者的TA–MTA / TB–MTB相似度，图内右上角为一列两行的Subject / Median图例。\n\n'
        '[Fig4J](figures/fig4-panelj.png) · [完整PDF](figures/fig4-complete-layout.pdf) · '
        '[完整预览](figures/fig4-complete-layout-preview.png) · [编号说明](panel_map.md)\n\n'
        '点为患者，横线为中位数，竖段与端帽为四分位区间。PNG/PDF/SVG同源导出，科学数据保持冻结快照。'
        '生成入口：`python scripts/paper_figures/build_fig4_current_aj.py`。\n\n'
        '上一版A–I保留在[归档](../archive/2026-09-29_pre_aj_legend_fig4/fig4/README.md)。\n')
    if CORE_PREVIEW is not None:
        p=DEST/'README.md'
        text=p.read_text().replace('C为E1146单核／双核训练loss曲线',
            'C为E1146一至四core训练loss曲线，细线小点为4次优化重复的均值，阴影为±样本标准差，共同前31个epoch')
        text+=f'\n2026-10-08按用户要求更新C；更新前完整包保留在 `{ARCHIVE.relative_to(ROOT)}`。\n'
        p.write_text(text)
    if PARAMETER_PREVIEW is not None:
        p=DEST/'README.md'
        text=p.read_text().replace('2026-10-08按用户要求更新C；','2026-10-08按用户要求更新C，并纠正D的版本入口；')
        text+='\nD采用三误差曲线、核内E→E七点扫描、低损失核位置密度和E/F实际参数虚线；详见[独立D](figures/fig4-paneld.pdf)。\n'
        p.write_text(text)
    panel_j=current.pop('panel_i',current.get('panel_j',{}))
    panel_j.update(legend_present=True,legend_location='upper right',legend_labels=['Subject','Median'],
                   legend_columns=1,legend_rows=2)
    palette=current['palette'];palette.pop('I',None)
    palette['J']=dict(TA_MTA='#C63D3A',TB_MTB='#287FA1',legend_present=True)
    current.update(panel_letters=list('ABCDEFGHIJ'),asset_id='patient_geometry_prior_snn_complete_a_j',
        producer=PRODUCER,published_from=str(OUT.relative_to(ROOT)),outputs_sha256=hashes,
        parent_package=str(PARENT.relative_to(ROOT)),panel_j=panel_j,palette=palette,
        previous_version_archive=str(ARCHIVE.relative_to(ROOT)))
    if CORE_PREVIEW is not None:
        current.update(updated_on='2026-10-08',panel_c=record['panel_c'])
        current['palette']['C_core_counts']={str(k):v for k,v in base.progress.COLORS.items()}
    if PARAMETER_PREVIEW is not None:
        current['panel_d']=record['panel_d']
    current.update(updated_on='2026-10-09',panel_b=record['panel_b'])
    if ALIGN_EFH_WITH_B:current['layout_revision']=record['layout_revision']
    if MATCH_DG_WIDTH_TO_A:current['left_column_layout']=record['left_column_layout']
    if FINISH_D_RIGHT_COLUMN:current['d_layout_revision']=record['d_layout_revision']
    current['palette']['B_iterations']=record['original_panel_checks']['B_iteration_palette']
    p=DEST/'README.md'
    p.write_text(p.read_text()+'\n2026-10-09更新B：仅保留原第6–16阶段194条线；第1–5阶段为标签辅助开发先验，历史来源保留，后续评分不使用TA/TB标签。本次仅改变B展示，其余A、C–J保持原版本。新版待作者目视检查。\n')
    if DENSITY_PREVIEW is not None:
        text=p.read_text().replace('本次仅改变B展示，其余A、C–J保持原版本。','此前更新仅改变B展示。')
        p.write_text(text+'\n2026-10-09按用户确认将补充实验的位置密度接入D：418个完成配置中413个可评分，低J_joint前20%=83个；其余A/B/C/E–J及D三张响应曲线保持不变。直杆、触点覆盖、单色立体曲面和竖向色条同步更新，主图排版待作者目视检查。\n')
    if ALIGN_EFH_WITH_B:
        p.write_text(p.read_text().replace('其余A/B/C/E–J及D三张响应曲线保持不变','数据及D三张响应曲线保持不变')+
            '\n同次按用户要求将E/F/H与B左边界对齐，压紧右侧子图间距，给D新增18 mm空间；I/J右移4 mm，字体不变。A/B/C/G独立文件保持逐字节一致。\n')
    if MATCH_DG_WIDTH_TO_A:
        p.write_text(p.read_text()+f'\n最新调整：A/D/G整体可见宽度统一为{record["left_column_layout"]["common_visible_width_mm"]:.3f} mm，左右边界对齐。只改变D/G的横向排版；A/B/C/E/F/H/I/J独立PNG/PDF/SVG保持逐字节一致，前述B/E/F/H对齐和全部科学数据保留。待作者检查新版排版。\n')
    if FINISH_D_RIGHT_COLUMN:
        p.write_text(p.read_text()+'\n本次D细调：位置图图例取消边框和底框，保留图内右上角的位置、字号及条目。右上角度响应图与位置图连同色条等宽，其他九个独立面板逐字节不变，待作者检查本次排版。\n')
    path=DEST/'current_version.json';temporary=path.with_suffix('.json.tmp')
    write(temporary,current);temporary.replace(path)
    print(DEST/'figures'/('fig4-paneld.png' if PARAMETER_PREVIEW is not None else 'fig4-panelc.png' if CORE_PREVIEW is not None else 'fig4-panelj.png'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--render-only',action='store_true')
    mode.add_argument('--publish-only',action='store_true')
    parser.add_argument('--core-count-preview',type=Path,
                        help='Author-selected four-core mean/SD preview package to use in C.')
    parser.add_argument('--parameter-preview',type=Path,
                        help='Use the latest three-error response and position-density D from this package.')
    parser.add_argument('--density-preview',type=Path,
                        help='Author-selected completed core-position snapshot for the density inset in D.')
    parser.add_argument('--align-efh-with-b',action='store_true',help='Align E/F/H with the current B and open space beside D.')
    parser.add_argument('--match-dg-width-to-a',action='store_true',help='Match D/G visible width and left/right edges to A.')
    parser.add_argument('--finish-d-right-column',action='store_true',help='Align D right-column widths and move the density legend inside.')
    args=parser.parse_args()
    pointer=plate.read(DEST/'current_version.json')
    current_d=require_current_d(pointer)
    registered_d=require_current_d(plate.read(DEST/'figure4_panel_registry.json'),publication=True)
    if current_d!=registered_d:
        raise ValueError('Fig4 panel_d differs between current_version.json and the registry; repair the version records first.')
    selected=args.core_count_preview or pointer.get('panel_c',{}).get('preview_package')
    if selected:
        CORE_PREVIEW=(ROOT/selected).resolve()
        OUT=ROOT/'results/paper-ready-figure/fig4/candidates/complete_aj_core_mean_sd_20261008'
        ARCHIVE=ROOT/'results/paper-ready-figure/archive/2026-10-08_pre_core_mean_sd_fig4/fig4'
    parameters=args.parameter_preview or current_d['source_package']
    if parameters:
        PARAMETER_PREVIEW=(ROOT/parameters).resolve()
        OUT=ROOT/'results/paper-ready-figure/fig4/candidates/complete_aj_latest_d_20261008'
        ARCHIVE=ROOT/'results/paper-ready-figure/archive/2026-10-08_pre_latest_d_fig4/fig4'
    OUT=ROOT/'results/paper-ready-figure/fig4/candidates/complete_aj_post_prior_b_20261009'
    ARCHIVE=ROOT/'results/paper-ready-figure/archive/2026-10-09_pre_post_prior_b_fig4/fig4'
    density=args.density_preview or current_d.get('density_source_package')
    if density:
        DENSITY_PREVIEW=(ROOT/density).resolve()
        OUT=ROOT/'results/paper-ready-figure/fig4/candidates/complete_aj_completed_density_d_20261009'
        ARCHIVE=ROOT/'results/paper-ready-figure/archive/2026-10-09_pre_completed_density_d_fig4/fig4'
    ALIGN_EFH_WITH_B=args.align_efh_with_b or pointer.get('layout_revision',{}).get('version')=='efh_aligned_to_b_v1'
    if ALIGN_EFH_WITH_B:
        OUT=ROOT/'results/paper-ready-figure/fig4/candidates/complete_aj_density_aligned_20261009'
        ARCHIVE=ROOT/'results/paper-ready-figure/archive/2026-10-09_pre_density_aligned_fig4/fig4'
    MATCH_DG_WIDTH_TO_A=args.match_dg_width_to_a or pointer.get('left_column_layout',{}).get('version')=='adg_equal_visible_width_v1'
    if MATCH_DG_WIDTH_TO_A:
        if not ALIGN_EFH_WITH_B:raise ValueError('Preserve the current B/E/F/H alignment when matching A/D/G widths.')
        OUT=ROOT/'results/paper-ready-figure/fig4/candidates/complete_aj_adg_width_20261009'
        ARCHIVE=ROOT/'results/paper-ready-figure/archive/2026-10-09_pre_adg_width_fig4/fig4'
    FINISH_D_RIGHT_COLUMN=args.finish_d_right_column or pointer.get('d_layout_revision',{}).get('version')=='angle_matches_density_with_colorbar_v1'
    if FINISH_D_RIGHT_COLUMN:
        if not MATCH_DG_WIDTH_TO_A:raise ValueError('Preserve A/D/G widths before adjusting D right-column layout.')
        OUT=ROOT/'results/paper-ready-figure/fig4/candidates/complete_aj_d_frameless_20261009'
        ARCHIVE=ROOT/'results/paper-ready-figure/archive/2026-10-09_pre_d_frameless_fig4/fig4'
    if not args.publish_only:render()
    if not args.render_only:publish()
