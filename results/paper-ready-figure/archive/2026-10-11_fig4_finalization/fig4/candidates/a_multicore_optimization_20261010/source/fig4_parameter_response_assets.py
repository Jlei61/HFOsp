"""Frozen position-density and within-core EE observations for Fig4 D."""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DENSITY=ROOT/'results/paper-ready-figure/fig4/candidates/best_joint_positions_top20_paper_style_20260929'
EE=Path('/data/hfosp/topic4_sef_hfo/cohort_recurrent_expansion_20260927/subjects/epilepsiae_1146')
METRICS=('mean_rank_error','within_shaft_order_error','participation_error')
ERROR_LABELS=('Mean rank','Order','Participation')
REFERENCE_PARAMETERS=('EE_core_to_out_scale','EE_angle_offset_deg','EE_same_core_scale')


def workpoint_references(plate,cases):
    """Read E/F coordinates from the designs actually used by the examples."""
    verified={}
    for label,case in cases.items():
        path=Path(case['source'])
        raw=plate.read(plate.freeze(path))
        assert plate.sha(path)==case['source_sha256']
        design_path=path.parent.parent/'design.json'
        applied_path=path.parent.parent/'applied_physics.json'
        design=plate.read(plate.freeze(design_path))
        applied=plate.read(plate.freeze(applied_path))
        assert plate.sha(design_path)==raw['design_sha256']
        assert plate.sha(applied_path)==raw['applied_physics_sha256']
        candidate=design['candidate'];params=candidate['parameters']
        assert candidate['id']==raw['job']['candidate']==case['candidate']
        assert candidate==applied['candidate']
        vector=json.loads(case['vector'])
        np.testing.assert_array_equal(candidate['centers_mm'],np.array(vector[:4]).reshape(2,2))
        np.testing.assert_array_equal([params[k] for k in REFERENCE_PARAMETERS[:2]],vector[4:])
        factors=applied['graph']['stage_audits']['weights']['factors']
        assert factors['EE_same_core_scale']==params['EE_same_core_scale']
        assert factors['EE_core_to_out_scale']==params['EE_core_to_out_scale']
        assert applied['graph']['kernel']['angle_offset_deg']==params['EE_angle_offset_deg']
        verified[label]=dict(candidate=case['candidate'],parameters={k:params[k] for k in REFERENCE_PARAMETERS},
                             centers_mm=candidate['centers_mm'],source=str(path),
                             design_sha256=raw['design_sha256'],applied_physics_sha256=raw['applied_physics_sha256'])
    references={}
    for key in REFERENCE_PARAMETERS:
        positions={}
        for label,row in verified.items():positions.setdefault(row['parameters'][key],[]).append(label)
        references[key]=[('/'.join(labels),value) for value,labels in positions.items()]
    metadata=dict(cases=verified,reference_lines=references,
        meaning='Coordinates of the E/F example parameter vectors, not the error values where lines cross a fixed-background sweep.',
        original_unlabelled_lines='Fixed sweep references (outward EE 1.25, angle 0 degrees, core EE 0.85); not separate E/F workpoints.',
        verified_against_executed_design_and_applied_physics=True)
    (plate.SOURCE/'D_EF_workpoint_references.json').write_text(json.dumps(metadata,indent=2)+'\n')
    plate.CHECKS['D_EF_workpoint_references']=metadata
    return references


def error_curves(plate,ax,x,values,xlabel,xticks,references):
    """Use one numeric Error scale for all three contact observations."""
    plate.axis_type(ax)
    for key,label,marker,ys in zip(plate.base.KEYS,ERROR_LABELS,['o','s','^'],values):
        line,=ax.plot(x,ys,color=plate.ERROR_COLORS[key],marker=marker,lw=1,ms=2.5,label=label)
        np.testing.assert_array_equal(line.get_xdata(),x)
        np.testing.assert_array_equal(line.get_ydata(),ys)
    for label,value in references:
        ax.axvline(value,color='#777777',ls='--' if label=='E' else ':',lw=.8,zorder=1)
        ax.text(value,.03,label,transform=ax.get_xaxis_transform(),ha='center',va='bottom',
                fontsize=plate.FONT['legend'],weight='bold',color='#444444',
                bbox=dict(facecolor='white',edgecolor='none',pad=.5),zorder=4)
    ax.set(ylim=(0,.42),yticks=[0,.2,.4],xticks=xticks,xlabel=xlabel,ylabel='Error')
    ax.tick_params(axis='y',left=True,labelleft=True,right=False,labelright=False,pad=1)
    ax.spines['right'].set_visible(False)
    ax.xaxis.label.set_fontsize(plate.MIDDLE_LABEL_PT)
    ax.yaxis.label.set_fontsize(plate.MIDDLE_LABEL_PT)
    ax.xaxis.labelpad=ax.yaxis.labelpad=2
    ax.legend(loc='upper right',fontsize=plate.FONT['legend'],frameon=True,framealpha=1,
              facecolor='white',edgecolor='none',borderpad=.2,labelspacing=.15,
              handlelength=1.1,handletextpad=.35,borderaxespad=.15)


def recurrent_data(plate):
    """Audit the seven already-completed, otherwise identical EE probe runs."""
    folder=ROOT/'scripts/topic4_cohort_recurrent_expansion'
    common=plate.module('fig4_recurrent_common',folder/'common.py')
    with patch.dict(sys.modules,{'common':common}):
        objective=plate.module('fig4_recurrent_objective',folder/'objective.py')
    target_path=plate.freeze(EE/'fit_summary.pkl')
    with target_path.open('rb') as handle:target=pickle.load(handle)
    configs=[plate.read(plate.freeze(EE/f'candidates/initial_{i:02d}.json')) for i in range(7)]
    reference=configs[0]
    results=[plate.read(plate.freeze(EE/f'units/initial_{i:02d}/result.json')) for i in range(7)]
    rows=[];audit=[]
    for i,(config,result) in enumerate(zip(configs,results)):
        cid=f'initial_{i:02d}'
        score_path=plate.freeze(EE/f'scores/{cid}.json');score=plate.read(score_path)
        assert config['role'] in ('unchanged_previous_training_nominee','paired_recurrent_EE_probe')
        assert config['centers_mm']==reference['centers_mm']
        assert {k:v for k,v in config['parameters'].items() if k!='EE_same_core_scale'}=={
            k:v for k,v in reference['parameters'].items() if k!='EE_same_core_scale'}
        for key in ('radii_mm','core_mean_rate_scale','core_ou_correlation','shape','outgoing'):
            assert config[key]==reference[key]
        assert result['physical_status']=='COMPLETE_NO_RUNAWAY' and result['actual_duration_ms']==60000
        assert result['job']['topology_seed']==2511 and result['job']['dynamics_seed']==847401
        for key,value in result['static_identity'].items():
            if key!='ampa_values_sha256':assert value==results[0]['static_identity'][key],key
        raw_path=plate.freeze(EE/f'units/{cid}/trajectory.npz')
        assert plate.sha(raw_path)==score['raw_sha256']
        with np.load(raw_path,allow_pickle=False) as raw:
            names=raw['contact_names'].tolist();times=raw['centroid_ms'][raw['primary_event_indices']]
        recomputed=objective.descriptive(times,target,names)
        assert len(times)==score['N'] and recomputed['full_contact_pair_support']
        np.testing.assert_allclose([recomputed[k] for k in METRICS],
                                   [score['descriptive_fit'][k] for k in METRICS],rtol=0,atol=1e-12)
        rows.append(dict(candidate=cid,EE_same_core_scale=config['parameters']['EE_same_core_scale'],
                         N=score['N'],**{k:recomputed[k] for k in METRICS}))
        audit.append(dict(candidate=config,score=score,result=result))
    rows.sort(key=lambda r:r['EE_same_core_scale'])
    with (plate.SOURCE/'D_recurrent_EE_response.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (plate.SOURCE/'D_recurrent_EE_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    plate.CHECKS['D_recurrent_EE']=dict(subject='epilepsiae_1146',n_conditions=7,
        factors=[r['EE_same_core_scale'] for r in rows],metrics=METRICS,
        metrics_recomputed_from_primary_event_times=True,all_contact_pairs_supported=True,
        fixed_geometry_topology_noise_and_other_parameters=True,topology=2511,noise=847401,
        duration_ms=60000,background='Previous cohort training nominee; distinct from the upper parameter sweeps',
        statistical_unit='One fixed-network/noise simulation per parameter condition; descriptive FIT errors')
    return rows


def position_data(plate):
    snapshot=plate.read(plate.freeze(DENSITY/'source/snapshot.json'))
    registry=plate.read(plate.freeze(DENSITY/'figure_registry.json'))
    completed=(DENSITY/'source/core-position-search-context.npz').exists()
    density_path=plate.freeze(DENSITY/('source/core-position-search-context.npz' if completed else 'source/core-position-best-joint.npz'))
    geometry_path=plate.freeze(DENSITY/'source/electrode_geometry.json')
    with np.load(density_path) as z:arrays={k:z[k] for k in z.files}
    geometry=plate.read(geometry_path)
    selection=snapshot['selections']['0.2']
    assert snapshot['main_fraction']==.2
    points=np.array([r['centers_mm'] for r in snapshot['records'] if r['candidate'] in selection['candidate_ids']])
    xx,yy=np.meshgrid(arrays['x_mm'],arrays['y_mm'])
    sigma=snapshot['sigma_mm'];assert sigma==.35
    parts=np.array([sum(np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma**2)) for x,y in points[:,core])/
                    (len(points)*2*np.pi*sigma**2)/2 for core in range(2)])
    if completed:
        np.testing.assert_array_equal(points,arrays['selected_positions_mm'])
        assert len(points)==selection['count']==83 and snapshot['pool_scorable']==413
        np.testing.assert_allclose(parts*2,arrays['per_core_density_per_mm2'],rtol=0,atol=1e-12)
        arrays['core_components_per_mm2']=parts
        straight_path=plate.freeze(DENSITY/'source/electrode_straight_display.json')
        straight=plate.read(straight_path)
        indices=[straight['names'].index(name) for name in geometry['contact_names']]
        np.testing.assert_allclose(np.asarray(straight['measured_sheet_xy'])[indices],geometry['contact_xy_mm'],atol=1e-6,rtol=0)
        geometry['straight_contact_xy_mm']=np.asarray(straight['rod_sheet_xy'])[indices].tolist()
        contours_path=plate.freeze(DENSITY/'density_contours.json')
        geometry['density_contours']=plate.read(contours_path)['contours']
        shutil.copyfile(straight_path,plate.SOURCE/'D_position_straight_electrodes.json')
        shutil.copyfile(contours_path,plate.SOURCE/'D_position_contours.json')
    else:
        assert registry['main_fraction']==.2 and selection['count']==52
        np.testing.assert_allclose(parts,arrays['core_components_per_mm2'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(parts.sum(0),arrays['density_per_mm2'],rtol=0,atol=1e-12)
    mass=arrays['density_per_mm2'].sum()*np.diff(arrays['x_mm'])[0]*np.diff(arrays['y_mm'])[0]
    np.testing.assert_allclose(mass,1,rtol=0,atol=2e-6)
    shutil.copyfile(density_path,plate.SOURCE/'D_position_density.npz')
    shutil.copyfile(geometry_path,plate.SOURCE/'D_position_electrodes.json')
    metadata=dict(selection=selection,pool_scorable=snapshot['pool_scorable'],asof=snapshot['asof'],
                  sigma_mm=sigma,density_integral=float(mass),selection_metric='J_joint',unit='mm^-2',
                  interpretation='Position density conditional on the lowest 20% J_joint; not error magnitude or a posterior')
    if completed:
        metadata.update(source_package=str(DENSITY.relative_to(ROOT)),source_snapshot_sha256=plate.sha(DENSITY/'source/snapshot.json'),
            pool_completed=snapshot['pool_completed'],new_completed=snapshot['new_completed'],new_scorable=snapshot['new_scorable'],
            straight_electrodes=True,contacts_overlay_above_surface=True,
            interpretation='Descriptive density of the lowest 20% J_joint in the evaluated search pool; deliberate local densification affects concentration. Secondary good regions remain; not unique convergence or a posterior.')
        shutil.copyfile(DENSITY/'source/snapshot.json',plate.SOURCE/'D_position_snapshot.json')
    (plate.SOURCE/'D_position_selection.json').write_text(json.dumps(metadata,indent=2)+'\n')
    plate.CHECKS['D_position_density']=metadata
    return arrays,geometry


def density_layer(arrays,geometry):
    """Render the reference surface natively, with final-size labels separate.

    The raster contains only the surface and electrodes. Projected zero-height
    XY corners calibrate the 2D axis and peak-label coordinates in the plate.
    """
    xx,yy=np.meshgrid(arrays['x_mm'],arrays['y_mm']);z=arrays['density_per_mm2']
    deep=np.array(to_rgb('#65509A'))
    cmap=LinearSegmentedColormap.from_list('position_purple',[np.ones(3)*(1-t)+deep*t for t in np.linspace(.035,1,256)])
    norm=Normalize(0,float(z.max()))
    temp=plt.figure(figsize=(4,4),dpi=240)
    ax=temp.add_axes([0,0,1,1],projection='3d',computed_zorder=False)
    ax.plot_surface(xx,yy,z,rcount=251,ccount=251,cmap=cmap,norm=norm,linewidth=0,
                    edgecolor='none',antialiased=True,shade=False,zorder=2)
    names=np.array(geometry['contact_names']);xy=np.array(geometry['contact_xy_mm'])
    for shaft in ('SCL','ICL'):
        ix=sorted(np.flatnonzero(np.char.startswith(names,shaft)),key=lambda i:int(names[i][len(shaft):]),reverse=True)
        ax.plot(*xy[ix].T,zs=.0001,color='#9B9AA2',lw=.75,marker='o',ms=2.4,mfc='white',mec='#8B8993',mew=.5,zorder=3)
    ax.set(xlim=(0,20),ylim=(0,20),zlim=(0,float(z.max())*1.2))
    ax.view_init(elev=72,azim=-90.000001);ax.set_proj_type('ortho');ax.set_box_aspect((1,1,.49));ax.set_axis_off()
    temp.canvas.draw()
    def projected(x,y,h):
        px,py,_=proj3d.proj_transform(x,y,h,ax.get_proj())
        return ax.transData.transform(np.column_stack([np.atleast_1d(px),np.atleast_1d(py)]))
    plane=projected(np.array([0,20,20,0]),np.array([0,0,20,20]),np.zeros(4))
    lo,hi=plane.min(0),plane.max(0)
    rgba=np.asarray(temp.canvas.buffer_rgba()).copy();height=rgba.shape[0]
    crop=rgba[int(np.floor(height-hi[1])):int(np.ceil(height-lo[1])),int(np.floor(lo[0])):int(np.ceil(hi[0]))]
    peak_labels=[]
    for component in arrays['core_components_per_mm2']:
        iy,ix=np.unravel_index(np.argmax(component),component.shape)
        pixel=projected(xx[iy,ix],yy[iy,ix],z[iy,ix]+z.max()*.075)[0]
        peak_labels.append((pixel-lo)/(hi-lo)*20)
    plt.close(temp)
    return crop,cmap,norm,peak_labels


def completed_density_layer(arrays,geometry):
    """Reuse the approved relief, contours and fitted shafts at plate resolution."""
    xx,yy=np.meshgrid(arrays['x_mm'],arrays['y_mm']);z=arrays['density_per_mm2']
    deep=np.array(to_rgb('#65509A'))
    cmap=LinearSegmentedColormap.from_list('position_purple',[np.ones(3)*(1-t)+deep*t for t in np.linspace(.015,1,256)])
    norm=Normalize(0,float(z.max()));ground=.00005
    temp=plt.figure(figsize=(4,4),dpi=300)
    ax=temp.add_axes([0,0,1,1],projection='3d',computed_zorder=False)
    gx,gy=np.meshgrid([1,19],[1,19])
    ax.plot_surface(gx,gy,np.zeros_like(gx),color=cmap(0.),shade=False,linewidth=0,antialiased=False,zorder=.5)
    background=np.unique(arrays['all_positions_mm'].reshape(-1,2),axis=0)
    ax.scatter(*background.T,zs=ground,s=3.2,c='#B6B2BF',alpha=.36,linewidths=0,depthshade=False,zorder=1)
    rgba=cmap(norm(z));spacing=float(np.diff(arrays['x_mm'])[0])
    rgba[...,:3]=LightSource(azdeg=315,altdeg=50).shade_rgb(rgba[...,:3],z*10.5/(z.max()*1.2),
        dx=spacing,dy=spacing,blend_mode='soft',fraction=.65)
    view=np.ix_((arrays['y_mm']>=1)&(arrays['y_mm']<=19),(arrays['x_mm']>=1)&(arrays['x_mm']<=19))
    ax.plot_surface(xx[view],yy[view],np.where(z[view]>=z.max()*1e-4,z[view],np.nan),
        rcount=321,ccount=321,facecolors=rgba[view],linewidth=0,edgecolor='none',antialiased=False,shade=False,zorder=2)
    isolated=[]
    for core in range(2):
        contour=geometry['density_contours']['AB'[core]]['0.9']
        for points in contour['paths_mm']:
            points=np.asarray(points)
            ax.plot(*points.T,zs=ground,color='#9784B6',lw=.6,alpha=.8,zorder=5)
        points=arrays['selected_positions_mm'][:,core]
        squared=((points[:,None,:]-points[None,:,:])**2).sum(2)
        values=np.exp(-squared/(2*.35**2)).mean(1)/(2*np.pi*.35**2)
        isolated.extend(points[values<contour['threshold_per_mm2']])
    isolated=np.unique(np.asarray(isolated).reshape(-1,2),axis=0)
    ax.scatter(*isolated.T,zs=ground,s=12,facecolors='none',edgecolors='#65509A',linewidths=.65,depthshade=False,zorder=6)
    names=np.array(geometry['contact_names']);xy=np.array(geometry['straight_contact_xy_mm'])
    for shaft in ('SCL','ICL'):
        ix=sorted(np.flatnonzero(np.char.startswith(names,shaft)),key=lambda i:int(names[i][len(shaft):]))
        points=xy[ix];direction=points[-1]-points[0];direction/=np.linalg.norm(direction)
        np.testing.assert_allclose(points-points[0],((points-points[0])@direction)[:,None]*direction,atol=1e-10,rtol=0)
        ax.plot(*points[[0,-1]].T,zs=ground,color='#948F9D',lw=.6,zorder=7)
        ax.plot(*points.T,zs=ground,ls='none',marker='o',ms=2.6,mfc='white',mec='#948F9D',mew=.65,zorder=7.1)
    ax.set(xlim=(1,19),ylim=(1,19),zlim=(0,float(z.max())*1.2))
    ax.view_init(elev=64,azim=-90.000001);ax.set_proj_type('ortho');ax.set_box_aspect((18,18,10.5));ax.set_axis_off()
    temp.canvas.draw()
    def projected(x,y,h):
        px,py,_=proj3d.proj_transform(x,y,h,ax.get_proj())
        return ax.transData.transform(np.column_stack([np.atleast_1d(px),np.atleast_1d(py)]))
    plane=projected([1,19,19,1],[1,1,19,19],np.zeros(4));lo,hi=plane.min(0),plane.max(0)
    rgba=np.asarray(temp.canvas.buffer_rgba()).copy();height=rgba.shape[0]
    crop=rgba[int(np.floor(height-hi[1])):int(np.ceil(height-lo[1])),int(np.floor(lo[0])):int(np.ceil(hi[0]))]
    labels=[]
    for component in arrays['core_components_per_mm2']:
        iy,ix=np.unravel_index(np.argmax(component),component.shape)
        pixel=projected(xx[iy,ix],yy[iy,ix],z[iy,ix]+z.max()*.05)[0]
        labels.append(1+(pixel-lo)/(hi-lo)*18)
    plt.close(temp)
    return crop,cmap,norm,labels


def draw_lower(plate,fig,references):
    arrays,geometry=position_data(plate);rows=recurrent_data(plate)
    completed='straight_contact_xy_mm' in geometry
    pixels,cmap,norm,labels=(completed_density_layer if completed else density_layer)(arrays,geometry)
    y=plate.MID_ROWS['E']
    error=fig.add_axes(plate.rect(13,y,40,40))
    x=np.array([r['EE_same_core_scale'] for r in rows])
    error_curves(plate,error,x,[[r[k] for r in rows] for k in METRICS],
                 r'$J_{\mathrm{EE,core}}$',[.6,.85,1.25],references['EE_same_core_scale'])
    error.set_xlim(.57,1.28)
    error.set_xticklabels(['0.60','0.85','1.25'])
    # Keep the subscripted label within the established inter-row space.
    error.xaxis.labelpad=1
    # The smaller spatial map shares the angle panel's horizontal centre;
    # its colour scale fits within the same right-hand column.
    size=40 if completed else 32;bottom=y+(40-size)/2;left=87-size/2
    limits=(1,19) if completed else (0,20);ticks=[5,10,15] if completed else [0,10,20]
    position=fig.add_axes(plate.rect(left,bottom,size,size))
    position.imshow(pixels,extent=(*limits,*limits),origin='upper',aspect='equal')
    position.set(xlim=limits,ylim=limits,xticks=ticks,yticks=ticks,xlabel='x (mm)',ylabel='y (mm)')
    plate.axis_type(position)
    position.xaxis.label.set_fontsize(plate.MIDDLE_LABEL_PT);position.yaxis.label.set_fontsize(plate.MIDDLE_LABEL_PT)
    position.xaxis.labelpad=position.yaxis.labelpad=2
    for xy,label in zip(labels,['Core A','Core B']):
        position.text(*xy,label,fontsize=plate.FONT['legend'],ha='center',va='bottom',color='#403949')
    if completed:
        handles=[Line2D([],[],marker='o',ls='none',mfc='#B6B2BF',mec='none',ms=2.5,label='Search'),
                 Line2D([],[],marker='o',ls='none',mfc='none',mec='#65509A',mew=.65,ms=3.2,label='Outlier'),
                 Line2D([],[],color='#9784B6',lw=.6,label='90% density')]
        legend=position.legend(handles=handles,loc='upper right',bbox_to_anchor=(1,1.18),
            fontsize=plate.FONT['legend'],frameon=True,framealpha=.96,
            facecolor='white',edgecolor='#B6B1BE',borderpad=.25,labelspacing=.15,handlelength=.8,handletextpad=.3,borderaxespad=.25)
        legend.get_frame().set_linewidth(.6)
    bar=fig.add_axes(plate.rect(left+size+2,bottom,1.4,size))
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=bar,ticks=[0,.2,.4])
    cb.outline.set_visible(False);plate.axis_type(bar)
    cb.set_label(r'Position density (mm$^{-2}$)',fontsize=plate.FONT['label'],labelpad=1 if completed else 3)
    if completed:
        cb.ax.tick_params(pad=1)
        plate.CHECKS['D_position_density'].update(xy_label_font_pt=plate.MIDDLE_LABEL_PT,
            tick_font_pt=plate.FONT['tick'],legend_font_pt=plate.FONT['legend'],
            legend_order=['Search','Outlier','90% density'],colorbar_equal_height=True,
            camera_elevation_deg=64,same_linear_height_scale_for_both_cores=True)
    plate.CHECKS['D_lower_axis_height_mm']={'EE':40,'position_density':size}
    plate.CHECKS['D_recurrent_EE']['artist_values_verified']=True
    return error,position,bar
