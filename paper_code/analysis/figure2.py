"""Paper Fig2 display inputs built from the new E1146 event table and raw samples."""
import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import next_fast_len
from analysis.src.interictal_spatial_information_gain import fit_evaluate_all_event_crossfit_fold
from analysis.src.topic5_interictal_direction_rose import axis_pair_display_basis,project_directions_to_angles
from analysis.event_spectrogram import compute_group_event_spectrogram_stack
from analysis import event_frames as frames
from analysis.envelope_normalization import baseline_robust_z
from analysis.display_field import _smooth_rank_field_mm
from data_processing import raw_to_events as raw


def heldout_rose(events,xyz,directions):
    fold=fit_evaluate_all_event_crossfit_fold(events['ranks'],events['participating'],directions['directions'],events['block_ids'],xyz,fold_index=0)
    axes=fold['models']['timing_plus_space']['axes'];basis=axis_pair_display_basis(*axes)
    angles=project_directions_to_angles(fold['evaluations']['common_test_directions'],basis['axis_a'],basis['transverse'])['angles']
    methods={name:dict(axis_angles_rad=project_directions_to_angles(model['axes'],basis['axis_a'],basis['transverse'])['angles']) for name,model in fold['models'].items()}
    return dict(all_event_angles=angles[np.isfinite(angles)],methods=methods,fold=0)


def display_geometry(events,xyz,field_a):
    plane=field_a['plane'];relative=xyz-plane['origin'];points=np.column_stack([relative@plane['u'],relative@plane['w']])
    names=events['channel_names'].tolist();shafts=np.array([''.join(ch for ch in n if not ch.isdigit()) for n in names])
    limits=[(float(points[:,i].min()-12),float(points[:,i].max()+12)) for i in range(2)]
    return dict(names=names,shafts=shafts,points_mm=points,ax_mm=points[:,0],axial_shaft=max(set(shafts),key=lambda s:np.sum(shafts==s)),display_xlim_mm=limits[0],display_ylim_mm=limits[1],display_sigma_mm=6.,plane=plane)


def dynamic_events(events,clustered,geometry,config,raw_root):
    # Paper examples have fixed full-recording IDs. The short demo selects the
    # first event with maximum contact participation per class, independently of direction.
    output=[]
    for k in range(2):
        candidates=np.flatnonzero(clustered['labels']==k);counts=events['participating'][:,candidates].sum(0)
        ei=int(candidates[np.argmax(counts)]);bid=str(events['block_ids'][ei]);record=next(r for r in config['records'] if str(r['block_id'])==bid)
        pre,detected,features=raw.process_record(record,config,raw_root)
        local=int(np.flatnonzero(np.flatnonzero(events['block_ids']==events['block_ids'][ei])==ei)[0])
        names=features['channel_names'].tolist();order=[names.index(n) for n in geometry['names']]
        lo,hi=features['windows_sec'][local];fs=pre['sfreq'];start=max(0,round((lo-1)*fs));stop=min(pre['band_V'].shape[1],round((hi+1)*fs));t=np.arange(start,stop)/fs
        band=pre['band_V'][order,start:stop];env=np.abs(hilbert(band,N=next_fast_len(band.shape[-1]),axis=-1)[...,:band.shape[-1]]);env_z=baseline_robust_z(env,t,lo,hi)
        tile=pre['band_V'][order,round(lo*fs):round(hi*fs)];duration=tile.shape[1]/fs
        stack,st,f,centers=compute_group_event_spectrogram_stack(tile,fs,np.array([duration]),spec_window='hamming',spec_win_sec=.05,spec_overlap_sec=.04,spec_freq_range=(50.,300.),gaussian_sigma=1.5,enhancement_threshold=.70)
        part=events['participating'][:,ei];ct=centers[:,0,0];t0=lo+float(np.nanmin(ct[part]));cm=(lo+ct-t0)*1000
        output.append(dict(event_index=ei,part=part,usable=part&np.isfinite(cm),env_z=env_z,t=t,t0=t0,
            spec=stack.reshape(len(names),len(f),-1),spec_t_ms=(lo+st-t0)*1000,spec_freq_hz=f,centroid_ms=cm,
            centroid_freq_index=centers[:,0,1],tile_lo_ms=(lo-t0)*1000,tile_hi_ms=(lo+duration-t0)*1000))
    lo,hi=frames._frame_window(*output);scales=frames._event_normalization_scales(*output,lo,hi)
    times,selection=frames._select_joint_visible_frame_times(*output,geometry,lo,hi,scales)
    return dict(events=output,times_ms=times,selection=selection)


def rank_field(geometry,ranks,support):
    values=np.asarray(ranks,float);valid=np.isfinite(values)&(support>0)
    lo,hi=np.min(values[valid]),np.max(values[valid]);values=(values-lo)/(hi-lo)
    X,Y,F,S,mask=_smooth_rank_field_mm(*geometry['points_mm'].T,values,support,geometry['display_xlim_mm'],geometry['display_ylim_mm'],6.)
    return dict(X=X,Y=Y,field=F,values=values,support=support)
