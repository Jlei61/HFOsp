"""Versioned ellipse and local outgoing-EE interventions; no patient routes.

Ellipse membership preserves the corresponding circular E count by default.
Dose matching preserves total threshold lowering, including the 11 mV floor.
Outgoing resampling changes only A-core E -> outside E edges, preserving each
source's outgoing edge count and total weight, with distance-dependent delays.
"""
import copy
import numpy as np
from src import topic4_core_connectivity_v2 as v2

VERSION='core_shape_output_v1'


def shape_field(pos_e,pos_i,centers,radii,scales,shape,**kwargs):
    reference=v2.threshold_field(pos_e,centers,radii,scales,**kwargs)
    ec=reference['core_index'];ic=v2.core_index_for(pos_i,centers,radii)
    ar=float(shape.get('aspect_A',1.));angle=float(shape.get('angle_A_deg',0.))
    if not np.isfinite([ar,angle]).all() or ar<1:raise ValueError('invalid ellipse')
    center=np.asarray(centers[0]);r=float(radii[0]);theta=np.deg2rad(angle)
    axes=np.array([r*np.sqrt(ar),r/np.sqrt(ar)])
    u=np.array([np.cos(theta),np.sin(theta)]);v=np.array([-u[1],u[0]])
    def metric(pos):
        d=np.asarray(pos)-center
        return (d@u/axes[0])**2+(d@v/axes[1])**2
    qmax=1.;members=ec==0
    if ar!=1:
        q=metric(pos_e);count=int(members.sum())
        chosen=np.lexsort((np.arange(len(q)),q))[:count];qmax=float(q[chosen].max())
        members=np.zeros(len(q),bool);members[chosen]=True
        if np.any(members&(ec==1)):raise ValueError('E ellipse overlaps core B')
        ec=np.where(ec==0,-1,ec);ec[members]=0
        new_i=metric(pos_i)<=qmax
        if np.any(new_i&(ic==1)):raise ValueError('I ellipse overlaps core B')
        ic=np.where(ic==0,-1,ic);ic[new_i]=0
    actual_axes=axes*np.sqrt(qmax)
    halfbox=np.sqrt((actual_axes[0]*u)**2+(actual_axes[1]*v)**2)
    if np.any(center-halfbox<0) or np.any(center+halfbox>kwargs.get('sheet_mm',20.)):
        raise ValueError('ellipse crosses sheet boundary')
    d=reference['d_latent'];base=float(kwargs['v_base']);floor=float(kwargs.get('floor_mv',11.));cap=base-floor
    scale=float(scales[0]);target=None
    if shape.get('match_dose',False):
        rr=list(radii);rr[0]=float(shape.get('dose_reference_radius_mm',r))
        target_ref=v2.threshold_field(pos_e,centers,rr,scales,**kwargs)
        target=float(-target_ref['delta_vtheta'][target_ref['core_index']==0].sum())
        if cap*np.count_nonzero(d[members])<target:raise ValueError('cannot match lowering dose')
        lo,hi=0.,max(1.,scale)
        while np.minimum(hi*d[members],cap).sum()<target:hi*=2
        for _ in range(64):
            mid=(lo+hi)/2
            if np.minimum(mid*d[members],cap).sum()<target:lo=mid
            else:hi=mid
        scale=(lo+hi)/2
    lowering=np.zeros(len(d));lowering[ec==0]=np.minimum(scale*d[ec==0],cap)
    lowering[ec==1]=np.minimum(float(scales[1])*d[ec==1],cap)
    vtheta=np.full(kwargs['n_total'],base);vtheta[:len(d)]=base-lowering
    audit=dict(reference['audit'],physics=VERSION,members=[int((ec==k).sum()) for k in [0,1]],
        n_members=int((ec>=0).sum()),n_raised=int((vtheta>base).sum()),n_lowered=int((lowering>0).sum()),
        total_lowering_mV=float(lowering.sum()),total_lowering_per_core_mV=[float(lowering[ec==k].sum()) for k in [0,1]],
        mean_lowering_per_member_mV=[float(lowering[ec==k].mean()) for k in [0,1]],
        floor_clipped_count=int(np.sum((scale*d>cap)&(ec==0))+np.sum((float(scales[1])*d>cap)&(ec==1))),
        applied_A_depth_scale=scale,shape_requested=shape,ellipse_A_semiaxes_mm=actual_axes.tolist(),ellipse_A_angle_deg=angle,
        min_vtheta_mV=float(vtheta[:len(d)].min()),max_vtheta_mV=float(vtheta[:len(d)].max()),
        floor_clipped_total_mV=float(np.maximum(scale*d[ec==0]-cap,0).sum()+np.maximum(float(scales[1])*d[ec==1]-cap,0).sum()),
        ellipse_metric_cutoff=qmax,matched_E_count=bool((ec==0).sum()==(reference['core_index']==0).sum()),
        dose_target_mV=target,dose_error_mV=None if target is None else float(lowering[ec==0].sum()-target),
        boundary_clipped=[False,reference['audit']['boundary_clipped'][1]])
    return dict(vtheta=vtheta,delta_vtheta=-lowering,core_index=ec,i_core_index=ic,d_latent=d,audit=audit)


def local_outgoing(ampa,pos,n_e,ec,kernel,config,p,topology_seed):
    if not config.get('resample',False):return ampa,dict(enabled=False)
    rows,cols,data,steps=v2._edges(ampa)
    selected=(rows<n_e)&(ec[np.minimum(rows,n_e-1)]<0)&(ec[cols]==0)
    outside=np.flatnonzero(ec<0);sources=np.flatnonzero(ec==0)
    counts=np.bincount(cols[selected],minlength=n_e);weights=np.bincount(cols[selected],weights=data[selected],minlength=n_e)
    k=copy.deepcopy(kernel);k['l_par']*=float(config.get('parallel_scale',1.));k['l_perp']*=float(config.get('perp_scale',1.))
    k['theta_deg']+=float(config.get('angle_offset_deg',0.));k['theta_rad']=np.deg2rad(k['theta_deg'])
    rr=[];cc=[];ww=[]
    for source in sources:
        count=int(counts[source])
        if not count:continue
        if count>len(outside):raise ValueError('too many outgoing edges')
        rng=np.random.default_rng(np.random.SeedSequence([int(topology_seed),20260911,73,int(source)]))
        keys=v2._positive_weight_keys(v2._kernel_weights(pos[source],pos[outside],k),rng)
        target=outside[np.argpartition(keys,count-1)[:count]]
        rr.append(target);cc.append(np.full(count,source));ww.append(np.full(count,weights[source]/count))
    rr=np.concatenate(rr);cc=np.concatenate(cc);ww=np.concatenate(ww)
    dist=np.linalg.norm(pos[rr]-pos[cc],axis=1);ss=v2._delay_steps(dist,p)
    assert np.array_equal(np.bincount(cc,minlength=n_e),counts)
    assert np.allclose(np.bincount(cc,weights=ww,minlength=n_e),weights,rtol=1e-10,atol=1e-10)
    updated=v2._regroup(np.r_[rows[~selected],rr],np.r_[cols[~selected],cc],np.r_[data[~selected],ww],np.r_[steps[~selected],ss],ampa[0].shape[0],n_e)
    displacement=pos[rr]-pos[cc];cov=np.cov(displacement.T,aweights=ww);ev,vec=np.linalg.eigh(cov)
    return updated,dict(enabled=True,scope='E_A -> E_out only; B, within-core and E->I edges preserved',kernel=k,
        resampling_key_namespace=[int(topology_seed),20260911,73],source_count_preserved=True,source_weight_preserved=True,
        n_edges=int(len(rr)),mean_distance_mm=float(dist.mean()),distance_quantiles_mm=np.quantile(dist,[.05,.5,.95]).tolist(),
        actual_weighted_axis_deg=float(np.rad2deg(np.arctan2(vec[1,-1],vec[0,-1]))),
        actual_weighted_axis_ratio=float(np.sqrt(ev[-1]/max(ev[0],1e-12))),mean_delay_ms=float(ss.mean()*p.dt),
        changed_target_input_expected=True)
