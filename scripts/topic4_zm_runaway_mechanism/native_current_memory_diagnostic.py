"""Separate persistent within-group input differences from fluctuating input.

Read-only, one original history. No covariance multiplier or response fitting.
"""
from common import OUT,np,read,write,log,model
from native_input_spread_resolution import SOURCE
from datetime import datetime

DEST=OUT/'native_current_memory'


def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'result.json').exists()
    write(DEST/'contract.json',dict(status='REGISTERED_BEFORE_SCORING',created_local=datetime.now().astimezone().isoformat(),
        question='How much unresolved E-cell input spread persists as cell-specific offsets, and how much is changing within each existing state window?',
        data='Original5ms perEcell fields9-10.37s; no replay, newneuralnoise, training or modelparameter changes.',
        windows_ms=[[9000,9420],[9420,9868.5],[9868.5,10370]],grids=[20,40],
        definition='Within each originalgroup subtract instantaneousexactcellmean, then decompose residual into eachcells windowtime-mean plus its remainder. Totalpooledvariance=offsetvariance+dynamicvariance exactly. This is a window-dependent algebraic decomposition; offset does not by itself establish quenched randomness.',
        correlation='Cell-weighted pooled residual correlations at5,10,20,40,80ms, before andafter removing percellwindowmean. Nonstationary descriptive correlations, no independent-trial inference or spectral stability.',
        baseline='Same fixednativecellmembership andstates, g20 versusg40, physicalcoremembership. No claim that measured cross-cell variance equals private diffusion.'))
    times=[];signals=[]
    for path in sorted(SOURCE.glob('*.npz')):
        if int(path.stem.split('_')[1])<=90000 or int(path.stem.split('_')[0])>=103700:continue
        z=np.load(path);t=z['zm_step']*.1;keep=(t>=9000)&(t<10370)
        if not keep.any():continue
        a=z['ie'][keep].astype(float);g=z['ii'][keep].astype(float);zg=z['z'][keep].astype(float)*g;m=.0005*z['m'][keep].astype(float)
        times.append(t[keep]);signals.append(np.stack([a,g,zg,m,a-zg-m],axis=1))
    t=np.concatenate(times);x=np.concatenate(signals);assert x.shape[2]==32000 and np.max(abs(np.diff(t)-5))<1e-8
    names=['AMPA','GABA_raw','GABA_effective','M_current','net'];rows=[];temporal=[]
    for grid in [20,40]:
        s=model(grid);groups=s.geo['cell_group'][:32000];region=s.geo['group_region'][groups]
        assert np.array_equal(np.bincount(region),[754,786,30460]);sizes=np.bincount(groups,minlength=s.P);den=np.maximum(sizes,1)
        residual=np.empty_like(x)
        for k in range(len(t)):
            for ch in range(len(names)):
                mean=np.bincount(groups,weights=x[k,ch],minlength=s.P)/den
                residual[k,ch]=x[k,ch]-mean[groups]
        for lo,hi in [(9000,9420),(9420,9868.5),(9868.5,10370)]:
            keep=(t>=lo)&(t<hi);y=residual[keep];offset=y.mean(0);dynamic=y-offset
            for reg,name in enumerate(['Core A','Core B','Surround']):
                cells=region==reg
                for ch,kind in enumerate(names):
                    v=float(np.mean(y[:,ch,cells]**2));q=float(np.mean(offset[ch,cells]**2));r=float(np.mean(dynamic[:,ch,cells]**2))
                    error=abs(v-q-r);assert error<1e-9*max(v,1)
                    rows.append(dict(grid=grid,window_ms=[lo,hi],region=name,current=kind,total_variance_mv2=v,window_offset_variance_mv2=q,
                        remaining_variance_mv2=r,offset_fraction=q/v if v else None,decomposition_error=error))
                for label,xx in [('raw_residual',y),('offset_removed',dynamic)]:
                    for lag_ms in [5,10,20,40,80]:
                        lag=lag_ms//5
                        for ch in [0,2,4]:
                            a=xx[:-lag,ch,cells];b=xx[lag:,ch,cells];norm=np.sqrt(np.mean(a*a)*np.mean(b*b))
                            corr=float(np.mean(a*b)/norm) if norm>0 else None
                            temporal.append(dict(grid=grid,window_ms=[lo,hi],region=name,current=names[ch],residual=label,lag_ms=lag_ms,normalized_cross_moment=corr))
                cov_total=float(np.mean(y[:,0,cells]*y[:,2,cells]));cov_offset=float(np.mean(offset[0,cells]*offset[2,cells]));cov_dyn=float(np.mean(dynamic[:,0,cells]*dynamic[:,2,cells]))
                assert abs(cov_total-cov_offset-cov_dyn)<1e-9*max(abs(cov_total),1)
                rows.append(dict(grid=grid,window_ms=[lo,hi],region=name,current='AMPA_x_GABA_effective',total_covariance_mv2=cov_total,window_offset_covariance_mv2=cov_offset,remaining_covariance_mv2=cov_dyn))
        log('NATIVE CURRENT MEMORY',grid,'COMPLETE')
    write(DEST/'result.json',dict(status='READ_ONLY_DIAGNOSTIC_COMPLETE',sampling_ms=5,samples=len(t),rows=rows,temporal=temporal,
        scope='One nativehistory, Ecellphysicalregions. Algebraicwindowoffsets are not causal allocation or proof of quenchedparameters. No fittedmodel, population inference or bifurcationclaim.',model_promoted=False))
    log('CURRENT MEMORY PREENTRY',[(r['grid'],r['region'],r.get('offset_fraction')) for r in rows if r['current']=='net' and r['window_ms'][0]==9000])


if __name__=='__main__':main()
