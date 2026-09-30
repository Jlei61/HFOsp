"""Read-only forcing diagnostic: external AMPA mean bypass in the rate field."""
from common import OUT,ROOT,np,model,write,log
from run_network import group_drive
from numba import njit

@njit
def filters(force,tr,td):
    # force is held over each original1ms recorded interval. Record interval mean.
    dt=.05;a=np.exp(-dt/tr);d=np.exp(-dt/td);b=tr/(tr-td)*(a-d)
    q=np.zeros(force.shape[1]);I=np.zeros_like(q);out=np.empty_like(force)
    for k in range(len(force)):
        total=np.zeros_like(q)
        for step in range(20):
            old=q.copy();q=a*q+(1-a)*force[k];I=b*old+d*I+(1-d-b)*force[k];total+=I/20
        out[k]=total
    return out

def main():
    s=model();nu=group_drive(s,'seed9108401');instant=nu*(s.tm*s.area[0]*s.jext)
    filtered=filters(instant,s.rise[0],s.decay[0]);reg=s.geo['group_region'];rows=[]
    for label,mask in [('Core A',s.E&(reg==0)),('Core B',s.E&(reg==1)),('Surround',s.E&(reg==2)),('I',~s.E)]:
        weights=s.sizes[mask]/s.sizes[mask].sum();x=instant[100:,mask];y=filtered[100:,mask];err=x-y
        rows.append(dict(region=label,cells=int(s.sizes[mask].sum()),
            RMS_difference_mV=float(np.sqrt((np.mean(err**2,axis=0)*weights).sum())),
            mean_absolute_difference_mV=float((np.mean(abs(err),axis=0)*weights).sum()),
            maximum_absolute_difference_mV=float(abs(err).max()),
            original_mean_mV=float((np.mean(x,axis=0)*weights).sum()),
            original_temporal_SD_mV=float(np.sqrt((np.var(x,axis=0)*weights).sum())),
            filtered_temporal_SD_mV=float(np.sqrt((np.var(y,axis=0)*weights).sum()))))
    path=OUT/'external_mean_filter_audit';path.mkdir(exist_ok=True)
    np.savez_compressed(path/'forcing.npz',time_ms=np.arange(len(nu))+1.,instantaneous_mean_mV=instant.astype('f4'),filtered_mean_mV=filtered.astype('f4'))
    result=dict(status='FORCING_DIAGNOSTIC_COMPLETE_NO_NETWORK_CHANGE',rows=rows,
        native_source=str(ROOT/'src/topic4_raster_protocol_engine.py'),
        native_lines='430-435: external Poisson counts increment s_E; I_E then low-pass follows s_E, sameAMPApath as recurrent excitation.',
        rate_source=str(ROOT/'scripts/topic4_zm_runaway_mechanism/refractory_spatial_diagnostic.py'),
        rate_difference='physical_step adds pm=tau_m*area_A*J_ext*nu directly after recurrentAMPAfilter; externalvariance is already passed through currentcovariance.',
        protocol='Original recorded1ms external-rate averages held constant insideeach1ms; existing area_A maintainsnative discreteDCgain. Exactcontinuous two-pole mean update at.05ms,1ms outputmean.',
        limitation='Not a reconstruction of sub-ms nativeexternalrates or realizedPoisson currents. Difference is measured within the current rate-input convention. Does not establish the contribution to onset or change any activebatch.',
        model_promoted=False)
    write(path/'result.json',result);log('EXTERNAL MEAN FILTER',rows)

if __name__=='__main__':main()
