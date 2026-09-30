"""Recompute autonomous probe observables and exact static residuals."""
from common import OUT,np,read,write,model
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from refractory_spatial_resolution import mapping,projections
from scipy.signal import find_peaks
D=OUT/'transient_autonomous_Z_probe_20260923'
assert read(D/'jobs.json')['status']=='COMPLETE'
s=PhysicalDelayConditionalDrift();coarse=model(20);parent,_=mapping(coarse,s)
P,count=projections(s,coarse,parent)[20]
rows=[]
for tm in [9420,9870]:
 z=np.load(D/str(tm)/'trajectory.npz');f=np.load(D/str(tm)/'final_state.npz');old=read(D/str(tm)/'result.json')
 r=z['group_rate_hz'].astype(float);field=z['field_E_hz'].astype(float);whole=z['global_E_hz'];Z=z['Z']
 assert np.array_equal(z['elapsed_time_ms'],np.arange(1,5001.))
 assert np.array_equal(z['state_time_ms'],np.arange(10,5001.,10))
 assert np.array_equal(Z,np.load(D/'fields.npz')[str(tm)])
 assert np.array_equal(f['syn'][5],Z)
 assert np.isfinite(r).all() and r.min()>=0
 # 1/ref bounds stationary rates, not a transient 1ms bin shorter than ref.
 # The physical bound is integrated occupancy of each refractory window.
 h=f['history'];tick=int(f['clock'][0]);ordered=h[(tick-len(h)+1+np.arange(len(h)))%len(h)]
 assert tick==280000
 assert np.array_equal(f['emitted'],f['rate']) and np.array_equal(f['emitted'],h[tick%len(h)])
 occupancy=[]
 for mask,ref in [(s.E,2),(~s.E,1)]:
  used=np.lib.stride_tricks.sliding_window_view(ordered[:,mask],round(ref/.05),axis=0).sum(-1)*.05
  assert used.min()>=-1e-12 and used.max()<=1+1e-9
  occupancy.append(float(used.max()))
 assert np.isfinite(z['M_current']).all() and z['M_current'].min()>=0
 err={'field':float(abs((P@r.T).T-field).max()),'whole':float(abs(r[:,s.E]@s.mean_weights-whole).max()),'field_whole':float(abs(field@(count/count.sum())-whole).max())}
 assert max(err.values())<1e-4
 peaks,_=find_peaks(whole[-2000:],prominence=5,distance=10)
 assert len(peaks)==old['peaks_in_final2s'] and abs(whole[-2000:].mean()-old['tail_global_mean_hz'])<1e-12
 persistent=float((count/count.sum())[(field[-1000:]>50).mean(0)>=.9].sum())
 assert abs(persistent-old['tail_persistent_fraction'])<1e-12
 s.set_Z(Z);mean=z['mean_tail_rate_per_ms'];res=s.residual(mean)
 op=s.moments(mean)
 row=dict(native_Z_time_ms=tm,D=s.D,readout_errors_hz=err,maximum_final_refractory_occupancy_E_I=occupancy,
  tail_mean_static_residual_max_hz=float(abs(res).max()*1000),
  tail_mean_static_residual_RMS_hz=float(np.sqrt(np.mean(res**2))*1000),
  tail_mean_regional_rates_hz=s.regional_rates(mean),
  mean_input_mu_range_mV=[float(op[0].min()),float(op[0].max())],
  tail_M_stationary_target_max_difference_mV=float(abs(z['M_current'][-200:].mean(0)-.5*s.E*mean).max()),
  tail_persistent_fraction=persistent, scope='Static residual of a varying trajectory mean is a nonlinearity/convergence diagnostic, not an equilibrium or stability certificate.')
 rows.append(row);print(row,flush=True)
write(D/'independent_audit.json',dict(status='READOUT_AUDIT_PASS',rows=rows,model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))
