"""Test the Z population closure and the diffusion-variance assumption directly on native per-cell records.

Native fields record raw I_E, I_I per E cell every 10 ms (5 ms in 8.8-10.4 s). For each
(time, E group): native fraction 1[I_I >= 95.198...] vs Gaussian closure with the group mean and
(i) the diffusion variance predicted from the mean, (ii) the measured across-cell variance.
d<z>/dt = (<1[I_I<th]> - <z>)/tau_z holds exactly for group means, so the closure error is
entirely in P(I_I<th). No fitting.
"""
from common_v3 import *
from scipy.special import ndtr
import glob
THRESHOLD=95.19851312666987
geo=dict(np.load(OPERATORS/'g20/geometry.npz'));prm=read(OPERATORS/'g20/prepared.json')['params']
E=geo['population']==0;P=len(E);cg=geo['cell_group'][:32000];size=geo['group_size']
tauG=prm['tau_r_GABA']+prm['tau_d_GABA'];tauA=prm['tau_r_AMPA']+prm['tau_d_AMPA'];tmE=prm['tau_m_E']
areaG=.1/(prm['tau_r_GABA']*(1-np.exp(-.1/prm['tau_r_GABA'])));areaA=.1/(prm['tau_r_AMPA']*(1-np.exp(-.1/prm['tau_r_AMPA'])))
# effective J per E group from operators (sum J^2 / sum J over all delays and sources)
def effJ(name):
    a=sparse.load_npz(OPERATORS/f'g20/mean_{name}.npz');q=sparse.load_npz(OPERATORS/f'g20/variance_{name}.npz')
    return np.asarray(q.sum(axis=1)).ravel()/np.maximum(np.asarray(a.sum(axis=1)).ravel(),1e-300)
JG=effJ('gaba');JA=effJ('ampa')
files=sorted(glob.glob(str(NATIVE/'fields/*.npz')))
rows=[];global_rows=[]
for f in files:
    z=np.load(f);t=z['zm_step']*.1
    if t[0]<3000:continue
    assert np.array_equal(geo['group_cell'][cg],z['cell_e']) if 'cell_e' in z.files else True
    for k in range(len(t)):
        ii=z['ii'][k].astype(float);ie=z['ie'][k].astype(float);zz=z['z'][k]
        s1=np.bincount(cg,weights=ii,minlength=P);s2=np.bincount(cg,weights=ii*ii,minlength=P);n=np.bincount(cg,minlength=P).astype(float)
        m=s1/np.maximum(n,1);v=s2/np.maximum(n,1)-m*m
        above=np.bincount(cg,weights=(ii>=THRESHOLD),minlength=P)/np.maximum(n,1)
        vpred=tmE*areaG*JG*m/(2*tauG)   # diffusion variance of filtered GABA current from the group mean
        pg=1-ndtr((THRESHOLD-m)/np.sqrt(np.maximum(vpred,1e-20)))
        pm=1-ndtr((THRESHOLD-m)/np.sqrt(np.maximum(v,1e-20)))
        e1=np.bincount(cg,weights=ie,minlength=P)/np.maximum(n,1);e2=np.bincount(cg,weights=ie*ie,minlength=P)/np.maximum(n,1)-e1*e1
        epred=tmE*areaA*JA*e1/(2*tauA)
        sel=E&(n>0)
        for g in np.flatnonzero(sel&((m>30)|(above>0))):
            rows.append((t[k],g,geo['group_region'][g],n[g],m[g],v[g],vpred[g],above[g],pg[g],pm[g],zz[cg==g].mean(),e1[g],e2[g],epred[g]))
        w=n[sel]/n[sel].sum()
        global_rows.append((t[k],float((ii>=THRESHOLD).mean()),float(pg[sel]@w),float(pm[sel]@w),float(zz.mean()),float(ii.mean()),float(ie.mean())))
rows=np.array(rows);G=np.array(global_rows)
np.savez_compressed(DEST/'diagnostics/native_z_closure_rows.npz',rows=rows,global_rows=G,
    columns='t,group,region,n,mean_II,var_II,var_pred,frac_above_native,frac_above_gauss_pred,frac_above_gauss_measvar,z_mean,mean_IE,var_IE,var_pred_IE')
act=rows[rows[:,4]>30]
ratioI=act[:,5]/np.maximum(act[:,6],1e-12);ratioE=act[:,12]/np.maximum(act[:,13],1e-12)
print('samples with mean I_I>30 mV:',len(act))
print('across-cell Var(I_I)/diffusion prediction: median %.2f  IQR %.2f-%.2f'%(np.median(ratioI),*np.percentile(ratioI,[25,75])))
print('across-cell Var(I_E)/diffusion prediction: median %.2f  IQR %.2f-%.2f'%(np.median(ratioE),*np.percentile(ratioE,[25,75])))
for lo,hi in [(30,60),(60,80),(80,95),(95,110),(110,140),(140,1e9)]:
    s=act[(act[:,4]>=lo)&(act[:,4]<hi)]
    if len(s):print(f'mean I_I in [{lo},{hi}): n={len(s):6d} native P(above)={s[:,7].mean():.3f} gauss(pred var)={s[:,8].mean():.3f} gauss(meas var)={s[:,9].mean():.3f} var ratio med={np.median(s[:,5]/np.maximum(s[:,6],1e-12)):.2f}')
# global depletion drive: mean over all E cells of 1[I_I>=th] integrated over time vs closure
dt=np.diff(G[:,0]);
print('global: time-integral of native P(above) %.3f s, gauss-pred %.3f s, gauss-measvar %.3f s over %.1f-%.1f s'%((G[:-1,1]*dt).sum()/1000,(G[:-1,2]*dt).sum()/1000,(G[:-1,3]*dt).sum()/1000,G[0,0]/1000,G[-1,0]/1000))
for a,b in [(3000,8000),(8000,9420),(9420,9870),(9870,10400),(10400,12500)]:
    s=G[(G[:,0]>=a)&(G[:,0]<b)]
    if len(s):print(f'  {a}-{b} ms: native <1[above]>={s[:,1].mean():.4f} gauss_pred={s[:,2].mean():.4f} gauss_meas={s[:,3].mean():.4f} <z>={s[:,4].mean():.3f} <I_I>={s[:,5].mean():.1f} <I_E>={s[:,6].mean():.1f}')
write(DEST/'diagnostics/native_z_closure_summary.json',dict(n_active_samples=len(act),var_ratio_II_median=float(np.median(ratioI)),var_ratio_IE_median=float(np.median(ratioE)),
      windows=[dict(window_ms=[a,b],native=float(G[(G[:,0]>=a)&(G[:,0]<b),1].mean()),gauss_pred=float(G[(G[:,0]>=a)&(G[:,0]<b),2].mean()),gauss_measvar=float(G[(G[:,0]>=a)&(G[:,0]<b),3].mean())) for a,b in [(3000,8000),(8000,9420),(9420,9870),(9870,10400),(10400,12500)] if ((G[:,0]>=a)&(G[:,0]<b)).any()]))
