from rate_periodic import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);a=p.parse_args();s=RateField();o=Periodic(s,128,a.device);cp=o.cp;rows=[]
 for fam,ids in [('arcA',[0,29,64,74,76,87,100,104,105,106,107,110,130,150,175]),('arcB',[0,74,100,102,110,116,126,130,142,150,156])]:
  for i in ids:
   path=PERIODIC_OUT/f'orbits/{fam}_{i:04d}_N64.npz';z=np.load(path);r=cp.asarray(resample(z['r'],128,axis=0));T=float(z['T']);J=float(z['J']);ker=o.kernels(T,J);mom=o.moments(r,ker)+o.private[:,None,:]
   error=(r-o.filt(o.phi(mom),ker[-2]))*1000;ii=int(cp.argmax(cp.abs(error)));ti,pi=np.unravel_index(ii,(128,s.P))
   q=dict(orbit=str(path),J_EE_core=J,T_ms=T,oversampled_residual_hz=float(cp.max(cp.abs(error))),maximum_error_group=int(pi),maximum_error_region=int(s.geo['group_region'][pi]),minimum_group_rate_hz=float(cp.min(r)*1000),maximum_group_rate_hz=float(cp.max(r)*1000))
   rows.append(q);write(PERIODIC_OUT/'secondary_mesh_defects.json',dict(rows=rows));print(q,flush=True)

if __name__=='__main__':main()
