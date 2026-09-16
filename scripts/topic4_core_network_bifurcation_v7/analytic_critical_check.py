from common import *
from periodic import Orbit
from analytic_antiperiodic import build
from analytic_gains import gains
import numpy as np
s=System();rows=[]
for row in read(OUT/'joint_critical_gallery.json'):
    z=np.load(row['source']);r=z['r'];N=len(r);T=float(z['T']);g=float(z['g'])
    if 'mode' in z:
        K=build(s,r,T,g);v=z['mode'].ravel();w=z['left_mode'].ravel()
        result=dict(label=row['label'],source=row['source'],analytic_right_null_max=float(abs(v-K@v).max()),analytic_left_null_max=float(abs(w-K.T@w).max()))
    else:
        o=Orbit(s,g,N);H,Hp,L,Lp=o.kernels(T);mu=s.ext_mu+o.mean(r,H);ve=s.ext_var+(r[:,:3]@o.Q[:,:3].T)*s.tm;vi=(r[:,3:]@o.Q[:,3:].T)*s.tm
        u,v,h=gains(s,mu,ve,vi);phi=o.phi(mu,ve,vi);right=z['right_null'];dr=right[:-1].reshape(N,6)*.01
        dmu=o.mean(dr,H);dve=(dr[:,:3]@o.Q[:,:3].T)*s.tm;dvi=(dr[:,3:]@o.Q[:,3:].T)*s.tm
        colp=-(o.filt(u*o.mean(r,Hp),L)+o.filt(phi,Lp))/.01
        null=(dr-o.filt(u*dmu+v*dve+h*dvi,L))/.01+colp*right[-1]
        result=dict(label=row['label'],source=row['source'],analytic_right_null_max=float(abs(null).max()))
    rows.append(result);print(result,flush=True)
write('analytic_critical_mode_validation.json',rows)
