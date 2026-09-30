"""Native external drive record -> compact arrays for paired rate-model runs (stage A4).

fields/*.npz: drive_mean (per 1 ms, 20x20 cell mean applied E external rate, 1/ms), drive_var,
glob (I external rate per 1 ms), xi (global OU per 0.1 ms step), in_step.
"""
from common_v3 import *
import glob
for label,run in [('seed9108401',NATIVE),('seed9108402',NATIVE2)]:
    files=sorted(glob.glob(str(run/'fields/*.npz')));dm=[];dv=[];gl=[];xi=[];st=[]
    for f in files:
        z=np.load(f);dm.append(z['drive_mean']);dv.append(z['drive_var']);gl.append(z['glob']);xi.append(z['xi']);st.append(z['in_step'])
    dm=np.concatenate(dm);dv=np.concatenate(dv);gl=np.concatenate(gl);xi=np.concatenate(xi);st=np.concatenate(st)
    assert np.all(np.diff(st)==10),np.unique(np.diff(st))
    np.savez_compressed(DEST/f'native_reference/{label}_external_drive.npz',drive_mean=dm,drive_var=dv,glob=gl,xi=xi,in_step=st)
    print(label,dm.shape,'E drive mean/ms: overall %.4f (nu_ext_per_ms nominal 1.3175)'%dm.mean(),'I glob mean %.4f'%gl.mean(),'xi std %.3f'%xi.std(),'cell std of time-mean %.4f'%dm.mean(0).std(),'time std of cell-mean %.4f'%dm.mean(1).std())
