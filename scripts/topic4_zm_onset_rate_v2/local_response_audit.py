"""Offline, uncoupled colored-LIF assays at depleted-Z workpoints.

These neurons measure a local transfer response; they are not substituted for
the spatial rate network. No fitting is performed. Mean and both variance
channels are compared with the frozen calibrated response and the rate DDE.
"""
from audit_and_spectrum import states
from model_zm import *
import ast
import hashlib
import time
import cupy as cp
import argparse
from response import susceptibility


def kernel_source(filename,folder,dt):
    path=ROOT/'scripts/topic4_brunel_spatial_bifurcation'/filename
    code=next(ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
              if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODE' for t in n.targets))
    target=folder/filename;target.parent.mkdir(exist_ok=True);target.write_text(path.read_text())
    assert code.count('p[5]*.1')==2
    code=code.replace('p[5]*.1',f'p[5]*{dt:.17g}')
    return code,dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),dt_ms=dt)


def main(args):
    cp.cuda.Device(0).use();s=ZMSpatialRate();rows=[];params=[];channels=[]
    folder=DEST/args.label;folder.mkdir(exist_ok=True)
    sys.path.insert(0,str(ROOT/'scripts/topic4_zm_onset_bifurcation'))
    from response_transport import white_transport
    import response
    response.white=white_transport
    for label,path in states():
        if label not in args.states:continue
        z=np.load(path);r=z['r'];D=float(z['D']);s.set_D(D);mu,ve,vi=s.moments(r);gain=s.gains((mu,ve,vi))
        # Primary sampled direction: the DDE growing mode, when already saved.
        modefile=DEST/'critical_spectra'/f'{label}_rate_dde_0.npz'
        weight=abs(np.load(modefile)['vector'])**2*s.sizes if modefile.exists() else abs(z['v'])**2*s.sizes if 'v' in z else gain[0]**2*s.sizes
        for pop in [0,1]:
            mask=s.E if pop==0 else ~s.E;ids=np.flatnonzero(mask);g=ids[np.argmax(weight[ids])]
            # Re-running an assay must retain its originally selected groups,
            # independent of which spectral files have appeared meanwhile.
            original=folder/'contract.json'
            if not original.exists():original=DEST/'local_response/contract.json'
            if original.exists():
                matched=[q for q in read(original)['rows'] if q['state']==label and q['population']=='EI'[pop]]
                if matched:g=matched[0]['group']
            for channel in ['mean','variance_E','variance_I']:
                for freq in args.frequencies:
                    lam=2j*np.pi*freq/1000;chi=susceptibility(s,r,1.,lam,'calibrated_full');idx=['mean','variance_E','variance_I'].index(channel)
                    h=1. if idx==0 else 1/(1+lam*s.tau[idx-1]/2)
                    p=np.zeros(20 if idx==0 else 21);p[:6]=[mu[g],s.theta[g],ve[g],vi[g],(.15 if idx==0 else .05)*args.amplitude_scale,lam.imag]
                    if idx:p[20]=idx-1
                    for k,var in enumerate([ve[g],vi[g]]):
                        tr,td=s.rise[k],s.decay[k];ar,ad=np.exp(-args.dt/tr),np.exp(-args.dt/td);b=tr/(tr-td)*(ar-ad)
                        S=s.tm[g]*var/2*np.array([[1/tr,1/(tr+td)],[1/(tr+td),1/(tr+td)]])
                        A=np.array([[ar,0],[b,ad]]);C=np.linalg.cholesky(S-A@S@A.T)
                        if k==0:p[6:9]=[ar,b,ad];p[11:14]=[C[0,0],C[1,0],C[1,1]]
                        else:p[9:11]=[b,ad];p[14:17]=[C[0,0],C[1,0],C[1,1]];p[17]=ar
                    p[18]=np.exp(-args.dt/s.tm[g]);p[19]=round(s.ref[g]/args.dt)
                    amp=p[4] if idx==0 else p[4]*p[1+idx]
                    rows.append(dict(state=label,D=D,group=int(g),population='EI'[pop],channel=channel,frequency_hz=freq,
                        mu_mv=mu[g],variance_E=ve[g],variance_I=vi[g],threshold_mv=s.theta[g],predicted_rate_hz=r[g]*1000,
                        amplitude=amp,predicted_calibrated=chi[idx][g]*h*1000,
                        predicted_rate_dde=gain[idx][g]*s.filter_response(lam)[g]*h*1000,
                        selection='Largest weighted mode component in this population, falling back to static mode/gain if unavailable'))
                    params.append(p);channels.append(idx==0)
    kernels=[];R=4096;T=4000;started=time.time()
    write(folder/'contract.json',dict(status='DEFINED',conditions=len(rows),replicates_per_condition=R,duration_ms=T,
        frequencies_hz=args.frequencies,mean_amplitude_mv=.15*args.amplitude_scale,variance_relative_amplitude=.05*args.amplitude_scale,dt_ms=args.dt,
        refit=False,scope='Independent local response assay, not a particle replacement for the network',rows=rows))
    for mean,filename in [(True,'check_response_v2.py'),(False,'check_variance_response.py')]:
        indices=np.flatnonzero(np.array(channels)==mean);pars=np.array([params[k] for k in indices]);P=len(pars)
        code,identity=kernel_source(filename,folder,args.dt);kernels.append(identity)
        kernel=cp.RawKernel(code,'response',options=('--fmad=false',));out=cp.zeros((P*R,4));print('START',filename,P,flush=True)
        kernel(((P*R+127)//128,),(128,),(cp.asarray(pars),out,np.int32(R),np.int32(P),np.int32(round(T/args.dt)),np.int32(round(1000/args.dt)),np.uint64(2026091719)))
        observed=out.get().reshape(P,R,4)
        np.savez_compressed(folder/('mean_raw.npz' if mean else 'variance_raw.npz'),observed=observed,parameters=pars,indices=indices)
        for j,k in enumerate(indices):
            row=rows[k];obs=observed[j];est=(obs[:,0]+1j*obs[:,1])/(T*row['amplitude'])*1000
            measured=est.mean();sem=float(np.sqrt(np.mean(abs(est-measured)**2)/R))
            row.update(measured=measured,complex_sem=sem,measured_rate_hz=float((obs[:,2]+obs[:,3]).mean()/2/T*1000),
                relative_error_calibrated=float(abs(measured-row['predicted_calibrated'])/max(abs(measured),1e-12)),
                relative_error_rate_dde=float(abs(measured-row['predicted_rate_dde'])/max(abs(measured),1e-12)),
                response_snr=float(abs(measured)/max(sem,1e-12)))
        print('DONE',filename,'seconds',time.time()-started,flush=True)
    write(folder/'result.json',dict(status='COMPLETE',rows=rows,seconds=time.time()-started,kernel_sources=kernels,
        scope='Six selected local workpoints, paired common random numbers; no fit or full-network equivalence claim',
        limits='Finite modulation amplitude and finite dt; local Gaussian diffusion inputs do not reproduce all native spike correlations'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',default='local_response');p.add_argument('--dt',type=float,default=.1)
    p.add_argument('--amplitude-scale',type=float,default=1.);p.add_argument('--states',nargs='+',default=['SN1','SN7','upper_D0228'])
    p.add_argument('--frequencies',nargs='+',type=float,default=[0,6,25]);main(p.parse_args())
