#!/usr/bin/env python3
"""Measured spatial frequency operator with original discrete delays/G/M.

Frequency-domain loop eigenvalues are diagnostic samples, NOT characteristic
growth exponents. No stability or Hopf certification is made by this producer.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,eigs,splu
from campaign import ROOT,read,write,sha
from direct_response_system import DirectDC
from prepare_target_density import OUT as TARGET
from measure_all_target_frequency import OUT as MEASURED,FREQUENCIES

OUT=ROOT/'direct_frequency_operator'
NAMES=['mean_ampa','mean_gaba','variance_ampa','variance_gaba']


def temporal(e,f):
    z=np.exp(2j*np.pi*f*.0001)
    filters=[]
    for name in ['AMPA','GABA']:
        ar=np.exp(-.1/e.p['tau_r_'+name]);ad=np.exp(-.1/e.p['tau_d_'+name])
        filters.append((1-ar)*(1-ad)/((1-ar/z)*(1-ad/z)))
    ar=np.exp(-.1/15);ag=np.exp(-.1/500);am=1-.1/1000
    global_gain=.1*(1-ag)*(.1/15)/((z-ag)*(z-ar))
    M=(.1/1000)/(z-am)
    return z,np.array(filters),global_gain,M


def coefficients(e,chi,f):
    z,hf,hg,hm=temporal(e,f);h=1+e.reference_g
    D=1+.0005*e.E*chi[:,0]*hm/h
    C=np.array([chi[:,0]*e.tm*e.area[0]*hf[0]/(1000*h),
        -chi[:,0]*e.Z*e.tm*e.area[1]*hf[1]/(1000*h),
        chi[:,1]*e.tm*e.area[0]**2/(1000*h**2),
        chi[:,2]*e.tm*(e.Z*e.area[1])**2/(1000*h**2)])/D
    u=e.E*e.Z*chi[:,3]*hg/D
    return C,u,D


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_DYNAMIC_OPERATOR',created_epoch=time.time(),
        question='Do measured local cell responses, original target-wise edge delays, mean synaptic filters and causalR/G/localM form a consistent spatial loop operator?',
        convention='Rate inHz; z=exp(i*2pi*f*.0001). Source r[n] is the current-step spike readout; delays read r[n-d]. Mean synaptic filter uses current arrival in the new current. M andG seen by membrane are pre-update states.',
        formulas=dict(mean_filter='(1-ar)*(1-ad)/((1-ar/z)*(1-ad/z))',
            global_R_G='.1*(1-aG)*(.1/15)/((z-aG)*(z-aR))',local_M='(.1/1000)/(z-aM)',
            implicit_M='1+etaM*E*chi_mu*H_M/(1+g)',
            variance='Use measured driver-variance frequency response times delayed squared-weight operator; do NOT add the mean synaptic filter twice.',
            physical_G='Use chi_physicalG*Z*H_RG; physical chi already contains instantaneous rescaling of the pre-existing colored currents.'),
        diagnostics='At0Hz compare the full operator action to measured DC Jacobian+identity. Test delay compression against uncompressed native delay arrays. Check scalar recurrence phases against exact discrete recurrences. At1/5Hz report eigenvalues of loop operator nearest1 with residuals and replica-block action uncertainty; never label these as dynamical growth exponents.',
        limits='Two frequency samples and a local DC operator cannot count all characteristic roots or certify stability/Hopf. Measured finite-record susceptibilities, failed-amplitude components, source averaging and fixedM response approximation remain. Any next analysis needs frequency coverage and native correspondence.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    e=DirectDC();rows=[];rng=np.random.default_rng(929039);v=rng.standard_normal(e.P)+1j*rng.standard_normal(e.P)
    for f in [0.]+FREQUENCIES:
        folder=OUT/f'frequency_{f:g}Hz';folder.mkdir(exist_ok=True);z,*_=temporal(e,f)
        for n,name in enumerate(NAMES):
            a=sparse.load_npz(TARGET/f'{name}.npz').tocsr();delay=a.indices//e.P+1
            phase=np.exp(-2j*np.pi*f*.0001*delay)
            b=sparse.csr_matrix((a.data*phase,a.indices%e.P,a.indptr.copy()),shape=(e.N,e.P));b.sum_duplicates();b.sort_indices()
            depth=a.shape[1]//e.P;history=(np.exp(-2j*np.pi*f*.0001*np.arange(1,depth+1))[:,None]*v).ravel()
            expected=a@history;observed=b@v;error=float(abs(expected-observed).max());assert error<1e-9,error
            if f==0:
                diff=b-e.W[n];dcerror=float(abs(diff.data).max()) if diff.nnz else 0.;assert dcerror<1e-10
            else:dcerror=None
            sparse.save_npz(folder/f'{name}.npz',b)
            rows.append(dict(frequency_Hz=f,operator=name,delayed_action_error=error,DC_array_error=dcerror))
            print('DELAY OPERATOR',f,name,error,flush=True)
        # Compare exact initialized sinusoidal recurrences with analytic filters.
        _,hf,hg,hm=temporal(e,f);scalar=[]
        for j,name in enumerate(['AMPA','GABA']):
            ar=np.exp(-.1/e.p['tau_r_'+name]);ad=np.exp(-.1/e.p['tau_d_'+name])
            sh=(1-ar)/(1-ar/z);s=sh/z;current=hf[j]/z;err=0.
            for n in range(1000):
                signal=z**n;s=ar*s+(1-ar)*signal;current=ad*current+(1-ad)*s
                err=max(err,abs(current-hf[j]*signal))
            assert err<1e-10;scalar.append(float(err))
        ar=np.exp(-.1/15);ag=np.exp(-.1/500);am=1-.1/1000
        Rh=(.1/15)/(z-ar);R=Rh;G=hg;M=hm;errors=np.zeros(3)
        for n in range(1000):
            signal=z**n;errors=np.maximum(errors,abs(np.array([R-Rh*signal,G-hg*signal,M-hm*signal])))
            G=ag*G+.1*(1-ag)*R;R=ar*R+(.1/15)*signal;M=am*M+(.1/1000)*signal
        assert max(errors)<1e-9,errors
        write(folder/'temporal_qa.json',dict(status='PASS',mean_filter_errors=scalar,RG_M_errors=errors.tolist(),
            H_RG=[hg.real,hg.imag],H_M=[hm.real,hm.imag]))
    C,u,D=coefficients(e,e.chi,0)
    action=e.S@(sum(c*(w@v) for c,w in zip(C,e.W))+u*(e.weights@v))
    error=float(abs(action-(e.J@v+v)).max());assert error<1e-9,error
    write(OUT/'implementation_qa.json',dict(status='PASS',rows=rows,full_DC_action_error=error))
    print('DYNAMIC ASSEMBLY QA PASS',error,flush=True)


def analyze(index):
    assert read(OUT/'implementation_qa.json')['status']=='PASS'
    f=FREQUENCIES[index];folder=OUT/f'frequency_{f:g}Hz';source=MEASURED/f'frequency_{index:02d}'
    assert not (folder/'analysis.json').exists()
    e=DirectDC()
    with np.load(source/'measured_response.npz') as z:
        chi=z['gain'][:,:,0];blocks=z['replicate_block_gain'][:,:,0]
    W=[sparse.load_npz(folder/f'{name}.npz').tocsr() for name in NAMES]
    C,u,D=coefficients(e,chi,f)
    L=sum(e.S@(w.multiply(c[:,None])) for c,w in zip(C,W))+sparse.csr_matrix(np.outer(e.S@u,e.weights))
    J=(L-sparse.eye(e.P)).tocsc();lu=splu(J)
    inverse=LinearOperator(J.shape,matvec=lu.solve,dtype=np.complex128)
    values,vectors=eigs(L,k=8,sigma=1.,OPinv=inverse,tol=1e-7,maxiter=300)
    residual=np.array([np.linalg.norm(L@vectors[:,i]-values[i]*vectors[:,i]) for i in range(8)])
    assert max(residual)<1e-5,residual
    # Preserve the full sampled matrix-action uncertainty, rather than treating
    # weak individual cell channels as zero or reclassifying them as passed.
    actions=[]
    for b in range(8):
        cb,ub,db=coefficients(e,blocks[:,:,b],f)
        actions.append(e.S@(sum(c[:,None]*(w@vectors) for c,w in zip(cb,W))+ub[:,None]*(e.weights@vectors)))
    action_sem=np.std(np.array(actions),axis=0,ddof=1)/np.sqrt(8)
    np.savez_compressed(folder/'loop_modes.npz',eigenvalue=values,source_mode=vectors,eigen_residual=residual,
        block_action=np.array(actions),action_SEM=action_sem,implicit_M_denominator=D)
    result=dict(status='SAMPLED_LOOP_MODES_ONLY',frequency_Hz=f,
        eigenvalues=[dict(real=float(v.real),imag=float(v.imag),distance_to_one=float(abs(v-1)),eigen_residual=float(r),
            block_action_SEM_norm=float(np.linalg.norm(action_sem[:,i]))) for i,(v,r) in enumerate(zip(values,residual))],
        minimum_absolute_local_M_denominator=float(abs(D).min()),
        interpretation='These are eigenvalues of the frequency-domain loop operator at a fixed imaginary frequency, not growth rates of the delayed dynamics. Two frequencies cannot establish or exclude a temporal instability.',
        dynamic_stability_established=False,formal_bifurcation_allowed=False)
    write(folder/'analysis.json',result);print(result,flush=True)


def supervise():
    assert not (OUT/'analysis_supervisor.json').exists()
    write(OUT/'analysis_supervisor.json',dict(pid=os.getpid(),started_epoch=time.time()))
    for index in range(2):
        expected=MEASURED/f'frequency_{index:02d}'/'analysis.json'
        while not expected.exists():
            if (MEASURED/'result.json').exists() and read(MEASURED/'result.json')['status'].startswith('FAILED'):
                write(OUT/'analysis_progress.json',dict(status='STOPPED_ON_FAILED_MEASUREMENT'));return
            write(OUT/'analysis_progress.json',dict(status='WAITING_MEASURED_RESPONSE',index=index,pid=os.getpid(),updated_epoch=time.time()));time.sleep(15)
        analyze(index)
    write(OUT/'analysis_progress.json',dict(status='TWO_LOOP_FREQUENCIES_ANALYZED',dynamic_stability_established=False,formal_bifurcation_allowed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','analyze','supervise']);p.add_argument('--index',type=int)
    a=p.parse_args();prepare() if a.command=='prepare' else analyze(a.index) if a.command=='analyze' else supervise()
