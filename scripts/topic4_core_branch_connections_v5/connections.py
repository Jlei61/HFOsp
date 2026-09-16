"""Compare two independently continued full periodic solutions, modulo phase."""
from common import *
import numpy as np
from scipy.signal import resample
from scipy.optimize import minimize_scalar

def compare(a,b):
    x=np.load(a);y=np.load(b);N=8192;r=resample(x['r'],N,axis=0);q=resample(y['r'],N,axis=0);Q=np.fft.rfft(q,axis=0);freq=2j*np.pi*np.arange(N//2+1)
    corr=np.fft.irfft(np.sum(np.fft.rfft(r,axis=0)*np.conj(Q),axis=1),n=N);shift=int(np.argmax(corr))/N
    def moved(s):return np.fft.irfft(Q*np.exp(-freq[:,None]*s),n=N,axis=0)
    def fun(s):return np.mean((r-moved(s))**2)
    fit=minimize_scalar(fun,bounds=(shift-2/N,shift+2/N),method='bounded',options={'xatol':1e-15});rr=moved(fit.x)
    return dict(source_a=str(a),source_b=str(b),J_a=float(x['g']),J_b=float(y['g']),J_difference=float(x['g']-y['g']),period_difference_ms=float(x['T']-y['T']),phase_shift=fit.x,max_waveform_difference_hz=float(abs(r-rr).max()*1000),rms_waveform_difference_hz=float(np.sqrt(fun(fit.x))*1000),mean_difference_hz=((r.mean(0)-q.mean(0))*1000).tolist())

def main():
    cases=[('doublet_continuity',OUT/'periodic/join_doublet_up/g1.24620000_N2048.npz',OUT/'periodic/join_doublet_down/g1.24620000_N2048.npz'),('burst_mixed_connection',OUT/'means/burst_to_B200/mean200.000000.npz',OUT/'means/mixed_to_B200/mean200.000000.npz'),('mixed_tonic_connection',OUT/'means/connection_mixed_high_continued_N1024/mean210.000000.npz',OUT/'means/connection_tonic_low_continued_N1024/mean210.000000.npz')]
    rows=[]
    for name,a,b in cases:
        if a.exists() and b.exists():
            row=dict(name=name,**compare(a,b));rows.append(row);print(name,json.dumps(row),flush=True)
    write('connection_checks.json',rows)

if __name__=='__main__':main()
