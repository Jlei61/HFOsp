#!/usr/bin/env python3
"""Conditional readout sensitivity on the actual pilot window support.

This deliberately gives the readout the true synthetic factor. It is an
optimistic scoring/denominator diagnostic, not power of a learned observer.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v035.contracts import atomic_json


def run(root,output,repeats=500):
    if output.exists():raise FileExistsError(output)
    result=[];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for subject in ['epilepsiae_1096','epilepsiae_1125','epilepsiae_253']:
        path=root/'human_data_v2'/f'{subject}.pt';data=torch.load(path,map_location='cpu',weights_only=False)
        samples=[s for s in data['samples'] if s['targets'][1][2]];chosen=[]
        for phase in ['FIT','INNER','SELECTION']:
            last=-np.inf
            for i,s in enumerate(samples):
                if s['phase']==phase and s['anchor']>=last+1800:
                    chosen.append(i);last=s['anchor']
        samples=[samples[i] for i in chosen];times=np.array([s['anchor'] for s in samples]);phases=np.array([s['phase'] for s in samples])
        fit=phases=='FIT';inner=phases=='INNER';sel=phases=='SELECTION'
        # Known factor from available event counts older than 30 minutes.
        q=np.array([np.sum(s['histories']['8.0'][0][:,0])-np.sum(s['histories']['0.5'][0][:,0]) for s in samples])
        q=(q-q[fit].mean())/max(q[fit].std(),1e-8)
        design=np.c_[np.ones(len(q)),q];solve=np.linalg.pinv(design[fit])
        rows=[];rng=np.random.default_rng(39092026)
        innovations=rng.normal(size=(repeats,len(q)));noise=innovations.copy()
        # Correlated error at the real target-query spacing; gaps are not filled
        # with extra observations. Same draws across effect strengths.
        for i in range(1,len(q)):
            rho=np.exp(-(times[i]-times[i-1])/7200.)
            noise[:,i]=rho*noise[:,i-1]+np.sqrt(1-rho*rho)*innovations[:,i]
        for strength in [0.,.1,.25,.5,1.]:
            y=noise+strength*q[None];parent=np.broadcast_to(y[:,fit].mean(-1)[:,None],y.shape)
            prediction=(design@(solve@y[:,fit].T)).T
            use=((prediction[:,inner]-y[:,inner])**2).mean(-1)<((parent[:,inner]-y[:,inner])**2).mean(-1)
            prediction=np.where(use[:,None],prediction,parent)
            gains=(((parent[:,sel]-y[:,sel])**2)-((prediction[:,sel]-y[:,sel])**2)).mean(-1)
            rows.append(dict(signal_sd_per_fit_factor_sd=strength,repeats=repeats,
                fraction_positive_heldout_gain=float(np.mean(gains>1e-10)),fraction_parent_fallback=float(np.mean(~use)),
                gain_quantiles=dict(zip(['p05','p50','p95'],map(float,np.quantile(gains,[.05,.5,.95]))))))
        result.append(dict(subject=subject,data_sha256=sha(path),n_windows={p:int((phases==p).sum()) for p in ['FIT','INNER','SELECTION']},
            query_times=times.tolist(),known_factor_fit_sd=1.,results=rows))
    atomic_json(output,dict(status='COMPLETE',format='v039_oracle_readout_support_calibration_v1',rows=result,
        contract=dict(signal='Known standardized available 0.5-to-8h event count factor supplied directly to an affine readout',
            response='Synthetic future response = strength * factor + unit-variance OU error, 2h correlation',
            support='Actual 2h-lead/30min-target nonoverlapping FIT/INNER/SELECTION windows; real gaps and publication delay retained',
            training='OLS intercept+known factor on FIT; INNER chooses this or fitted intercept; all strengths/repeats retained',
            seed=39092026,scope='Optimistic conditional readout/scoring sensitivity only; no observer learning, no clinical seizure labels, no hypothesis-test power claim'),
        interpretation='A small denominator may remain uninformative even when the correct factor is handed to the readout. This does not calibrate the full nonlinear learning or seizure-transfer pipeline and cannot license a biological negative.',
        source_sha256=sha(__file__),development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False))
    print(json.dumps(dict(status='COMPLETE',subjects=len(result),replicates_per_strength=repeats)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
