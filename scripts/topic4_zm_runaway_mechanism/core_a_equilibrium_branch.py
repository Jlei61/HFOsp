"""Physical Core A Z continuation of exact full-network equilibria.

The observed native Core A field varies; every non-Core-A Z stays at native9s.
All spatial states are retained and M satisfies its dynamic-equilibrium law.
"""
from common import OUT,np,read,write,log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from current_rate_logit_root import value_and_jacobian
from scipy import sparse
from scipy.sparse.linalg import spsolve,eigs
from pathlib import Path
import argparse,os,time

DEST=OUT/'core_a_bifurcation_type_20260924/equilibrium_branch'
SOURCE=OUT/'core_a_resource_bifurcation_20260923'


class Family:
    def __init__(self,s):
        self.s=s;self.A=s.E&(s.geo['group_region']==0)
        self.background=np.load(SOURCE/'fields.npz')['reference9000'].copy()
        z=np.load(OUT/'transient_native_Z_path_20260923/native_Z_path.npz')
        self.time=z['time_ms'];self.fields=z['Z'];self.D=1-np.average(self.fields[:,self.A],axis=1,weights=s.sizes[self.A])
    def field(self,D):
        possible=np.flatnonzero((self.D[:-1]<=D)&(self.D[1:]>=D)&(self.time[:-1]>=9000))
        assert len(possible),D
        j=int(possible[0]);a=(D-self.D[j])/(self.D[j+1]-self.D[j])
        z=self.background.copy();z[self.A]=(1-a)*self.fields[j,self.A]+a*self.fields[j+1,self.A]
        return z,float((1-a)*self.time[j]+a*self.time[j+1])
    def set(self,D):
        z,tm=self.field(D);self.s.set_Z(z);return tm
    def evaluate(self,q,D,jac=True):
        self.set(D);value=value_and_jacobian(self.s,q,jac)
        if not jac:return value[1]
        h=1e-6
        self.set(D+h);plus=value_and_jacobian(self.s,q,False)[1]
        self.set(D-h);minus=value_and_jacobian(self.s,q,False)[1]
        self.set(D);return value[1],value[3],(plus-minus)/(2*h)


def main(a):
    source=Path(a.source);data=np.load(source);s=PhysicalDelayConditionalDrift();s.set_Z(data['Z']);r=data['r']
    assert abs(s.residual(r)).max()<1e-11
    family=Family(s);D=float(1-np.average(s.Z[family.A],weights=s.sizes[family.A]));family.set(D)
    assert np.max(abs(data['Z']-s.Z))<2e-12
    DEST.mkdir(exist_ok=True);assert not (DEST/'jobs.json').exists();P=s.P;weight=50.
    write(DEST/'contract.json',dict(source=str(source),question='Does an actual equilibrium branch turn or change temporal stability in the Core A local transition interval?',
        parameter='D_A=1-cell-weighted mean Z_A; original native within-Core-A field interpolation, outside-Core-A Z bitwise native9s. No global-mean resource path or uniform-core replacement.',
        equations='Current physical private-Q conditional drift; full spatial network, static constraint M=.5*E*r corresponding to dynamic M. Locked transient correction is zero at stationarity.',
        method='Pseudo-arclength in logit equilibrium coordinates and physical D_A. Original static residual<1e-11/ms at every accepted point. Tangent sign changes only nominate an equilibrium fold, requiring separate nullvector, nondegeneracy and temporal checks.',
        limits='D_A0.26to0.405, maximum100accepted/160trials. No inference that an unstable equilibrium fold is the observed aperiodic transition without correspondence.',model_promoted=False))
    def tangent(y,previous=None):
        F,J,col=family.evaluate(y[:-1],y[-1])
        if previous is None:v=np.r_[spsolve(J,col),-1.]
        else:
            row=np.r_[previous[:-1]/P,weight*weight*previous[-1]]
            B=sparse.vstack([sparse.hstack([J,sparse.csc_matrix(col[:,None])]),sparse.csc_matrix(row[None,:])],format='csc')
            v=spsolve(B,np.r_[np.zeros(P),1.])
        return v/np.sqrt(v[:-1]@v[:-1]/P+(weight*v[-1])**2)
    y=np.r_[data['q'],D];v=tangent(y);ds=.1;rows=[];trials=[];start=time.time();reason='BUDGET'
    def save(y,v):
        tm=family.set(y[-1]);rr,ff,op=value_and_jacobian(s,y[:-1],False);err=float(abs(s.residual(rr)).max());assert err<1e-11,err
        row=dict(index=len(rows),D_A=float(y[-1]),Z_A=float(1-y[-1]),native_time_ms=tm,
            regional_rates_hz=s.regional_rates(rr),global_rate_hz=s.global_rate(rr),residual_per_ms=err,
            D_tangent=float(v[-1]),seconds=time.time()-start,stability='NOT_COMPUTED')
        np.savez_compressed(DEST/f'point{len(rows):03d}.npz',q=y[:-1],r=rr,Z=s.Z,D_A=y[-1],tangent=v)
        rows.append(row);write(DEST/'accepted.json',rows);write(DEST/'jobs.json',dict(status='RUNNING',pid=os.getpid(),latest=row));log('CORE A EQUILIBRIUM BRANCH',row)
    save(y,v)
    for trial in range(160):
        pred=y+ds*v;x=pred.copy();arcrow=np.r_[v[:-1]/P,weight*weight*v[-1]];success=False;trace=[]
        if not .26<x[-1]<.405:reason='DECLARED_LOCAL_INTERVAL';break
        for it in range(14):
            F,J,col=family.evaluate(x[:-1],x[-1]);arc=float(arcrow@(x-pred));err=float(np.sqrt(F@F/P+arc*arc));trace.append(err)
            if max(abs(F).max(),abs(arc))<1e-10:success=True;break
            B=sparse.vstack([sparse.hstack([J,sparse.csc_matrix(col[:,None])]),sparse.csc_matrix(arcrow[None,:])],format='csc');change=spsolve(B,-np.r_[F,arc]);alpha=min(1.,3/max(abs(change[:-1]).max(),1e-12))
            for back in range(18):
                nxt=x+alpha*change
                if .2558<nxt[-1]<.41:
                    f=family.evaluate(nxt[:-1],nxt[-1],False);ar=float(arcrow@(nxt-pred))
                    if np.sqrt(f@f/P+ar*ar)<err:x=nxt;break
                alpha*=.5
            else:break
        trials.append(dict(trial=trial,accepted=success,D_A=float(x[-1]),step=ds,errors=trace));write(DEST/'trials.json',trials)
        if not success:
            ds*=.5
            if ds<1e-4:reason='NUMERICAL_MIN_STEP';break
            continue
        previous_v=v.copy();y=x;v=tangent(y,v);save(y,v)
        if v[-1]*previous_v[-1]<0:
            write(DEST/'fold_candidate.json',dict(status='TANGENT_TURN_ONLY_NOT_CERTIFIED',points=[len(rows)-2,len(rows)-1],D_A=[rows[-2]['D_A'],rows[-1]['D_A']]))
            reason='FOLD_CANDIDATE_FOR_REFINEMENT';break
        if len(rows)>=100:break
        ds=min(.3,ds*(1.3 if len(trace)<=4 else (1.05 if len(trace)<=7 else .7)))
    write(DEST/'result.json',dict(status='COMPLETE',stop_reason=reason,accepted_points=len(rows),trials=len(trials),bifurcation_type='NOT_ESTABLISHED',model_promoted=False))
    write(DEST/'jobs.json',dict(status='COMPLETE',pid=os.getpid(),stop_reason=reason))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');main(p.parse_args())
