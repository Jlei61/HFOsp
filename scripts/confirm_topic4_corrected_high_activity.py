#!/usr/bin/env python3
"""Finite confirmation/follow-up package selected from the completed first screen."""
from screen_topic4_corrected_high_activity import OUT, run, write
from concurrent.futures import ProcessPoolExecutor, as_completed


def main():
    tau=20.611550480127335
    jobs=[(.5,tau,20,6000.,.1,'high'),(.5,tau,10,6000.,.05,'high'),
          (.5,tau,10,6000.,.1,'low'),(.5,tau,10,6000.,.1,'high'),
          *[(q,tau,10,3000.,.1,'high') for q in (.30,.35,.40,.45)],
          (.5,9.,10,6000.,.1,'high')]
    write(OUT/'confirmation_protocol.json',{'selection':'q=.5 at original GABA decay shows stable large cycles but only four peaks in short screen; 9ms candidate is secondary comparison',
        'jobs':[dict(zip(['q','tau','grid','duration','dt','initial'],j)) for j in jobs],
        'interpretation':'A periodic train of separated bursts is distinguished from high-rate small ripple and sustained non-decaying modulation; no seizure/Hopf claim from periodicity alone',
        'SNN_pair':{'q':.5,'tau_gaba_ms':[9.,tau],'duration_ms':3000.,'dynamics_seed':9108301,'input':'OU off, physical Poisson on, native cold start'}})
    completed=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(run,*j) for j in jobs]
        for f in as_completed(fs):
            row=f.result();completed.append(row['name']);write(OUT/'confirmation_status.json',{'status':'RUNNING','completed':completed,'total':len(jobs)});print(row['name'],flush=True)
    write(OUT/'confirmation_status.json',{'status':'COMPLETE','completed':completed,'total':len(jobs)})


if __name__=='__main__':main()
