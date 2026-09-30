"""Export contact and conditional pair summaries with their actual denominators."""
from common import *
import csv

def main():
    contacts=[];pairs=[];names=read(V10/'native/a/observation_contract.json')['contact_names']
    for p in sorted((OUT/'summaries').glob('*.json')):
        q=read(p);s=q['summary']
        for j,name in enumerate(names):
            c=s['contacts'][name]
            contacts.append(dict(run=p.stem,N_valid_events=q['N'],contact=name,
                n_participating_events=c['n'],participation=c['participation'],mean_normalized_rank=s['mean_rank'][j],
                rank_q05=s['rank_q05'][j],rank_q95=s['rank_q95'][j]))
        for key,v in s['pairs'].items():
            pairs.append(dict(run=p.stem,N_valid_events=q['N'],pair=key,shaft=v['shaft'],
                n_joint_events=v['n'],joint_fraction=v['joint_fraction'],order_probability=v['order_probability'],
                median_lag_ms=v['median']))
    for name,rows in [('contact_observables.csv',contacts),('within_shaft_observables.csv',pairs)]:
        with (OUT/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(len(contacts),'contact rows;',len(pairs),'conditional pair rows',flush=True)

if __name__=='__main__':main()
