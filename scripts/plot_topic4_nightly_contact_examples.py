from pathlib import Path
import json,pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.analyze_topic4_observable_loss_physical_pilot import native_unit
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/topic4_sef_hfo/nightly_central_workpoint'
def main():
 old=pickle.load(open(ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1/training_objective_v2_1.pkl','rb'))
 patient=json.loads((ROOT/'results/topic4_sef_hfo/contact_event_objective_revision_v2/manifest.json').read_text())['patient_training']
 for batch,cid in [('A','central_A_anchor1_geom1'),('B','central_B_GABA_long'),('B','broad_B_geometry')]:
  rows=json.loads((OUT/f'batches/{batch}/scores.json').read_text())['candidates'];r=next(r for r in rows if r['candidate_id']==cid)
  clips=[]
  for uid,u in sorted(r['units'].items()):
   _,cl,_=native_unit(Path(u['worker_path']),r['candidate'],old);clips.extend((uid,g) for g in cl)
  for mode in [1,0]:
   selected=[(uid,g) for uid,g in clips if g['event']['mode']==mode];pp=[p for p in patient if p['mode']==mode]
   fig,axs=plt.subplots(len(selected),2,figsize=(11,2.5*len(selected)),squeeze=False)
   for j,(uid,g) in enumerate(selected):
    p=pp[j%len(pp)]
    with np.load(p['arrays_path']) as z:
     names=z['contact_names'].astype(str).tolist();ix=[names.index(n) for n in g['names']];win=z['packed_window_mask'].astype(bool);pm=z['positive_envelope_mass'][ix][:,win].astype(float);pt=z['time_ms'][win].astype(float);pt-=pt[0]
    order=sorted(range(15),key=lambda i:(not g['names'][i].startswith('SCL'),-int(''.join(filter(str.isdigit,g['names'][i])))))
    for ax,m,t,title in [(axs[j,0],pm,pt,f"Patient event {p['raw_global_event_index']}"),(axs[j,1],g['mass'],(np.arange(g['mass'].shape[1])+.5)*2,f"Model event {g['event']['event_id']} | {uid}")]:
     ax.pcolormesh(t,np.arange(15),m[order]/max(float(m.max()),1e-12),shading='nearest',cmap='magma',vmin=0,vmax=1,rasterized=True)
     ax.set(xlim=(0,250),ylim=(14.5,-.5),yticks=range(15),yticklabels=[g['names'][i] for i in order],title=title,xlabel='Window time (ms)',facecolor='black');ax.tick_params(labelsize=6);ax.axhline(3.5,color='white',lw=.5,alpha=.5)
   tag='TA' if mode else 'TB';fig.suptitle(f'{cid}: {tag}-labelled multiple events\nFixed contacts and real time; one maximum per event. Examples are not patient-model event pairs.',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
   for ext in ['png','pdf']:fig.savefig(OUT/f'figures/{cid}_{tag}_contact_comparison.{ext}',dpi=150,bbox_inches='tight')
   plt.close(fig)
if __name__=='__main__':main()
