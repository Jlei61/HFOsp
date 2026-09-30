"""Stage B summary figure and table: Z-field family along the actual entry (D(t), spatial RMS to the
prescribed path), the three control families (a: carried state x Z field; b: carried vs relaxed at same Z;
c: conditional attractor from relaxed state) and the alignment with the full Z/M trajectory."""
from common_v3 import *
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
OUT=DEST/'stage_b'
def main():
    b1=read(OUT/'b1_z_family.json');b2=read(OUT/'b2_controls.json');entry=b1['entry_ms']
    traj=np.load(DEST/'runs'/b1['source']/'trajectory.npz');tD=np.arange(len(traj['D']))*10.;D=traj['D'];g=traj['global_E_hz']
    cat_color={'LOW':'0.7','LOCAL_SELF_LIMITED':'tab:green','GLOBAL_SYNCHRONOUS_BURSTS':'tab:orange','SUSTAINED_PARTIAL':'tab:red','SUSTAINED_BROAD':'darkred'}
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    ax=axes[0,0];ax.plot(traj['time_ms']/1000,g,lw=.5,color='k');ax.set_ylabel('all-E rate (Hz)');ax.set_xlabel('time (s)');ax.axvline(entry/1000,color='tab:red',ls='--')
    ax2=ax.twinx();ax2.plot(tD/1000,D,color='tab:blue');ax2.set_ylabel('D',color='tab:blue');ax.set_title('Deterministic skeleton: global rate and D(t); dashed = entry (>=200 Hz for 200 ms)',fontsize=9)
    ax=axes[0,1];r=b1['rate_model'];ax.plot([q['D'] for q in r],[q['spatial_rms_to_power_path_same_D'] for q in r],'o-',label='rate-model entry fields');n=b1['native'];ax.plot([q['D'] for q in n],[q['spatial_rms_to_power_path_same_D'] for q in n],'s-',label='native checkpoint fields')
    ax.set_xlabel('D');ax.set_ylabel('spatial RMS of Z to prescribed path at equal D');ax.legend(fontsize=8);ax.set_title('B1: actual Z fields vs prescribed power path',fontsize=9)
    ax=axes[1,0]
    for i,q in enumerate(b2['results']['a']):ax.scatter(q['D'],0,color=cat_color[q['category']],s=80,marker='o');ax.text(q['D'],0.05,str(q['t_Z']),fontsize=6,rotation=90,ha='center')
    for q in b2['results']['b']:ax.scatter(q['D'],1 if q['state']=='carried' else 2,color=cat_color[q['category']],s=80,marker='s')
    for q in b2['results']['c']:ax.scatter(q['D'],3 if not str(q['field']).startswith('native') else 4,color=cat_color[q['category']],s=80,marker='^')
    ax.set_yticks([0,1,2,3,4]);ax.set_yticklabels([f'(a) carried state @{b2["t_ref"]} ms, Z from t_Z','(b) carried state, own Z','(b) relaxed state, same Z','(c) relaxed, long run (rate-model Z)','(c) relaxed, long run (native Z)'],fontsize=8)
    ax.set_xlabel('D of the frozen Z field');ax.set_title('B2 controls (colour = category in the last 2 s)',fontsize=9)
    for k,c in cat_color.items():ax.scatter([],[],color=c,label=k)
    ax.legend(fontsize=7,loc='upper left');ax.axvline(float(D[min(len(D)-1,int(entry/10))]),color='tab:red',ls='--')
    ax=axes[1,1];
    for q in b2['results']['c']:ax.scatter(q['D'],q['occupation_50Hz'],color=cat_color[q['category']],marker='^' if not str(q['field']).startswith('native') else 'v');
    ax.set_xlabel('D of frozen field');ax.set_ylabel('occupation (>50 Hz, last 2 s)');ax.set_title('(c) conditional attractor occupation',fontsize=9)
    fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/stage_b_summary.png',dpi=140);plt.close(fig)
    # table
    rows=[]
    for fam,lst in b2['results'].items():
        for q in lst:rows.append(dict(family=fam,label=q['label'],D=round(q['D'],4),category=q['category'],occupation=round(q['occupation_50Hz'],3),longest_quiet_ms=q['longest_quiet_ms'],tail_mean_hz=round(q['tail_mean_hz'],1),high_onset_ms=q['high_onset_ms']))
    write(OUT/'summary_table.json',dict(entry_ms=entry,D_at_entry=float(D[min(len(D)-1,int(entry/10))]),rows=rows));print(json.dumps(rows,indent=0)[:4000])
if __name__=='__main__':main()
