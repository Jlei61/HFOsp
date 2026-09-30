"""Stage A4 comparison: native SNN reference vs v3 rate-model runs (same readouts, same windows).

Tables: event statistics per window (count, duration, gap, core participation, area, direction),
high-activity onset, expansion times, Z depletion (D(t)). Figure: global E rate traces (native both
seeds, rate deterministic, rate stochastic), D(t), and 2-D field snapshots at matched times.
Tolerance basis: native seed-to-seed differences (registered in a4_contract.json before comparison).
"""
from common_v3 import *
from native_readouts import readouts,window_stats,cell_xy,AXIS
import argparse,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
geo=dict(np.load(OPERATORS/'g20/geometry.npz'))
def load_run(label):
    z=np.load(DEST/f'runs/{label}/trajectory.npz');res=read(DEST/f'runs/{label}/result.json');return z,res
def native_summary():
    return read(DEST/'native_reference/summary.json')
def native_traces(label):
    z=np.load(DEST/f'native_reference/{label}_readouts.npz');return z['t'],z['allE'],z['rate_cells']
def native_D(run):
    import glob
    ts=[];Ds=[]
    for f in sorted(glob.glob(str(run/'fields/*.npz'))):
        q=np.load(f);ts.append(q['zm_step']*.1);Ds.append(1-q['z'].mean(1))
    return np.concatenate(ts),np.concatenate(Ds)
def main(a):
    out=DEST/'a4_comparison';out.mkdir(exist_ok=True);fig_dir=out/'figures';fig_dir.mkdir(exist_ok=True)
    nat=native_summary();rows={}
    for lab in ['seed9108401','seed9108402']:rows['native_'+lab]=nat[lab]
    runs={}
    for lab in a.runs:
        try:z,res=load_run(lab);runs[lab]=(z,res);rows['rate_'+lab]=res
        except FileNotFoundError:log('missing run',lab)
    # ---- table
    keys=['n','median_duration_ms','median_gap_ms','both_cores','median_area','median_surround_cells','forward','reverse','median_extent_mm']
    table={}
    for name,r in rows.items():
        table[name]={'high_onset_ms':r.get('high_onset_ms'),'quiet_fraction':r.get('quiet_fraction'),'mean_allE_hz':r.get('mean_allE_hz'),
                     'expansion_first_ms':r.get('expansion_first_ms'),'windows':{w:{k:r['windows'][w].get(k) for k in keys} for w in ['1000-4000','4000-8000','8000-9420','1000-9420'] if w in r['windows']}}
    # tolerance basis: native seed differences
    n1,n2=nat['seed9108401'],nat['seed9108402'];basis={}
    for w in ['1000-9420','4000-8000']:
        basis[w]={k:[n1['windows'][w].get(k),n2['windows'][w].get(k)] for k in keys}
    write(out/'comparison_table.json',dict(table=table,native_seed_basis=basis,definitions='native_readouts.py milestone definitions; identical code applied to native cell counts and rate-model cell fields'))
    # ---- figure: traces + D + fields
    fig=plt.figure(figsize=(16,10));gs=fig.add_gridspec(4,6,height_ratios=[1.2,1.2,1,1.4])
    ax=fig.add_subplot(gs[0,:]);tn,an,_=native_traces('seed9108401');ax.plot(tn/1000,an,lw=.5,color='k',label='native SNN seed 9108401')
    colors={'A4_det_meandrive':'tab:red','A4_stoch_seed9108401':'tab:blue','A4_stoch_meandrive':'tab:green'}
    for lab,(z,res) in runs.items():ax.plot(z['time_ms']/1000,z['global_E_hz'],lw=.5,color=colors.get(lab,'gray'),label=f'rate {lab}',alpha=.8)
    ax.set_ylabel('all-E rate (Hz/neuron)');ax.set_xlim(0,12.5);ax.legend(loc='upper left',fontsize=8);ax.set_title('Global E rate, common physical start (t=0), same external drive definition')
    ax2=fig.add_subplot(gs[1,:]);ax2.plot(tn/1000,an,lw=.6,color='k')
    for lab,(z,res) in runs.items():ax2.plot(z['time_ms']/1000,z['global_E_hz'],lw=.6,color=colors.get(lab,'gray'),alpha=.8)
    ax2.set_xlim(6,10.5);ax2.set_ylim(0,300);ax2.set_ylabel('all-E rate (Hz)');ax2.set_xlabel('time (s)');ax2.set_title('Entry window 6-10.5 s')
    ax3=fig.add_subplot(gs[2,:3]);tD,Dn=native_D(NATIVE);ax3.plot(tD/1000,Dn,color='k',label='native D=1-<Z_E>')
    for lab,(z,res) in runs.items():ax3.plot(np.arange(len(z['D']))*10/1000,z['D'],color=colors.get(lab,'gray'),label=lab)
    ax3.set_xlabel('time (s)');ax3.set_ylabel('D');ax3.legend(fontsize=7);ax3.set_xlim(0,12.5)
    ax4=fig.add_subplot(gs[2,3:]);
    for name,r in rows.items():
        w=r['windows'].get('1000-9420',{});ax4.bar(name,w.get('n',0),color='k' if name.startswith('native') else colors.get(name[5:],'gray'))
    ax4.set_ylabel('events 1-9.42 s');ax4.tick_params(axis='x',labelrotation=20,labelsize=7)
    # field snapshots: native at 5 times, each run at the same times
    times=[3000,6000,9000,9800,10300,12000];xy=cell_xy();_,_,rc=native_traces('seed9108401')
    items=[('native',tn,rc)]+[(lab,z['time_ms'],z['field_E_hz']) for lab,(z,res) in runs.items()]
    for j,t0 in enumerate(times):
        for i,(name,tt,field) in enumerate(items[:3]):
            axf=fig.add_subplot(gs[3,j]) if i==0 else None
            if i>0:break
            k=np.searchsorted(tt,t0);img=field[max(0,k-50):k+50].mean(0).reshape(20,20)
            axf.imshow(img,origin='lower',vmin=0,vmax=300,cmap='magma',extent=[0,20,0,20]);axf.set_title(f'native {t0/1000:.1f} s',fontsize=8);axf.set_xticks([]);axf.set_yticks([])
    fig.tight_layout();fig.savefig(fig_dir/'a4_global_and_D.png',dpi=130);plt.close(fig)
    # separate field panel figure: rows = native + runs, cols = times
    nrow=len(items);fig,axes=plt.subplots(nrow,len(times),figsize=(2.3*len(times),2.3*nrow))
    for i,(name,tt,field) in enumerate(items):
        for j,t0 in enumerate(times):
            k=np.searchsorted(tt,t0);img=field[max(0,k-50):k+50].mean(0).reshape(20,20);axf=axes[i,j]
            im=axf.imshow(img,origin='lower',vmin=0,vmax=300,cmap='magma',extent=[0,20,0,20]);axf.set_xticks([]);axf.set_yticks([])
            if i==0:axf.set_title(f'{t0/1000:.1f} s (100 ms mean)',fontsize=8)
            if j==0:axf.set_ylabel(name,fontsize=8)
    fig.colorbar(im,ax=axes,shrink=.6,label='E rate (Hz/neuron)');fig.savefig(fig_dir/'a4_field_snapshots.png',dpi=130);plt.close(fig)
    log('wrote',out)
    print(json.dumps(clean(table),indent=1)[:6000])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',default=['A4_det_meandrive','A4_stoch_seed9108401','A4_stoch_meandrive']);main(p.parse_args())
