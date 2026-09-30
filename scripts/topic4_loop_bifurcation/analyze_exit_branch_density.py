#!/usr/bin/env python3
"""Matched first-ten-second native/candidate branch-point comparison."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import time
import numpy as np
from campaign import ROOT, read, write, sha
from exit_branch_density import OUT, NAMES
from coupled_density_exit import ADAPTED
from analyze_native import activity_censor, original


def native_prefix(name):
    root = ROOT/'exit_return_probes'
    job = read(root/'jobs'/f'{name}.json')
    with np.load(root/'extended_analysis'/f'{name}_readouts.npz') as z:
        rate5 = z['rate_5ms_Hz'][:2000]
        fields = z['field_rate_5ms_Hz'][:2000]
    start = job['branch_start_s']*1000
    data = original.load(root/'runs'/name/'mechanism_chunks', ['time_ms', 'global_E_rate_Hz', 'global_raw_conductance_ratio'])
    keep = (data['time_ms'] >= start) & (data['time_ms'] < start+10000)
    assert keep.sum() == 10000 and rate5.shape == (2000, 4)
    return dict(name=name, source='Native', rate5=rate5, fields=fields,
                R=data['global_E_rate_Hz'][keep], Graw=data['global_raw_conductance_ratio'][keep], global_time_ms=data['time_ms'][keep]-start)


def candidate(name, geo):
    with np.load(OUT/name/'trajectory.npz') as z:
        E = geo['population'] == 0
        masks = [E] + [E & (geo['group_region'] == q) for q in range(3)]
        rates = z['group_rate_Hz'].astype(float)
        regional = np.stack([np.average(rates[:, mask], weights=geo['group_size'][mask], axis=1) for mask in masks], axis=1)
        return dict(name=name, source='Density', rate5=regional.reshape(2000, 5, 4).mean(1),
            fields=z['field_E_Hz'].reshape(2000, 5, 400).mean(1), R=z['global_R_Hz'], Graw=30*z['global_s'], global_time_ms=z['elapsed_time_ms'])


def summarize(d, counts):
    rates = d['rate5'].reshape(1000, 2, 4).mean(1)
    events = original.event_audit.events(rates[:, :3].max(1), end=10.)
    events = [e for e in events if rates[round(e['start_s']*100):round(e['end_s']*100), 0].max() >= 20]
    windows = []
    for lo, hi in [(0, 5), (5, 10)]:
        r = rates[lo*100:hi*100]
        field = d['fields'][lo*200:hi*200].mean(0)
        keep = (d['global_time_ms'] >= lo*1000) & (d['global_time_ms'] < hi*1000)
        assert np.isclose(np.average(field, weights=counts), r[:, 0].mean(), atol=2e-4, rtol=1e-6)
        brief = [e for e in events if e['start_s'] >= lo and e['end_s'] <= hi and .02-1e-9 <= e['duration_s'] <= .2+1e-9]
        windows.append(dict(interval_s=[lo, hi], mean_Hz_allE_A_B_other=r.mean(0).tolist(),
            allE_high_fraction=float((r[:, 0] >= 200).mean()), joint_quiet_fraction=float((r[:, :3] < 5).all(1).mean()),
            complete_brief_events=len(brief), mean_field_Hz=field.tolist(),
            Graw_mean=float(d['Graw'][keep].mean()), Graw_min=float(d['Graw'][keep].min()), Graw_max=float(d['Graw'][keep].max())))
    return dict(source=d['source'], windows=windows, complete_events=events, censoring=activity_censor(rates))


def main(wait):
    geo = dict(np.load(ADAPTED/'geometry.npz'))
    counts = np.bincount(geo['group_cell'][geo['population'] == 0], weights=geo['group_size'][geo['population'] == 0], minlength=400)
    late = {x['name']: x for x in read(ROOT/'exit_return_probes/extended_analysis_summary.json')['rows']}
    done = []; rows = []; data = {}
    while len(done) < 4:
        for name in NAMES:
            result = OUT/name/'result.json'
            if name in done or not result.exists():
                continue
            assert read(result)['status'] == 'COMPLETE' and read(result)['held_Z_K_bitwise']
            n, c = native_prefix(name), candidate(name, geo)
            nn, cc = summarize(n, counts), summarize(c, counts)
            differences = []
            for a, b in zip(nn['windows'], cc['windows']):
                delta = np.array(b['mean_field_Hz'])-a['mean_field_Hz']
                differences.append(dict(interval_s=a['interval_s'], rate_delta_Hz_allE_A_B_other=(np.array(b['mean_Hz_allE_A_B_other'])-a['mean_Hz_allE_A_B_other']).tolist(),
                    weighted_mean_field_RMS_Hz=float(np.sqrt(np.average(delta**2, weights=counts))),
                    weighted_mean_field_MAE_Hz=float(np.average(abs(delta), weights=counts))))
            row = dict(name=name, native_prefix=nn, density_prefix=cc, matched_window_differences=differences,
                original_native20_30s_tail=dict(mean_Hz=late[name]['tail_mean_Hz'], brief_events=late[name]['tail_brief_events'],
                    quiet_fraction=late[name]['tail_joint_quiet_fraction'], not_a_matched_candidate_window=True))
            write(OUT/name/'comparison.json', row)
            rows.append(row); done.append(name); data[name] = (n, c)
            print('EXIT BRANCH COMPARISON', name, nn['windows'][-1]['mean_Hz_allE_A_B_other'], cc['windows'][-1]['mean_Hz_allE_A_B_other'], flush=True)
        write(OUT/'comparison.json', dict(status='COMPLETE' if len(done) == 4 else 'PARTIAL', completed=done, rows=rows,
            native_correspondence_certified=False, formal_bifurcation_allowed=False, producer_sha256=sha(__file__),
            interpretation='Fourconditionalpoint correspondence screen. Ten-second matchedprefix plusnative30s tails displayedseparately. Onepairednumericalstream, notstability/continuation or numericalconvergence.'))
        if len(done) < 4:
            if not wait:
                return
            write(OUT/'analysis_progress.json', dict(status='WAITING_FIXED_FOUR', pid=os.getpid(), completed=done, updated_epoch=time.time()))
            time.sleep(30)
    plot(data, geo)
    write(OUT/'analysis_progress.json', dict(status='COMPLETE', completed=done, updated_epoch=time.time()))


def plot(data, geo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), layout='constrained', gridspec_kw={'height_ratios':[.8,1,1]})
    for col, name in enumerate(NAMES):
        native, density = data[name]
        job = read(ROOT/'exit_return_probes/jobs'/f'{name}.json')
        for d, color in [(native, 'black'), (density, '#267f8e')]:
            axes[0,col].plot(d['global_time_ms']/1000, d['R'], color=color, lw=1, label=d['source'])
        axes[0,col].set_title(f'K={job["target_K"]:g}, {job["source_history"]} history', weight='bold', fontsize=10)
        axes[0,col].set_xlim(0,10);axes[0,col].set_ylim(-10,500);axes[0,col].set_xlabel('Time since clamp (s)')
        if col == 0: axes[0,col].set_ylabel('Causal E rate (Hz)'); axes[0,col].legend(frameon=False, fontsize=8)
        for row, d in enumerate([native,density],start=1):
            ax=axes[row,col];field=d['fields'][1000:2000].mean(0).reshape(20,20)
            im=ax.imshow(field,origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500)
            for center in geo['centers_mm']:ax.add_patch(Circle(center,1.5,fill=False,color='#00bec7',lw=.8))
            ax.set_title(d['source']+' mean, 5–10 s',fontsize=10);ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            if col==0:ax.set_ylabel('y (mm)')
            if row==2:ax.set_xlabel('x (mm)')
    fig.colorbar(im,ax=axes[1:].ravel().tolist(),shrink=.85,pad=.02,label='E population rate (Hz)')
    fig.suptitle('Actual exit spatial fields: local correspondence before continuation',weight='bold')
    fig.text(.5,-.012,'Held Z mean=.21 and original16.7-s spatial Z/K shapes. G/M and recurrent activity dynamic. Four conditional points; no stability certification.',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'exit_branch_density.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);write(OUT/'figure_metadata.json',dict(agent_visual='PENDING',human_review='PENDING',producer_sha256=sha(__file__)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');main(parser.parse_args().wait)
