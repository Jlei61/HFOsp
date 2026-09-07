#!/usr/bin/env python3
"""Describe short-run geometry ranking reproducibility; no acceptance changes."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import spearmanr
from scripts.run_topic4_xy_research import read,write,sha


def main():
    root=ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v2';source=root/'baseline_scores.json'
    data=read(source)['candidates']
    rows=[r for r in data if len(r['units'])==2 and all(u['metrics']['joint_distance'] is not None for u in r['units'])]
    out={'n_candidates':len(rows),'n_networks':2,'descriptive_only':True,
         'diagnostic':'Geometry ranking under two common network/dynamics seeds; dependent comparisons, not population inference.',
         'source_hashes':{str(p):sha(p) for p in [source,Path(__file__)]}}
    for metric in ['joint_distance','exploration_score','D_order','D_lag']:
        valid=[r for r in rows if all(u['metrics'][metric] is not None for u in r['units'])]
        x=np.array([r['units'][0]['metrics'][metric] for r in valid]);y=np.array([r['units'][1]['metrics'][metric] for r in valid])
        a=set(np.argsort(x)[:10]);b=set(np.argsort(y)[:10])
        out[metric]={'n_candidates':len(valid),'rank_correlation':float(spearmanr(x,y).statistic),'top10_overlap':len(a&b),
            'median_absolute_seed_difference':float(np.median(abs(x-y))),
            'iqr_between_geometry_mean':float(np.subtract(*np.quantile((x+y)/2,[.75,.25])))}
    write(root/'short_run_seed_reproducibility.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
