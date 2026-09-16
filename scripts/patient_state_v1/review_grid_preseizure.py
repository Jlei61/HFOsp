"""Compare full-joint-grid and Gaussian pre-seizure state reconstructions."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
ROOT=RUN/'joint_grid_preseizure_v1_25'

def main():
    assert json.loads((ROOT/'status.json').read_text())['status']=='COMPLETE';rs=[json.loads(p.read_text()) for p in (ROOT/'evaluations').glob('*.json')];assert len(rs)==48;rows=[]
    for r in rs:
        row={k:r[k] for k in ['sz','label','coupled','grid','max_mass_loss','max_edge_mass','max_adf_state_difference','max_adf_probability_difference']}
        for key,value in r['summary'].items():
            for window,v in value.items():row[f'{key}_{window}']=v
        rows.append(row)
    df=pd.DataFrame(rows).sort_values(['sz','coupled','grid']);df.to_csv(ROOT/'resolution_readouts.csv',index=False);checks=[]
    for (sz,c),part in df.groupby(['sz','coupled']):
        x=part.set_index('grid');a=np.load(ROOT/'evaluations'/f'sz{sz}_c{int(c)}_g256.npz');b=np.load(ROOT/'evaluations'/f'sz{sz}_c{int(c)}_g512.npz');assert np.array_equal(a['times'],b['times']);ok=a['covered'];checks.append(dict(sz=int(sz),coupled=bool(c),max_state_difference=float(np.max(abs(a['mode_state'][ok]-b['mode_state'][ok]))),max_probability_difference=float(np.max(abs(a['uniform_tb_probability'][ok]-b['uniform_tb_probability'][ok]))),pre15_state_difference=float(abs(x.loc[256,'mode_state_pre15']-x.loc[512,'mode_state_pre15'])),pre15_probability_difference=float(abs(x.loc[256,'uniform_tb_probability_pre15']-x.loc[512,'uniform_tb_probability_pre15']))))
    write_json(ROOT/'scientific_audit.json',dict(status='COMPLETE',resolution_checks=checks,maximum_mass_loss=float(df.max_mass_loss.max()),maximum_edge_mass=float(df.max_edge_mass.max()),state_interval='Full grid posterior 2.5/97.5 percentiles at fixed preinterval fitted parameters, excluding parameter uncertainty',query='Completed bins only; exact OU finite-volume propagation to the query. No future seizure type or observations used.',limits='No biological drift/reset identification; development labels, two TB-source seizures, and recording-stage confounding remain. Parameter fitting itself was Laplace; this audit changes filtering only.'))
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for col,sz in enumerate([19,22]):
        z=np.load(ROOT/'evaluations'/f'sz{sz}_c1_g512.npz');x=z['minutes'];mask=z['covered'];curves={k:np.where(mask,z[k],np.nan) for k in ['mode_state','mode_state_lower','mode_state_upper','uniform_tb_probability','adf_mode_state','adf_uniform_tb_probability']}
        axs[0,col].fill_between(x,curves['mode_state_lower'],curves['mode_state_upper'],color='#2166ac',alpha=.12,label='Grid state 95% interval');axs[0,col].plot(x,curves['adf_mode_state'],color='gray',lw=1,label='Gaussian approximation');axs[0,col].plot(x,curves['mode_state'],color='#2166ac',lw=1.2,label='Full joint grid');axs[1,col].plot(x,curves['adf_uniform_tb_probability'],color='gray',lw=1);axs[1,col].plot(x,curves['uniform_tb_probability'],color='#2166ac',lw=1.2);axs[0,col].set_title(f'SZ{sz} | TB-source')
        for ax in axs[:,col]:ax.axvline(-15,color='gray',ls=':',lw=.8);ax.spines[['top','right']].set_visible(False);ax.set_xlim(-60,0)
        axs[1,col].set_xlabel('Minutes before seizure')
    axs[0,0].set_ylabel('Filtered mode state (TB log-odds)');axs[1,0].set_ylabel('Time-conditioned TB probability');axs[0,0].legend(fontsize=8);fig.suptitle('Does state-filtering approximation change the two TB-case trends?\n512 x 512 joint posterior; preinterval parameters fixed; state intervals exclude parameter uncertainty',fontsize=11);fig.tight_layout(rect=(0,0,1,.93))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'joint_preseizure_grid_audit.{ext}',dpi=180)
    plt.close(fig)
    readme=RUN/'figures/README.md'
    if '### joint_preseizure_grid_audit.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### joint_preseizure_grid_audit.png\n对两次TB型发作，比较相同发作前区间参数下的高斯近似与512×512完整联合状态过滤。阴影来自完整网格状态分布的2.5/97.5分位数，不包含参数不确定性，缺失覆盖保留为空白。\n**关注点**：两例状态变化方向是否依赖过滤近似；重建曲线仍不能直接当作SDE的drift力或发作预测。\n')
    print(df[(df.grid==512)&(df.label=='TB')].to_string(index=False));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
