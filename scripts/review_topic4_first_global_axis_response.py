"""First paired 20s angle response, including the predeclared rotation screen."""
from pathlib import Path
import sys,json,hashlib,csv,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import analyze_topic4_shape_output_response as an
from scripts.render_topic4_shape_output_gifs import rotation_clip
from src.topic4_pdf_font_guard import install
P=Path('/data/hfosp/topic4_sef_hfo/global_axis_residual_probe_20260913')
OUT=P/'analysis/first_angle_review'

def main():
    rt=an.rt;F=OUT/'figures';F.mkdir(parents=True,exist_ok=True);records={};sources=[]
    for p in (P/'analysis/units').glob('*/result.json'):
        r=rt.read(p);c=r['counts']
        if c['candidate'] in ['global_axis_+0','global_axis_-15'] and c['noise']==847101:
            records[c['candidate']]=r;sources.append(dict(path=str(p),sha256=rt.sha(p)))
    assert len(records)==2;rows=[]
    for cid,r in records.items():
        c=r['counts'];obs={x['mode']:x for x in r['observations'] if x['layer']=='primary'}
        for mode in ['ALL','TA','TB']:
            pair=next(x for x in r['pairs'] if x['mode']==mode and x['layer']=='primary' and x['contact_i']=='ICL11' and x['contact_j']=='ICL9')
            rows.append(dict(candidate=cid,noise=847101,mode=mode,n=obs[mode]['n'],fraction=obs[mode]['n']/c['primary'],upper=obs[mode]['SCL_upper_participation'],participation_error=obs[mode]['participation_mae'],order_error=obs[mode]['pair_order_probability_mae'],rod_median_ms=obs[mode]['SCL_minus_ICL_lag_median_ms'],return_probability=pair['model_i_precedes_j'],pair_n=pair['model_joint_n'],core_B_minus_A_t10_median=obs[mode]['B_minus_A_t10_ms_median'],full_run_L_search=c['L_search']))
    with (OUT/'observations.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    rec=records['global_axis_-15'];path=Path(rec['source']);r,a,ids=an.an.load_unit(path,1500.);c=rt.read(P/'candidates/global_axis_-15.json');c['topology']=2511
    physics=rt.read(path.parents[1]/'applied_physics.json');rp=P/'rotation'/hashlib.sha256(str(path).encode()).hexdigest()[:20]/'result.json';rotation=rt.read(rp)
    install();an.plt.rcParams.update({'font.family':'Noto Sans CJK JP','pdf.fonttype':3})
    movie=rotation_clip(c,847101,r,a,physics,F,rotation)
    tracks=[t for t in rotation['tracks'] if t['half_turn_candidate']]
    rt.write(OUT/'manifest.json',dict(created_unix=time.time(),sources=sources,rotation_source=str(rp),rotation_sha256=rt.sha(rp),movie=movie,rotation_screen=dict(tracks=len(tracks),candidate_time_fraction=rotation['candidate_time_fraction'],candidate_turns=[t['best_fixed_turns'] for t in tracks],candidate_duration_ms=[t['duration_ms'] for t in tracks],match_fraction_3_30Hz=[t['match_fraction_3_30Hz'] for t in tracks]),scope='One development graph identity and one noise, two matched20s physical durations; no loss or selection changes',producer=str(Path(__file__)),producer_sha256=rt.sha(Path(__file__))))
    note='''# 首个非零全局轴探针：分数下降，路径没有一起改善

只比较噪声847101、基础拓扑身份2511下的0与−15度，两条均为20秒并排除前1.5秒；0度是旧长轨迹的实际前缀重演，不能再计独立重复。几何旋转实际重采样EE边、距离时延并略改总输入，不是同一张物理图上的纯权重旋转。只有一个网络身份、一个噪声的开发对照，不能泛化。

|观测|0度|−15度|
|---|---:|---:|
|合格事件／TA／TB|53／22／31|47／23／24|
|TA上部SCL参与|54.5%|37.0%|
|TA顺序概率误差|0.137|0.177|
|TA杆间时差中位数|−12.39ms|−2.71ms|
|TB顺序概率误差|0.321|0.366|
|TB杆间时差中位数|35.31ms|31.94ms|
|TB的ICL11先于ICL9|8/31|0/23|
|整条运行训练分数|3.416|2.867|

患者FIT参考：TA上部SCL约84.8%、杆间时差−9.28ms；TB杆间时差+1.19ms，ICL11先于ICL9约69.4%（4879次共同参与）。顺序概率误差比较全部可比较触点对，不是单一接触对正确率。+15度及第二噪声尚未完成，不补值、不推广为整个方向族结论。

−15度使TB杆间摘要缩短约3.37ms，但固定ICL回返更差；TA参与与成对顺序也变差。分数下降不能当作传播恢复。TA标签比例由41.5%变为48.9%（患者66.6%），全检测61次中14次被既有重叠规则排除；因此合格事件数下降不等于神经元不活动。保持原资格和全部原始输出。

旋转筛查从0条变为4条候选，占18秒可分析时间的1.33%。这些候选仅约0.536–0.553圈、58–64ms；改用已有3–30Hz相位诊断时，3条匹配率为0，另1条约0.156。它们不构成稳定螺旋或新的独立旋转事件证明。按既定规则选择半圈候选中固定环转角最大的轨迹显示原生2ms动画，仍须目视区别波前弯曲、前沿相遇与持续旋转；不能将筛查值加入loss。

正常选例和各模式前三事件见[患者对照与多事件GIF](../figures/global_axis_-15/README.md)，未按患者相似度挑选。短轨迹事件支持有限，后续仅完成已批准角度和噪声，不因分数下降扩大搜索。
'''
    (OUT/'scientific_note.md').write_text(note)
    (F/'README.md').write_text('### global_axis_-15_847101_rotation_candidate.gif\n首个−15度运行中，按既定旋转筛查固定环转角最大规则选例；显示原生2ms场，白线为核、青环为固定诊断环。该例仅约半圈且相位带宽敏感。\n**关注点**：前沿弯曲或相遇不能直接称为稳定螺旋；这是诊断选例，不用于估计发生率。\n')
    print(json.dumps(dict(output=str(OUT),movie=movie['file'],candidate_tracks=len(tracks))))

if __name__=='__main__':main()
