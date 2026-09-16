"""Record the actual completed review and close this bounded, nonpositive round."""
from pathlib import Path
import sys, json, time, datetime, subprocess, re
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from PIL import Image
from scripts import analyze_topic4_three_observable_bo as a
O=a.OUT;A=a.A;D=A/'confirmation_review';F=D/'figures';N=O/'overnight_20260914'

def main():
    rows=a.records();g3=[r for r in rows if r['stage']=='confirmation'];train=[r for r in rows if r['stage'] in ['initial','adaptive']]
    assert len(rows)==72 and len(train)==64 and len(g3)==8
    assert all(r['J'] is not None and r['physical_status']=='COMPLETE_NO_RUNAWAY' for r in rows)
    assert sum(r['N'] for r in train)==9586 and sum(r['N'] for r in g3)==1261
    plan=a.rt.read(O/'plan.json');assert len(plan['reused_units'])==8
    assert not (O/'proposals/response.json').exists() and not list(O.glob('response*/units/*/*/workers/trajectory.json'))
    frozen=a.rt.read(A/'objective_frozen.json')
    assert a.rt.sha(A/'training_objective.pkl')==frozen['objective_sha256']
    pointer=a.rt.read(Path('/home/honglab/leijiaxin/HFOsp/config/topic4_local_geometry_ee_axis_reference_v1.json'))
    assert pointer==plan['reference_pointer']
    for kind in ['contract','runtime','candidate']:assert a.rt.sha(pointer[f'{kind}_path'])==pointer[f'{kind}_sha256']
    proposals=[]
    for b in range(1,5):
        p=O/f'proposals/adaptive_{b:02d}.json';q=a.rt.read(p);fb=a.rt.read(O/f'proposals/feedback_{b:02d}.json')
        assert len(q['ids'])==4 and len(fb['rows'])==4 and len(q['training_data'])==16+(b-1)*4
        proposals.append(dict(batch=b,training_conditions=len(q['training_data']),proposal_sha256=a.rt.sha(p),feedback_sha256=a.rt.sha(O/f'proposals/feedback_{b:02d}.json')))
    # These are the actual final PNGs viewed by the agent, including all 48
    # fixed-selected native-event sheets and all eight spectral comparisons.
    inspected=list(F.glob('*.png'))+list((D/'native_frames').glob('*.png'))+[A/'patient_count_block_reference/figures/patient_actual_N_confirmation_reference.png',A/'figures/raw_paired_response_TB.png']
    gifs=[]
    for cid in ['bridge_circle_out125_xminus075','g2_b01_p01']:
        for t in [3711,3712]:
            for n in [849401,849402]:
                folder=A/'native_review'/cid/f'{t}_{n}';inspected.append(folder/f'{cid}_patient_spectra_model_envelopes.png')
                gif=folder/'patient_mean_native_multievent.gif'
                with Image.open(gif) as im:
                    for i in range(im.n_frames):im.seek(i);im.load()
                    count=im.n_frames
                assert count==378
                gifs.append(dict(path=str(gif),sha256=a.rt.sha(gif),frames=count))
    for p in inspected:
        with Image.open(p) as im:im.verify()
    reason='新候选总体分布与TB条件距离改善，但患者频谱/模型包络和固定多事件原生场共同显示：两端扩散存在，TB上部ICL折返及杆间时序失配持续，TA局部参与/先后出现新退步。拓扑3711两噪声的代表TA丢失SCL8，全事件TA杆内顺序概率误差四配对均变差。不把已有TB残差说成新增核外随机源，也不把短暂旋转筛查当稳定螺旋。因此不接受双模式共同传播改善的科学门；数值触发通过与窗口关闭分别记录。'
    review=dict(review_type='ACTUAL_AGENT_VISUAL_AND_SCIENTIFIC_REVIEW',time=datetime.datetime.now().astimezone().isoformat(),inspected_files=[dict(path=str(p),sha256=a.rt.sha(p)) for p in inspected],scientific_reasoning=reason,
        scope=dict(native_events=48,native_sampled_frames=336,spectral_comparisons=8,full_GIF_frames_decoded=3024,full_GIF_continuous_visual_playback=False,all_60_second_frames_visually_reviewed=False),
        candidates={'g2_b01_p01':dict(no_new_contradictory_propagation=False,judgment='PARTIAL_DISTRIBUTION_IMPROVEMENT_WITH_TA_LOCAL_RECRUITMENT_AND_ORDER_TRADEOFF',full_dual_mode_recovery=False,explanation=reason)},human_review='PENDING_USER_VISUAL_REVIEW',source_report=str(D/'scientific_review.md'))
    assert a.rt.read(D/'execution_audit.json')['status']=='ALL_EIGHT_COMPLETE_EXECUTION_IDENTITIES_MATCH'
    pdf={}
    for p in [N/'final_optimization_report.pdf',D/'native_review_appendix.pdf']:
        info=subprocess.check_output(['pdfinfo',str(p)],text=True);pages=int(next(s.split(':')[1] for s in info.splitlines() if s.startswith('Pages:')))
        pdf[str(p)]=dict(pages=pages,sha256=a.rt.sha(p),bytes=p.stat().st_size)
    assert sorted(v['pages'] for v in pdf.values())==[16,30]
    missing=[]
    for target in re.findall(r'\]\(([^)]+)\)',(D/'scientific_review.md').read_text()):
        if '://' not in target and not (D/target).resolve().exists():missing.append(target)
    assert not missing,missing
    assert not list(A.glob('*failure.json'))+list(O.glob('*failure.json'))
    decision=a.rt.read(O/'g4_numerical_decision.json');assert decision['candidates'][0]['numerical_trigger'] is True
    proof=dict(status='COMPLETE_BOUND_SEARCH_AND_REVIEW',time=datetime.datetime.now().astimezone().isoformat(),slot_counts=dict(total=72,training=64,confirmation=8,new_physical=64,reused=8,G4=0),events=dict(training=9586,confirmation=1261,total=10847),engineering_failures=0,runaway=0,proposals=proposals,objective_and_reference_unchanged=True,reference_pointer_match=True,reference_contract_sha256=pointer['contract_sha256'],execution_audit=str(D/'execution_audit.json'),visual_review=str(A/'g3_agent_native_review.json'),decoded_GIFs=gifs,PNG_decodes=len(inspected),reports=pdf,link_errors=missing,full_bitwise_replay='NOT_PERFORMED',user_visual_acceptance='PENDING')
    a.rt.write(A/'g3_agent_native_review.json',review)
    for p in [D/'confirmation_summary.json',D/'native_frames/extraction.json']:
        x=a.rt.read(p);x['actual_agent_visual_review']=str(A/'g3_agent_native_review.json');x['human_review']='PENDING_USER_VISUAL_REVIEW';x['scientific_acceptance']='PARTIAL_DISTRIBUTION_ONLY';a.rt.write(p,x)
    a.rt.write(D/'completion_audit.json',proof)
    completion=dict(status='ROUND_COMPLETE_PARTIAL_RECOVERY_G4_NOT_STARTED',time=time.time(),counts=proof['slot_counts'],events=proof['events'],numerical_decision=decision,scientific_review=str(A/'g3_agent_native_review.json'),G4_dispatched=0,G4_not_started_reasons=['AUTONOMOUS_DISPATCH_WINDOW_CLOSED','TA_LOCAL_RECRUITMENT_AND_ORDER_TRADEOFF_NOT_ACCEPTED_AS_DUAL_MODE_PROPAGATION_IMPROVEMENT'],window_end=a.rt.read(N/'window.json')['latest_review_beijing'],complete_report=str(N/'final_optimization_report.pdf'),scientific_report=str(D/'scientific_review.md'),full_patient_dual_mode_recovery=False,model_frozen=False,Fig5_started=False,human_review='PENDING_USER_VISUAL_REVIEW')
    a.rt.write(O/'optimization_complete.json',completion)
    a.rt.write(O/'optimizer_status.json',dict(status=completion['status'],updated_unix=time.time(),completion=str(O/'optimization_complete.json')))
    a.rt.write(O/'status.json',dict(status=completion['status'],time=time.time(),active=[],queued=0,complete=72,completion=str(O/'optimization_complete.json')))
    print(json.dumps(dict(status=completion['status'],inspected_PNGs=len(inspected),decoded_GIFs=len(gifs),decoded_frames=sum(x['frames'] for x in gifs),pages=[v['pages'] for v in pdf.values()]),ensure_ascii=False))

if __name__=='__main__':main()
