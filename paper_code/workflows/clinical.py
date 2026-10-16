"""Clinical inputs and result saving for the final Fig5 notebook cells."""
import json
import numpy as np
from paper_paths import INPUT_ROOT
from workflows.common import write_json,publish_session
from workflows.tutorial import CAPTIONS,save_model_analysis as save_model
from analysis.clinical_fields import clinical_candidates

CAPTIONS.update({
 'fig5_E_model_patient_fields':'由新模型电流和原始发作电压分别重算的robust-z场。患者示例采用原描述性选择规则；图内min-max归一化仅用于显示空间形状。',
 'fig5_F_field_statistic':'新模型与本患者合格发作窗口的dB场匹配，完整触点置换在平滑前执行，每次重新选择最佳发作。E1146为一个患者统计单位，不代表论文13人队列。',
})


def prepare_clinical(contact_names):
    return clinical_candidates(INPUT_ROOT/'E1146_demo',contact_names)


def save_model_analysis(output,*args,clinical_rows,model_energy,comparison,statistic,patient_statistic,**kwargs):
    save_model(output,*args,**kwargs)
    write_json(output.path/'clinical_phase_readouts.json',clinical_rows)
    write_json(output.path/'model_phase_energy.json',model_energy)
    write_json(output.path/'clinical_descriptive_example.json',comparison)
    write_json(output.path/'clinical_field_statistic.json',statistic)
    if patient_statistic['status']=='OK':
        np.savez_compressed(output.path/'clinical_field_null.npz',null=patient_statistic['null'])
    report=json.loads((output.path/'run_report.json').read_text())
    report.update(clinical_fig5E=comparison['status'],clinical_fig5F=statistic['status'],
                  clinical_candidates=len(clinical_rows),clinical_eligible=sum(r['eligible'] for r in clinical_rows),
                  clinical_patients=1,full_paper_cohort='NOT_RUN_SINGLE_PATIENT_DEMO')
    publish_session(output.name,output.path,report)
