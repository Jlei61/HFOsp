"""Notebook input/output. Scientific calculations stay in the notebook steps."""
import json
import os
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from paper_paths import CODE_ROOT, INPUT_ROOT
from workflows.common import new_session, previous_session, publish_session, write_json
from plotting.computational import save


def raw_inputs(default='raw_example.json'):
    config = json.loads((CODE_ROOT / 'config' / os.environ.get(
        'HFO_RAW_CONFIG', default)).read_text())
    return config, INPUT_ROOT / config.get('input_directory', 'E1146_demo')


def model_inputs():
    config = json.loads((CODE_ROOT / 'config' / os.environ.get(
        'HFO_SNN_CONFIG', 'snn_tutorial.json')).read_text())
    root = INPUT_ROOT / 'E1146_demo'
    with np.load(root / 'E1146_training_observations.npz', allow_pickle=False) as z:
        patient = SimpleNamespace(fit=z['fit_centroid_ms'], cal=z['cal_centroid_ms'],
            cal_draw_indices=z['cal_draw_indices'], names=z['contact_names'].tolist(),
            contacts=z['contact_xy_mm'])
    patient.observer = json.loads((root / 'E1146_observer.json').read_text())
    with np.load(root / 'E1146_template_observations.npz', allow_pickle=False) as z:
        patient.template_observations = {k:z[k] for k in z.files}
    patient.contact_contract = json.loads((root / 'E1146_contact_contract.json').read_text())
    coordinates=json.loads((root/'E1146_coordinates.json').read_text())
    patient.coords_mm=np.asarray([coordinates[name] for name in patient.names],float)
    if patient.observer['contact_names'] != patient.names:
        raise ValueError('Observer and patient contact order differ')
    config['contact_names']=patient.names
    return config, patient


def load_events(session='fig1'):
    source = previous_session(session)
    config = json.loads((source / 'run_report.json').read_text())['configuration']
    with np.load(source / 'events.npz', allow_pickle=False) as z:
        events = {key: z[key] for key in z.files}
    return events, events.pop('coords_mm'), config


# Export captions live here so saving files does not interrupt the scientific story.
CAPTIONS = {
    'fig1_A_waveforms': '新检测群事件的三个160ms电压片段。显示通道间波形与时序。',
    'fig1_B_centroids': '三个新检测群事件的时频谱。竖线为完整事件窗的S³时间质心。',
    'fig1_C_ranks': '全部示例片段的群事件rank矩阵。空白表示该通道未参与。',
    'fig1_E_kmeans': '按本次masked-rank K-means标签排列事件。K=2为模板设定。',
    'fig1_D_statistics': '事件顺序一致性与置换基线的比较。统计对象为一个患者的片段。',
    'fig1_F_template_statistics': '两个模板内部的时序一致性。与全部事件的结果并列比较。',
    'fig2_A_geometry': '按名称对齐的三维触点坐标。空间分析使用这些实测位置。',
    'fig2_B_rank_gradient': '从事件rank与坐标拟合early-to-late方向。三维方向显示于固定MRI x/y平面。',
    'fig2_B_crossfit': '在相同held-out事件上比较两种模板训练方法。置换形成空间信息增益的基线。',
    'fig2_C_dynamic_fields': '两个新检测事件的波形、时频谱及四帧动态包络场。沿用论文6mm空间平滑、逐帧前三触点归一化及gamma=0.5配色。',
    'fig2_D_template_fields': '两模板在同一TA平面上的归一化rank场。viridis表示从早到晚，弱支持区留白。',
    'fig2_field_comparison': '共同平面内的TA/TB场相关与触点置换基线。单患者演示场比较的计算。',
    'fig4_A_model': 'E/I局部回路、core与触点几何、Gaussian采样权重及新事件包络。core降低局部E细胞阈值。',
    'fig4_F_fitted_waveforms': '新spike读出的30–80Hz波形及检测事件窗。这是活动代理，单位不同于SEEG电压。',
    'fig4_B_observable_errors': '新候选的rank、杆内顺序及参与率误差。这些可观察误差与优化目标J分开。',
    'fig4_C_core_search': 'K=1至4在相同预算下的优化曲线。曲线展示本次训练误差。',
    'fig4_D_direction_response': '改变E→E连接核方向后重新仿真的误差响应。其他参数与噪声保持固定。',
    'fig4_E_core_positions': '两core候选的实际位置散点，颜色为重新计算的J_joint，黑边标出最低20%。小预算不估计位置密度。',
    'fig4_G_mean_rank': '患者观测与新模型事件的平均传播rank。未参与触点保留缺失。',
    'fig4_H_crossfit': '一半触点分配类别，另一半触点评分。两半交换后汇总相关。',
    'fig4_I_template_similarity': 'E1146患者与重新积分的论文xy_left_20工作点的TA–MTA及TB–MTB模板相似度。单患者点不代表论文25人队列。',
    'fig5_A_waveforms': '新Z/M轨迹的10kHz电流读出，经80–1000Hz滤波及募集前固定P99.5缩放。绘图使用20261011六面板版的原函数。',
    'fig5_B_slow_variables': '同次积分的全体E与Core A的平均Z及有效适应电流ηM。慢变量与活动来自同一轨迹。',
    'fig5_C_spatial_activity': '预定时刻的20ms平均E活动空间图。显示活动在空间中的扩展。',
    'fig5_D_state_trajectory': 'Z、ηM与E发放率的三维轨迹。单条轨迹用于描述本次活动变化。',
    'fig5_E_model_onset_field': '新电流进入后1–150Hz能量的robust-z场。沿用论文逐场min-max/Blues显示，空间平面固定于患者间期观测。',
    'supplementary_onset_signed_z': '同一新仿真进入场，色条恢复实际baseline robust-z。原始正负值完整保留，不能由主图相对高低判断增强方向。',
    'supplementary_gamma_field_scores': '新电流的gamma能量及固定场评分随时间变化。场相关与投影分别显示。',
}


class TutorialOutput:
    def __init__(self, name):
        self.name = name
        self.path = new_session(name)
        self.figures = self.path / 'figures'

    def show(self, figure, name):
        save(figure, self.figures, name, CAPTIONS[name])
        plt.show()

    def simulation(self, name, run):
        from modeling.computational import save_run
        save_run(self.path / (name + '.npz'), run)
        write_json(self.path / (name + '.json'), {
            k: v for k, v in run.items() if not isinstance(v, np.ndarray) and k != 'slow'})

    def finish(self, **report):
        publish_session(self.name, self.path, report)
        print('本次结果：', self.path)


def save_events(output, events, statistics, config, raw_root):
    from analysis.from_events import align_coordinates
    coordinates = json.loads((raw_root / config['coordinates']).read_text())
    np.savez_compressed(output.path / 'events.npz', **events,
                        coords_mm=align_coordinates(events, coordinates))
    write_json(output.path / 'statistics.json', statistics)
    output.finish(scope=config['scope'], configuration=config, n_patients=1,
                  n_events=events['ranks'].shape[1])


def save_fields(output, gain, clustered, record, field_a, field_b, similarity):
    for name, data in [('crossfit', gain), ('field_record', record),
                       ('full_timing_plus_space_fit', clustered['fit']),
                       ('shared_field_similarity', similarity)]:
        write_json(output.path / (name + '.json'), data)
    np.savez_compressed(output.path / 'spatial_fields.npz',
        X_a=field_a['grid']['X'], Y_a=field_a['grid']['Y'],
        F_a=field_a['field'][0], S_a=field_a['field'][1],
        X_b=field_b['grid']['X'], Y_b=field_b['grid']['Y'],
        F_b=field_b['field'][0], S_b=field_b['field'][1])
    output.finish(input_events=str(previous_session('fig2_raw') / 'events.npz'),
        template_method=clustered['fit']['method'], template_policy=clustered['fit']['event_policy'],
        cosine=float(field_a['direction'] @ field_b['direction']),
        spatial_gain=gain['spatial_information_gain'], field_status=record['status'])


def save_model_analysis(output, config, objective, matching, selection, onset, entry,
                        onset_field, energy, *, angle_rows):
    write_json(output.path / 'model_configuration.json', config)
    write_json(output.path / 'objective_configuration.json', dict(
        version=objective.version, scales=objective.scales, n_fourier=config['n_fourier']))
    write_json(output.path / 'model_patient_matching.json', {
        k: v for k, v in matching.items() if k != 'within_shaft_null'})
    np.savez_compressed(output.path / 'model_patient_matching_null.npz',
                        null=matching['within_shaft_null'])
    write_json(output.path / 'video_selection.json', selection)
    write_json(output.path / 'angle_response.json', angle_rows)
    write_json(output.path / 'new_onset_broadband_field.json', onset_field)
    write_json(output.path / 'fixed_energy_geometry.json', dict(
        plane=energy['plane'], contact_mask=energy['contact_mask']))
    np.savez_compressed(output.path / 'new_energy_fields.npz', time_sec=energy['time_sec'],
        power=energy['power'], robust_z=energy['robust_z'], **energy['scores'])
    output.finish(scope=config['scope'], configuration=config, video2=selection['status'],
        historical_t120_sec=onset, entry=entry, model_onset_field=onset_field['status'],
        clinical_fig5E='NOT_RUN_REQUIRES_RAW_ICTAL_EXAMPLES',
        clinical_fig5F='NOT_RUN_REQUIRES_RAW_COHORT', fig4I='E1146_TEMPLATE_SIMILARITY_ONLY', fig5_layout='reference_layout_20261011')
