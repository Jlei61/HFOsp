# 当前交接：直接K点已经验证，完整时变响应正在运行

**Goal ACTIVE，完整目标未缩减，NOT_ESTABLISHED正式分岔。不要mark complete/blocked。** 用户要Fig5原生闭环及相关分岔/机制，禁止将局部数值图交差。无子agent/push/清理授权；保留原生物理、density_spatial.py、已接受状态空间/正式Fig5及其他脏工作。新脚本在scripts/topic4_loop_bifurcation；解释器cuda_env。旧完整事实与公式见HANDOFF_before_direct_branch_steps.md、DIRECT_RESPONSE_HANDOFF.md及更早handoff_archive_before_target_repair.md。本轮没有新增原生seed或改方程。

## 立即工作：先核实时状态，不重复启动

- `all_target_frequency_direct`：已启动原单频kernel的全部40000目标、四通道（I无G）、全/半幅度、256replica、4s+1s，固定两频率1Hz和5Hz。两个目标分区GPU0/1；第一频完成自动收集后才启动第二频，没有更多频率或参数的自动扩张。producer`measure_all_target_frequency.py`，不能在worker运行中修改它或`measure_all_target_dc.py`。
- `direct_frequency_operator`：0/1/5Hz全部目标原延迟矩阵已生成，延迟动作/突触滤波及因果R/G/M递推核对PASS。完整DC回路动作与J+I误差1.9e-15。CPU analyzer等待每频的完整响应，随后自动组装L(z)，计算接近1的8个回路特征值及8块矩阵作用采样误差。producer`direct_frequency_operator.py`。
- 实时PID和已完成目标数见下方快照，也要读当前progress。若分析器报错，按原数据修复，不重跑计数。矩阵谱是L(e^iw)的反馈特征值，**不是时间增长率**；两个频率不能认证Hopf或稳定/不稳定，也不能证明不存在其他模式。

## 本轮已完成的实质结果

1. `direct_response_system.py`组装真实目标DC导数、implicitM、physicalG及K参数导数。`direct_parameter_step.py`第一个较大步0.05/0.025/0.0125都因预测越过非负/步幅约束停下，保留在`direct_exit_first_K_step`，没有派发这几个点。原producer版本另存`producer_at_registration.py`，后续只增加命令行可选输出/步长，原失败不抹去。

2. `audit_direct_parameter_tangent.py`：完整、半幅度、两个独立128replica半样本的切线差异约1.2–1.3%、余弦>0.9999；allE切线约−42.2Hz/K，最大target约4449Hz/K。约98.9%绝对响应在核外，90%质量覆盖约6200E。它是参数切线不是失稳模态。另一次静态DC J最大实部特征值约−.03968，保存tangent_audit/static_spectrum，但**不标动态稳定**。

3. 更小步`direct_exit_small_K_step`已完整：K9.00625，predictor+一次freshcorrector验证，两×40000×256每eval，seeds928911–928914；最终E/I源群RMS .00647/.14558Hz，合并SEM RMS .00679/.14249，全部源群在6SEM界内，E targetRMS .03058 vscombinedSEM .02932。全E243.8332，A469.0673/B474.1836，Graw4.46468。仅近自洽候选，不是精确root或稳定支。

4. `continue_direct_parameter_pair.py` / `direct_exit_K_pair`已完整：K9.0125通过三次直接eval（predictor+两corrector），E/I RMS .00723/.14301、combinedSEM .00681/.14260，全E243.5517，A468.9802/B474.1479，G4.43645。请求下一点9.025时原guard缩为9.01875；它三次eval后整体RMS虽在采样量级，**仍有3个低率核外源群超6SEM，未接受并停止本有界批次**。不是fold/无解证据。详见point_01/evaluation_2/unresolved_groups.json：g139/178/401，率.0282/.00766/.568Hz，残差.00875/.00626/.05786Hz；其K9原率.550/.322/5.151Hz。冻结K9导数在近零尾部的变化/投影需要定位；不要把后续工作变成无关微小残差迭代，先评估这些误差对相关模式和终止机制的影响。没有放宽原规则或继续穿过失败点。

5. `review_direct_branch_steps.py`生成`figures/direct_branch_first_steps.png/svg`，仅K9/9.00625/9.0125，实际PNG目视PASS/SVGXMLPASS，humanPENDING，README已写。约98.94%的逐细胞绝对变化位于核外；两核持续高率、G约4.4且仍高于Z恢复阻断值2.6694。**这不是终止或恢复图，也没替换Fig5**。曲线冻结region定义，圆圈沿用1.5mm物理核标记，不混为observer1.75定义。

6. 为动态测量提速做了一次`direct_multisine.py`+`validate_direct_multisine.py`pilot，130条件/8192rep/4s+1s/1,5,20,80Hz，kernel零频4通道逐位及单频计数逐位PASS。400个DC可估计比较396pass，全部520比较514pass，**未达到其预登记全适用项通过规则，未推广**。4个可估计失败为I方差通道的弱差别（35838-vI-5Hz、33797-vI-20Hz、35908-vE-20Hz单频比较、38688-vE-80Hz幅度），原失败保留。不能把未通过等同于物理错误，也不能因为大多数通过就改原gate；当前完整空间采用之前已核验的单频方法，勿继续优化这个捷径占据主线。

## 当前科学方向，防止走偏

新工作已经得到真实直接响应的局部条件分支，当前仅K9附近很小一段。尚未达到终止边界；不能用只影响核外的第一个转折解释两核退出。实际原生K9high约243Hz，而K10.5/12两历史静默是条件证据，不能当正式稳定性。当前持久G稳态约4.4，自然退出携带Graw约11.5；高活动K和G同为.5秒，所以准静态K图可能不够，需要G滞后/吸引域/携带状态及原生动态对应。完整目标要保留这些区分，而不是追求一个好看的分岔名称。

等1/5Hz响应与回路结果完整后，先看相位/近单位模式/块误差及所有幅度失败，再决定必要频段及原生定向扰动。两频不足以数全时间特征根，不能静态矩阵特征值冒充动态增长率。若仍只见局部核外模态，要进一步检查它与真正终止之间的联系，不自动加大全域频率扫描。新的近点导数需要更新时应围绕实际场，不回旧NN全域拟合或已拒绝的22个数值根。

正式稳定性缺口仍包括：局部频率响应的有限窗口/相位误差、固定M均值近似、源群平均、完整频段及原生转换对应。半样本块不包含这些系统近似误差。原生成功闭环及恢复机制不受这些新条件试验改写；G/K先降率，G消退解除Z阻断，R<=5Hz下K的5秒保留给Z时间，Z恢复与短事件返回不同时间。

## 交付和读取入口

- scientific_review_live.md已改为当前简明判断；旧全量阶段记录保留scientific_review_before_direct_branch_steps.md。
- direct_response_operator.md追加K延拓、失败边界、完整离散Hsyn/H_RG/H_M和解释。
- mechanism_model.md/bifurcation_scope.md/completion_audit_live.md更新顶部与过时状态；旧版本另存*_before_direct_branch_steps.md。
- Figure candidate有真实文件且已目视，未人工验收。新图只诊断本地分支，最终新Fig5分岔图仍未完成。
- Memory使用MEMORY.md324–325，最终append citation，rolloutids01a09eae-c163-7cf2-8f2d-f11d43bdeaaf、01a0add1-6bb8-78f1-8af3-98dc7b0724b2；无memory写入授权。

## 写入时实时状态（进程可能随后完成，必须刷新）

```json
{
  "frequency": {
    "status": "MEASURING_FREQUENCY",
    "frequency_Hz": 1.0,
    "pid": 20790,
    "workers": [
      {
        "pid": 21312,
        "status": null
      },
      {
        "pid": 21313,
        "status": null
      }
    ],
    "updated_epoch": 1790539496.9202724
  },
  "analysis": {
    "status": "WAITING_MEASURED_RESPONSE",
    "index": 0,
    "pid": 21019,
    "updated_epoch": 1790539502.595574
  }
}
```
