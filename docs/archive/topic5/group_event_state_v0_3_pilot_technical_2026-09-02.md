# Group-Event State v0.3 pilot：技术报告

- source commit: `bb4a51c93d91e784206b80f38bbc9638b744319c`
- completed runs: 9/9
- partition: 16/4/50/10/20 nested development physical-time split；formal sealed partition 未打开
- grammar: calibration-prefix single K categorical, reported as continue + K|continue + product-form conditional K-subset likelihood
- timing: trapezoidal marked point-process likelihood over valid exposure, including terminal censoring
- state: 16 dimensions; fixed taus = 300/1800/7200/21600 s; bounded tau-dependent event correction
- TBPTT: max 1024 events AND 1800 s; carry+detach, no chunk reset; 300 s segment burn-in
- slow objective: fixed-anchor future-count Poisson NLL at 300/1800/7200 s
- source boundary: legacy learned decoder weights excluded, but legacy full-record contact selection remains upstream-transductive
- open-loop: no future event update; horizons 300/1800/7200 s

## Per-subject seed-median contrasts

### epilepsiae_1146

- 300s: count−multiscale=1.6878; count−shift=-1.8613; continue−shift=0.0001; positive-size−shift=-0.0007; subset−shift=-0.0037; anchors=133, matched=133, pair-admissible seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 3, 'correct_vs_state_free': 0}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.
- 1800s: count−multiscale=1.3366; count−shift=-3.4482; continue−shift=-0.0001; positive-size−shift=-0.0038; subset−shift=0.0273; anchors=113, matched=103, pair-admissible seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 1, 'correct_vs_state_free': 0}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.
- 7200s: count−multiscale=NA; count−shift=1.6435; continue−shift=0.0004; positive-size−shift=-0.0002; subset−shift=0.0010; anchors=70, matched=70, pair-admissible seeds={'correct_vs_multiscale': 0, 'correct_vs_shifted': 2, 'correct_vs_state_free': 0}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.

### yuquan_pengzihang

- 300s: count−multiscale=NA; count−shift=NA; continue−shift=-0.0000; positive-size−shift=-0.0000; subset−shift=-0.0000; anchors=45, matched=45, pair-admissible seeds={'correct_vs_multiscale': 0, 'correct_vs_shifted': 0, 'correct_vs_state_free': 0}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.
- 1800s: count−multiscale=NA; count−shift=NA; continue−shift=-0.0000; positive-size−shift=-0.0000; subset−shift=-0.0000; anchors=40, matched=40, pair-admissible seeds={'correct_vs_multiscale': 0, 'correct_vs_shifted': 0, 'correct_vs_state_free': 0}; coverage-insufficient seeds=0; ridge-edge seeds=3/3.
- 7200s: count−multiscale=NA; count−shift=NA; continue−shift=NA; positive-size−shift=NA; subset−shift=NA; anchors=0, matched=0, pair-admissible seeds={'correct_vs_multiscale': 0, 'correct_vs_shifted': 0, 'correct_vs_state_free': 0}; coverage-insufficient seeds=3; ridge-edge seeds=0/3.

### yuquan_zhangkexuan

- 300s: count−multiscale=2.9126; count−shift=0.0055; continue−shift=0.0000; positive-size−shift=-0.0000; subset−shift=-0.0000; anchors=49, matched=49, pair-admissible seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 3, 'correct_vs_state_free': 3}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.
- 1800s: count−multiscale=6.1813; count−shift=0.0298; continue−shift=0.0000; positive-size−shift=-0.0000; subset−shift=-0.0000; anchors=44, matched=44, pair-admissible seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 3, 'correct_vs_state_free': 3}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.
- 7200s: count−multiscale=33.9859; count−shift=NA; continue−shift=NA; positive-size−shift=NA; subset−shift=NA; anchors=26, matched=0, pair-admissible seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 0, 'correct_vs_state_free': 3}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.

## Interpretation boundary

Negative contrasts are favourable. The pilot is not powered for cohort inference. Two subjects selected the first trained epoch in all seeds, so their near-zero mark contrasts are not affirmative evidence. One subject selected checkpoints at the training-budget edge, so a negative result would remain optimization-limited. A ridge-edge baseline is retained with a caveat rather than converted into a biological result. Count contrasts enter aggregation only when both arms pass the fitted-intercept audit. Mark comparisons against wrong-time and state-free grammar are valid as development diagnostics; a full capacity-matched multiscale mark adapter remains a later comparison and is not silently claimed here.

Machine report: `/data/hfosp_group_event_state_v0_3/pilot/summary_main.json`
