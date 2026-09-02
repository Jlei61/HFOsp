# Group-Event State v0.3.1：审阅后技术收口

- source commit: `bb4a51c93d91e784206b80f38bbc9638b744319c`
- completed runs: 9/9
- closeout status: `V0_3_1_PILOT_CLOSED_MAJOR_REVISION`
- scientific status: `instrument_complete_state_learning_unresolved`
- partition: 16/4/50/10/20 nested development physical-time split；formal sealed partition 未打开
- grammar: calibration-prefix single K categorical, reported as continue + K|continue + product-form conditional K-subset likelihood
- timing: trapezoidal marked point-process likelihood over valid exposure, including terminal censoring
- state: 16 nominal dimensions at 300/1800/7200/21600 s, but learned state-to-state mixing means these are not identifiable physiological time constants
- TBPTT: closes when either 1024 events or 1800 s is reached; carry+detach, no chunk reset; 300 s segment-level burn-in
- slow objective: fixed-anchor future-count Poisson NLL at 300/1800/7200 s
- source boundary: legacy learned decoder weights excluded, but legacy full-record contact selection remains upstream-transductive
- open-loop: no future event update; horizons 300/1800/7200 s
- missing primary estimand: H+S_correct vs H, H+S_correct vs H+S_shifted, dynamic S vs TRAIN-mean S
- count overdispersion audit: variance/mean across existing patient×phase×horizon cells = 7.1–384.8

## Existing contrasts retained as diagnostics, not H1/H2a tests

All finite development-test scores are retained below. The fitted-intercept audit is now a flag and never changes the denominator. `flagged` reports how many scored seeds would have been removed by the deprecated post-hoc rule.

### epilepsiae_1146

- 300s: count−multiscale=1.6878; count−shift=-1.8613; continue−shift=0.0001; positive-size−shift=-0.0007; subset−shift=-0.0037; anchors=133, matched=133, pair-scored seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 3, 'correct_vs_state_free': 3}; posthoc-flagged seeds={'correct_vs_multiscale': 0, 'correct_vs_shifted': 0, 'correct_vs_state_free': 3}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.
- 1800s: count−multiscale=1.3366; count−shift=-4.3725; continue−shift=-0.0001; positive-size−shift=-0.0038; subset−shift=0.0273; anchors=113, matched=103, pair-scored seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 3, 'correct_vs_state_free': 3}; posthoc-flagged seeds={'correct_vs_multiscale': 0, 'correct_vs_shifted': 2, 'correct_vs_state_free': 3}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.
- 7200s: count−multiscale=-92.8397; count−shift=1.2276; continue−shift=0.0004; positive-size−shift=-0.0002; subset−shift=0.0010; anchors=70, matched=70, pair-scored seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 3, 'correct_vs_state_free': 3}; posthoc-flagged seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 1, 'correct_vs_state_free': 3}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.

### yuquan_pengzihang

- 300s: count−multiscale=-60.2723; count−shift=-0.0051; continue−shift=-0.0000; positive-size−shift=-0.0000; subset−shift=-0.0000; anchors=45, matched=45, pair-scored seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 3, 'correct_vs_state_free': 3}; posthoc-flagged seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 3, 'correct_vs_state_free': 3}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.
- 1800s: count−multiscale=10.2968; count−shift=0.0102; continue−shift=-0.0000; positive-size−shift=-0.0000; subset−shift=-0.0000; anchors=40, matched=40, pair-scored seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 3, 'correct_vs_state_free': 3}; posthoc-flagged seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 3, 'correct_vs_state_free': 3}; coverage-insufficient seeds=0; ridge-edge seeds=3/3.
- 7200s: count−multiscale=NA; count−shift=NA; continue−shift=NA; positive-size−shift=NA; subset−shift=NA; anchors=0, matched=0, pair-scored seeds={'correct_vs_multiscale': 0, 'correct_vs_shifted': 0, 'correct_vs_state_free': 0}; posthoc-flagged seeds={'correct_vs_multiscale': 0, 'correct_vs_shifted': 0, 'correct_vs_state_free': 0}; coverage-insufficient seeds=3; ridge-edge seeds=0/3.

### yuquan_zhangkexuan

- 300s: count−multiscale=2.9126; count−shift=0.0055; continue−shift=0.0000; positive-size−shift=-0.0000; subset−shift=-0.0000; anchors=49, matched=49, pair-scored seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 3, 'correct_vs_state_free': 3}; posthoc-flagged seeds={'correct_vs_multiscale': 0, 'correct_vs_shifted': 0, 'correct_vs_state_free': 0}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.
- 1800s: count−multiscale=6.1813; count−shift=0.0298; continue−shift=0.0000; positive-size−shift=-0.0000; subset−shift=-0.0000; anchors=44, matched=44, pair-scored seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 3, 'correct_vs_state_free': 3}; posthoc-flagged seeds={'correct_vs_multiscale': 0, 'correct_vs_shifted': 0, 'correct_vs_state_free': 0}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.
- 7200s: count−multiscale=33.9859; count−shift=NA; continue−shift=NA; positive-size−shift=NA; subset−shift=NA; anchors=26, matched=0, pair-scored seeds={'correct_vs_multiscale': 3, 'correct_vs_shifted': 0, 'correct_vs_state_free': 3}; posthoc-flagged seeds={'correct_vs_multiscale': 0, 'correct_vs_shifted': 0, 'correct_vs_state_free': 0}; coverage-insufficient seeds=0; ridge-edge seeds=0/3.

## Implementation audit

- Adapter is not in an exact zero-gradient dead zone: final projections are N(0,1e-3), gates initialise at sigmoid(-4)=0.017986, and recorded gradients are non-zero. Effective logit modulation/Jacobian was not measured, so functional trainability remains unresolved.
- Per-time state LayerNorm and free state-to-state mixing are present; both prevent interpreting latent amplitude or nominal tau labels physiologically.
- Production TBPTT uses the first event/time limit reached, but 7200-s targets receive at most 1800 s of gradient credit assignment.
- Each validation/development-test pass replays the current checkpoint from the segment boundary; no stale checkpoint trajectory path was found.
- Poisson count is materially misspecified given the dispersion audit and must be replaced by a paired negative-binomial family before a new state experiment.
- The 80–100% block is development-consumed and cannot be reused as a future independent final test.

## Interpretation boundary

Negative raw contrasts are favourable, but none of the above is the required residual H+S estimand. `epilepsiae_1146` provides a limited correct-time diagnostic at 5/30 min; the other two subjects are dominated by first-epoch selection, coverage limits or long-horizon miscalibration. Near-zero mark contrasts are not biological nulls. The pilot supports engineering integration only; state learning, H1 and H2a remain unresolved.

Machine report: `/data/hfosp_group_event_state_v0_3/pilot/summary_v0_3_1_closeout.json`
