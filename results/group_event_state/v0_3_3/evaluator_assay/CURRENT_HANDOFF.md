# Agent A (Evaluator / Assay / Data Contract) — CURRENT HANDOFF

- worktree: `.worktrees/topic5-ges-v033-evaluator-assay`, branch `codex/topic5-ges-v033-evaluator-assay`, base `233f3ad1` (assumed clean release base; no `V0_3_3_EXECUTION_RELEASE.json` exists → audit/implement/test/synthetic-smoke mode only)
- plan of record: `docs/superpowers/plans/2026-09-02-topic5-ges-v033-agent-a-evaluator-assay.md`
- status JSON: `/data/hfosp_group_event_state_v0_3_3/agent_a/agent_a.status.json`

## Done
- environment audit (worktrees / processes / GPU / RAM / disk), spec/plan/handoff read, v0.3.2 source read
- E1146 pre-check (read-only): anchor sets and counts identical across paths; prediction_H differs (registry H_strong 125 features @89e55a58 vs H1-eval refit 126 features @81d36b74); "H+S_correct" is a different estimator on each side

## Running
- baseline v0.3.2 test suite in the new worktree

## Pending
- Task 1 canonical evaluator (TDD) → Task 2 E1146 audit JSON → Task 3 boundaries → Task 4–6 DGP/oracle/power smoke → Task 7 eligibility → Task 8 reports

## Resources (snapshot 22:51)
- load 12/80 cores (Topic 4 rev21 ZM coarse controller + ~10 node workers, fig5a capture); GPUs idle (not used by this line); RAM 223 GiB available; /data 3.1 TB free
