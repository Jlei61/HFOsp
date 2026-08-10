# FCXR-LC4f — X-relay depth closure on the accepted no-kick entry

Date: 2026-08-10

Status: **DESIGN LOCK — one empirical X-depth candidate authorised**

## 1. Core question

LC4e closes the cell-load-current family.  The remaining shortest route is the already demonstrated
X relay offset surface.  This sprint asks one question:

> Can the existing X relay, with no new sensor or time constant, reach its measured termination depth
> after the accepted no-kick D/Z entry, autonomously offset within 1--5 s, protect recovery, and return
> to the frozen interictal statistical neighbourhood?

## 2. Locked candidate

Use the LC4c entry coordinate `theta_h_lc2=1.7317735254764568`, but disable the LC4 M actuator
entirely (`use_m=False`; executed M current exactly zero).  Keep all RC1, Z, H, X, geometry,
connection-seed and noise settings unchanged except:

```text
y_gate = 76.63856219587187  # unchanged frozen LC1 Q99.9 gate
K_y    = 3.0                # was 5.0
```

Keep `tau_y=120 ms`, `tau_x_down=500 ms`, `tau_x_up=5000 ms`, `x_min=0.1`, `hill_n=4`.
This is not a grid.  It is selected before execution because archived late-bout probes give
`X_min=0.377616` at K=3, crossing the frozen termination bracket (`<=0.380`), while K=4 gives
`0.383705` and K=5 gives `0.390174`.  The activity threshold is not lowered, so the candidate does
not deliberately expose more interictal samples to X.

## 3. X1 fresh 22 s screen

Run one fresh 40k trajectory, connection seed 1 / noise 401, with no kick, reset, state fork or
parameter step.  Require finite dynamics, zero clip, non-refractory carrier, at least 8 s and three
returning events before onset, a cumulative spontaneous onset, a 1--5 s ictal bout, autonomous
offset, and a fully observed 2 s no-relapse guard with post-offset rate below pre-onset rate.
The LC4 M current must be exactly zero throughout.

Positive label: `X_DEPTH_OFFSET_CANDIDATE`.  Distinguish no entry, overfast offset, late offset,
rapid relapse and no offset; do not collapse them into one NO-GO label.

## 4. X2 conditional 70 s lifecycle

Only X1 positive authorises one 70 s trajectory with the identical candidate.  Reuse the frozen LC4
nominal gate: >=8 s pre-onset interictal, >=3 returning IEDs, 1--5 s bounded carrier, autonomous
offset, postictal suppression, and a final fixed 8 s returning-event window within reference rate,
duration and participation bands.  Only nominal eligibility authorises the existing exact-final-D
12 s confirmation.

## 5. Stop and interpretation

- X1 negative: stop; the archived frozen X authority does not transfer to the natural-entry closed
  loop at this clean depth.  Do not scan K or change X time constants in this sprint.
- X1 positive / X2 recovery negative: accept X termination authority on natural entry, but report
  recovery as unresolved; the next axis is postictal protection, not carrier strength.
- Full pass: single-seed lifecycle candidate only; morphology and confirmation seeds remain later.

## 6. Resources

One 40k worker, one math thread.  Every long stage runs with `setsid nohup`, PID/SID, source lock,
stage flock, RUNNING/DONE/FAILED/STOP sentinel and resource log.  Do not submit below 128 GiB
MemAvailable; +256 MiB swap blocks later stages and +512 MiB terminates only the newest owned task.
X1 wall guard 4 h; X2 nominal 8 h; exact-D 3 h.  Never touch sibling processes or worktrees.

