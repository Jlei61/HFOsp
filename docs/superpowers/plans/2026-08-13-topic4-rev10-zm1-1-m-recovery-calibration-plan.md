# Topic 4 rev10-ZM1.1 execution plan

## 1. Freeze and fit

- Freeze slow-off plus `tau_adp={500,1000,1500,2000} ms` candidates.
- Run 20 s trajectories on seeds 1361-1363.
- Launch through `systemd-run --user -> nohup`.
- Measure one active-arm sentinel RSS before choosing 1-6 workers.
- Audit runaway, returned events, supervised modes, natural KMeans, patient
  geometry, and Z/M traces.
- Advance at most two safe Pareto candidates.

## 2. Independent selection

- Freeze the fit decision hash before launching selection networks.
- Run only the shortlisted candidates plus the same slow-off control on seeds
  1371-1374.
- Apply the predeclared multi-metric ordering without reading confirmation
  seeds.
- Freeze one selected candidate and its decision hash.

## 3. Confirmation

- Run only the frozen candidate plus slow-off on seeds 1381-1386.
- Keep the same 20 s duration, detector, field, OU process, and worker memory
  guard.
- Produce the confirmation decision and both canonical figures.
- Visually inspect PNG/PDF output and verify figure metadata/provenance.

## 4. Report boundary

The final result may be a pass or a bounded negative result. Allowed positive
language is limited to reproducing a stable two-mode interictal repertoire in
this SNN family. Do not claim a unique biological core, seizure reproduction,
or patient generalization.
