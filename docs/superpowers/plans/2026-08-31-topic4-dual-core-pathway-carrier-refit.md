# Topic 4 rev17 execution plan

1. Freeze the rev16 Node, classifier, detector, connection mapper and all existing artifacts by hash.
2. Complete the zero-simulation core-first-onset audit and the archived-carrier readout parity check.
3. Add configurable `g_EE` and `g_EtoI` expression doses without changing the zero-dose and unit-dose legacy paths; test exact parity and budget conservation.
4. Freeze the 20-cell pathway response surface and run three 8 s common-random-number seeds under nohup/systemd worker control.
5. Aggregate OOD, mode share, KMeans, event yield and absolute timing. Freeze four candidates mechanically from the registered score and Pareto table.
6. Run the four candidates on three new 12 s seeds and freeze one interictal pathway work point.
7. Run the minimal core-lesion audit if the zero-simulation core-first association is present.
8. Validate the unfiltered carrier readout. If the raw Node canary lacks local three-cycle bursts, run the six-cell AMPA/GABA decay capacity canary on three common seeds with the external Poisson mean frozen to baseline; report OOD/KMeans/event yield alongside carrier metrics.
9. Freeze one carrier-compatible work point and confirm it on 12 fresh 20 s seeds; do not open Z/M during this phase.
10. Regenerate the Fig.2C-style GIF and KMeans/OOD panels with explicit model-time playback and unfiltered/filtered readout labels.

Long runs use one numerical thread per worker, measured-RSS concurrency, at least 32 GiB free memory, atomic worker ownership and a 600 s monitor. Simulation errors, non-finite values and runaway are invalid cells; low event yield remains a scored scientific result rather than a missing worker.
