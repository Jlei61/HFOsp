# Topic 4 dual-core OOD execution plan

1. Freeze the patient classifier, shaft embedding, detector, engine and
   historical hand-core anchor by hash.
2. Add pure two-core candidate generation and OOD scoring tests, including an
   explicit test that unreadable returned events count as OOD.
3. Extend the existing frozen worker only for a phase manifest, candidate-level
   duration and optional spatial activity grids; prove old candidates are
   unchanged when those fields are absent.
4. Freeze and run the Sobol screen through `systemd-run --user` and `nohup`.
   Workers use one numerical thread. The controller checks every 600 s, keeps
   at least 32 GiB available and resumes from atomic worker artifacts.
5. Aggregate the screen, mechanically freeze six candidates, run selection,
   then freeze one candidate and run 12-seed confirmation.
6. Produce the full shaft-aware confirmation report and Fig.2C-style model GIF;
   inspect the first, middle and final frames.
7. Freeze four connectivity arms using the prior EE/E-to-I coefficient rows and
   run 12 paired seeds under the same detector and Node field.
8. Aggregate paired pathway effects, generate the OOD/mode/KMeans figure set,
   write the Chinese figure README and scientific audit, and update the Topic 4
   closeout only after all artifacts are complete.

Stop launching new work on non-finite values, late runaway, provenance drift,
or less than 32 GiB available memory. A candidate with zero events is a valid
scientific result with OOD 100%, not a missing worker.
