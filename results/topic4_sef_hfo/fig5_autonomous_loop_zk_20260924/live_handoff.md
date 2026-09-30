# Figure 5 completed campaign handoff — 2026-09-25

> FINAL HANDOFF: All 18 primary + 16 structural conditional branches and both 120 s native controls are COMPLETE. All 34 native conditional case images and final 24-point map PNG/PDF have been reviewed. No simulation remains. Final scientific review and delivery index are written. completion_audit.json is PASS_BOUNDED_DELIVERY_FORMAL_BIFURCATION_NOT_ESTABLISHED; final snapshot final_20260925_v2 contains 185 verified files. goal_requirements_audit.json finds no remaining authorized bounded work. All campaign workers/schedulers exited; final_process_check.json records the check. Historical running statements below are superseded by this paragraph. Human candidate review remains PENDING; formal bifurcation NOT_ESTABLISHED. No new dispatch is authorized by this handoff.

Historical execution instructions (superseded by FINAL HANDOFF): Complete the authorized bounded18 native Z/K branches +2 autonomous120s graph controls +16 conditional graph branches, then analysis, candidate figure review and reproducible closeout. Do not mark complete while required runs remain. Slow healthy workers are not blocked. No subagents, no Codex memory edits, preserve unrelated jobs/worktrees. Scientific Python: /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python.

## Verified live state

Snapshot 2026-09-25T07:23:01.303567+08:00. PRIMARY18 COMPLETE; STRUCTURAL15/16 COMPLETE, all15 nativePNGpairs reviewed. Nativegraph2/2COMPLETE andALL3full120comparison+isotropictailreviewed. BOTHhighZlowK historycomparison PNG+PDFreviewed. ALLFOUR devicehandoff firstblocksPASS. FullgoalACTIVE.

CURRENT structural supervisor 3290602; unchangedcapacity/guards. Active [{'condition': 'isotropic', 'name': 'z0.25_k0.02_interictal', 'pid': 3341973, 'time_s': 76.0, 'state': 'RUNNING', 'backend': 'cuda_ordered:0'}, {'condition': 'isotropic', 'name': 'z0.75_k8_high', 'pid': 3416565, 'time_s': 30.0, 'state': 'RUNNING', 'backend': 'cuda_ordered:0'}, {'condition': 'isotropic', 'name': 'z0.75_k8_interictal', 'pid': 3440713, 'time_s': 73.0, 'state': 'RUNNING', 'backend': 'locality_cpu'}]; pending []; failures [].

ALL FOUR DEVICE HANDOFF FIRST2sBLOCKS VERIFIED: isotropicZ.25K.02high/isotropicZ.95K.02high120000→140000,rotatedZ.25K.02interictal540000→560000,isotropicZ.25K.02interictal560000→580000. ALL8streampairedfilenames,shapes,timestamps,input,Z/K,countfieldconservationPASS; backuphashesintact. Currentpending0; futuretransfersneedverificationonlyifanynewhandoffoccurs.

Native2/2 COMPLETE; supervisor2603204 andworker3013518 exited normally. Analysisall3 COMPLETE, bout_mechanismall3 COMPLETE. FullcomparisonPNG+samePDF andisotropictail/initial/fullcontext actuallyreviewed. Postprocessor2679166 healthy. No pendingexecsessions.

## Current scientific result

Authoritative scientific_review_live.md, conditional_summary.json/table.md and axis_controls/native_runs/analysis.json. Current18/18 summary, all18 nativePNG reviews saved. Map18/18 PNG and same-statePDF reviewed; humanPENDING. Figure legend now Mostly quiet because quiet criterion is >=95percent jointquiet samples, not absence of events. No classification thresholds changed.

- Z.25 K.02 bothhistories: global high ~476Hz, dZall−.05/s, dK~+70.5/s. K2 both: globalhigh~471Hz, dZ−.05, dK~+64.3. These heldK plateaus are not full autonomous-system attractors.
- Z.25 K8: highhistory southwest persistentpatch includingCoreA (allE137/A406/B0); interictalhistory northeast persistentpatch outsidebothcores (allE191/A0/B0/other200). Samecoarse mixed label, huge spatialdifference328Hz weightedmeanabsolute. Mean dZ positiveboth, but highhistory coreA dZ−.047/s, interhistory corespositive~+.153/s. Finite-window spatialhistory dependence, notcertifiedbistability.
- Z.75 K.02 bothpersistent/mixed, nobriefs/nojointquiet; allE127/126Hz but coreA179/130Hz. Spatialmeandifference33Hz; dZall~−.085/s. Couldbephase/transient, notproveddifferentattractors.
- Z.75 K2/K8 bothhistories fullyquiet, dZall+.05/s; dK−.4/−1.6 persecond. Quiet is notinterictalreturn.
- Z.95 K.02 bothhistories: sparse separated briefpropagation; final10s5events,4strongdoublecore (3Afirst,1Bfirst), oneweaknoncore event. Firstchronologicaltailbrief highabsolute34.31/inter72.31: AthenB, movingfield, sequentialcontacts. Jointquiet95.4percent, allEmean1.127Hz. Prespecified recurrent_brief criterion notmet (needs10events/span5s); donotinterpret quietlabelasnocoreevents. AllE dZ+.0070/s, bothcorespositive. analysis/highZ_lowK_paired_native_tail.json: full30s counts differ, final10s E/I andregioncounts and400bin5msfields bitwiseequal. Onecommonnoisestream, notindependentreplication orwholeengineconvergence.

Allcompletedfutureinputrecords identical, count/fieldconservationPASS. analysis/K2_Z_paired_contrast.json verifies twohistories K2 .25vs.75 jobsdiffonlyZ/name, fullKexact/futureinputexact/Zmonotone. analysis/K8_Z_spatial_recovery_contrast.json sameforK8highhistory. analysis/central_Z_K_contrast.json sameZlowKvsK2. These are conditional causal contrasts, notautonomousexit.

## Native graph controls

Rotated full120s: entry89.64(confirm89.84), lowexit90.34(confirm92.34), bothcoreZ reachCOMMONreference93.22 (owninitialreference91.48). Firstcompletebrief114.86–115.01,150ms,AthenBstrong, field/contactpropagation. From115.04→120 noqualifyingjointquietboundary: rightcensored activity >=4.96s, notabsentactivity and notcompletebrief. Onlyonebrief afterZ, noaccepted recurrentreturn sequence. Owninitialbriefbaseline absent, temporalcomparison null/NOT_ESTIMABLE_OWN_INITIAL_EVENTS. Originalmatched120s4entries3exits2returns. Fullinputs matched/countfieldintegrityPASS. rotated_complete_review.json; nativefullcontext+initial+post_exit1_first_brief PNGs agentreviewed.
Analyzer now explicitly reports complete_activity_episodes and right_censored_activity_episode; unchanged detector/thresholds/source2returns. return_estimability_validation.json retains audit.
Allthree0–8s nativeprefix PNG+same-statePDF reviewed currentturn, prefix_comparison/agent_visual_review.json. Original initial36brief; rotated0completebrief but repeatedcoreburstpackets/residualactivity; isotropic26completebrief. Common prefeedback .5–6s: original25brief/50.7percentquiet/19.46Hz, rotated0/0percent/46.23Hz, isotropic20/61.5percent/31.01Hz. AllE recoveryeligible .7543/.5749/.7017 and dZ−.02848/−.04956/−.033915 persecond. Firstfeedback9.8736/6.4821/7.4936. Nooperationalentriesinfirst8s. Isotropic shortevents argue against originalaxis beingnecessaryforeventexistence; longloopnotyetknown.
Graph construction angular_reassignment/{rotated,isotropic} passedSTATIC_CONTROL_PASS. Sourcepartnerreassignment within target xsourceobserverregion xexactdelay preservesincomingdegree/fullweightmultiset/nonEE/noise/thresholds. Currentaxis148.41deg/AR1.936; rotated56.75deg/AR1.577, isotropicAR1.028. Outdegree/sourceidentity andanisotropymagnitude notmatched. LoweredE outgoingmean−5.5percentrot/−13.7percentiso; incomingloweredsourcefraction30.02→29.21/27.84percent. Compositegeometrycontrols, notpureaxiscausality oruniqueclinicalpath. Rejectedweight-onlygraphs DONOTRUN.
Structural16: fourcoordinates(.25,.02),(.95,.02),(.75,2),(.75,8) x2histories x2newgraphs; reuse8originalcases. SwitchEEgraph atbranchstart; carryV/ref/currents/G/M/RNG andpopulatedolddelayrings(past35.8ms). SamefullZKfieldandfutureinput, fixed30s, final10s. Noautonomouscredit/noautomaticexpansion.

## Successful source, feedback law and recovery

Source /data/hfosp/topic4_sef_hfo/fig5_global_feedback_response_20260923/runs/G30_response0.5_s9108405,240s,fourcompleteautonomousreturns. Firstentry9.94/exitobserver16.70/jointlow16.83/return49.31/nextentry59.74. Completefirstbriefs49.31,100.86(weaknoncore),151.94,203.46; strongaftersecondreturn101.42. Shortnonreturnexits63.56/166.47; final221.77→240censored.
Law: taum dV=−V+IE−ZII−etaM M−ZG_raw(V−EG)−K(V−EK). EG−17.6628479,EK−30,etaM.0005,tauM1s,spikeMjump1. CausalallErate15ms R; q=clip((R−200)/300,0,1); tauG.5s ds=(q−s)/tauG,Graw30s. EspikeKjump.16q; Ktau5s ifprestepR<=5Hz else.5s. Ztau5s; dZi=(1[Ji<95.198513]−Zi)/5; Ji=localII+(18−EG)Graw,Kexcluded. Noquiettimer orZ/Mreset; no time-triggeredintervention.
recovery_mechanism/analysis.json+scientific_review.md: actual0.1msZgain/loss summed20ms balances1e−15. FirstGraw<2.6694 at17.551; two-corecontinuousnetpositive from17.58. BothcoresCOMMONZref23.90; firstbrief49.31,25.41s later. Acrossfourreturns Zreference→brief23–25s. AtfirstZrefK2.718/Kcurrent30.91mVeq, G/Mnegligible; atfirstbriefK.0169. Quiet20–40 K5sexponential andeligibleZlawsverified, notmanualwindow. Pairedseeds8402/8403 tauG0vs.5 differonlytauG; sameinputs/originalrhythmPASS. Fastgroup neverR<=5Hz, noexits; .5s both2returns. Feedbackkineticscontributes, notHopf/SN.
32000E8000I; fixed80raster20eachA/B/other/I. Physicalcore1.5mm,observer1.75mm; loweredE781,none raised; A754/B786. Common7.98sZref A.7603118192/B.7344912027. 15contacts fixedSCL9..6,ICL11..1,2ms weightedabsIE+absZII+absGcurrent excludesK/M; notHFO/clinicalenergy.

## Figures and analysis limits

Candidate /home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig5/candidates/autonomous_loop_20260924. build_fig5_autonomous_loop.py A0–60/rasterzooms.55–.85,16.52–16.82,49.30–49.60; BZ/K/appliedG/effectiveM withZref23.90/brief49.31; Cnativefields; DoldZ/localH/r andnewZ/K/r actualadjacent5ms, noforcedclosure. OldHregional[:,:,6], notgenericcurrent[:,2]. AllPNG/PDFagentreviewed,humanPENDING; canonicaloldA–F untouched. OldetaMscan/patientenergycannotrelabelnewmodel.
Gridcoverageaudit: actualfirstentryZ.741/K.000165,exitZ.213/K12.66,firstbriefZ.9986/K.0169. Grid[.25,.75,.95]x[.02,2,8] missesactualentry/exit/return; do notclaimcompleteboundaries after18. HeldK/gL statecoordinate isNOTk100parameter. Same means notcompletefullstate.
Rate gate conductance_response/static_calibration_v1 STATIC_VALIDATION_FAIL: simpletimecompression19/36addedg; frozen1024Sobolx512training/256x1024validation241/256strict254/256broad. Twoactual13–14Hz predicted29–39Hz, exceededbroadtolerance. No validationrefit orthresholdrelax. Transient/fullspatialcorrespondencenotrun, formalSN/Hopf/limitcycleNOTESTABLISHED. Contractexplicitlyallowsnativeconditional/driftfallback. Do notrestartoldkineticproject.
Nativeviewer review_topic4_loop_native_states.py uses firstchronologicalcompletebrief (otherwisefixed300ms),5msfields/fixed80raster/15contacts. Sourceinitial36/31strongcore,4returnwindows43/42/43/42events with34/36/34/34strongcore. Weakfirst100.86 retained. Do notclaimeveryeventcoreorigin orrestored directiondistribution. Meanspatialoccupancydoesnotmeasurepropagation.
Analyzer conditional drift rightendpoint20ms bins fixed to last500 not501; rates/events unchanged. heldregionalZ+tauZ*dZ reconstructeligibility, sameobservermasks. Zcolumn8 fractionJ>=Ith, NOTZstate. AllEpositiveZdriftcanhideCoreAdepletion. Means/finiteclassesnotattractors.

## Frozen execution and reproducibility

DO NOT EDIT active scientific run_topic4_loop_zk_conditional.py, run_topic4_loop_axis_native.py, run_topic4_loop_axis_conditional.py; theirsourcehashes/protocols frozen. Alsoverified wrappers/backends topic4_loop_locality_cpu.py, run_topic4_loop_zk_locality_cpu.py, run_topic4_loop_cuda_override.py, run_topic4_loop_axis_cuda_override.py, src/topic4_cuda_ordered_scatter.py frozen.
qa/locality_native_gate.json andqa/cuda_device0_route/gate.json twohistories.2s allobservations+wholeenginebitwise. qa/axis_cuda_device0_route/gate.json clampedreferencegraphloaderbothhistoriesall8streams+wholeenginebitwise. Original4serial→CPUlocality→GPU0 handoffs fullcheckpointbackups,identity/sourcehashes/8endpoints. backend_handoffs/gpu0/verification.json PASS; allfourcompleted. No persistedobsdiscarded; atmostuncommitted2sblockrecomputed.
Primarymax8total/up to4GPU0. Host>=88GiB,CPU<=65percent,GPU0free>=9.3GiB befdispatch. Cross-pollstartupreservationfixed. Otherjobs untouched; freeGPUmemoryvaries andcurrentlybelow8nominalreserve, noOOM, do notclaimcontinuousreserve. Do notrewritehealthybackend/managers.
Reproducibility snapshot scripts/snapshot_topic4_loop_reproducibility.py --label execution_20260925:174files1.70MB,25frozenprotocolsourcesverified. Actual importedprojectmodules/enginehelpers/jobs/protocols/gates/runtime/analysis/figurecode; versions/condabuilds/GPUdriver, noprivateenv. Largecheckpoints/graphs/rawarraysatprotocolpaths, notstandalonerelocatablearchive. Someanalysis changedafterinitialsnapshot; runnewfinal-labelsnapshotatfinaldelivery, neveroverwriteold.

## Next actions

Primary18 and both primary candidate maps are complete and reviewed; do not repeat them. Structural16 is dispatched (first4 live) and isotropic120 remains live. Inspect newly completed graph responses, native figures and actual transitions, then update scientific review and the final bundle. No extra parameter grid, seeds or horizons. Goal remains ACTIVE. No new permission, goal or unchanged analysis needed. Verified waits on live workers are appropriate.
Memory citation final: MEMORY.md:277-297|note=[Native correspondence required before bifurcation interpretation]; rolloutIDs01a09eae-c163-7cf2-8f2d-f11d43bdeaaf and01a0add1-6bb8-78f1-8af3-98dc7b0724b2. Do noteditCodexmemory. Olderchronologicalhandoff retained handoff_archive/before_14_completed_20260925.md.

## Historical milestones below (superseded by PRIMARY18 COMPLETE above)

### Added spatial summary candidate (2026-09-25T04:03:54.256752+08:00)

New scripts/paper_figures/build_fig5_conditional_spatial_fields.py consumes existing summary meanfields only. Producescandidatefigures/fig5-zk-spatial-fields.png/pdf/svg andconditional_spatial_metadata.json,14/18PNG+same-statePDFreviewPASS/humanPENDING. Native20x20bins,0–20mm,Fig5C magma/PowerNorm(.6,0,500),reusecores painter(physical1.5/observer1.75mm). Cellmapping andcellweightedallEmeansasserted. SameZ.25K8 supportsdifferentpersistentterritories acrosshistories; timeaveragesnotpropagationdirection. This producerisNOTwatchedbycurrentrunningpostprocessor: rerunmanuallyonceall18complete,thenreviewPNG/PDFagain. snapshot_topic4_loop_reproducibility.py nowincludesnewproducer. No scientificrunners or thresholds changed.

LiveverifiedPIDsprimary[2982900, 2990497, 2999948, 3001296], isotropic{'condition': 'isotropic', 'pid': 3013518, 'time_s': 12.0, 'state': 'RUNNING'}; primarycompleted14.

## HighZ K2 pair complete

BothZ.95K2histories last10s allE/A/B/other0Hz, jointquiet1, noevents, dZall+.01/s, dK−.399996/s, allregionseligible1. Nativefullcontext/fixedfinal300ms reviewed. analysis/highZ_K_paired_contrast.json checks eachhistory K.02vs2: onlyjobname/targetKdiff, configurations/identityequal, allrecordedZcolumns0..7exact (notcolumn8exposure), futureinputsexact. K.02 has5briefs/final10s vsK2none. Sameone-noiseseed, notindependentreplications orcertifiedbifurcation; addedauditdidnotunpicklefullZKarrays. Bothconditionalmap andspatialmeanfieldcandidate regenerated16/18, PNG+PDFreviewed. Spatialproducerstillmanual: rerunat18complete. Scientificreview updated16.

## Primary18 complete / structure16 launched

LastZ.95K8pair full30s/Equiet, allE dZ+.01/s,dK−1.599984/s; bothnativePNGs reviewed. Primaryplot+spatialplotfinal18/18 PNG/PDFreviewPASS, no needrerunabsentnewerror. primary_completion_audit.json verifiesall18endpoints30s, nativecountfieldintegrity, input18common/9pairs, networkidentity,24recordedsourcefiles+runnercurrenthash, all18percasevisualrecords/images. Audit initiallyfoundmissingpercaserecordsfor.75K.02high/.75K2both; actuallyviewedall6PNGagainandwroterecords, rerunauditPASS. No scientificdataerror. primary_scientific_review.md compacttableandboundedinterpretation. Fullgoalnotcomplete: 16structuralconditions+isotropic120stillrunning. NextfollowlivePIDsofstructuralbatch,notoldprimary; do notstartduplicateworkers.


## Completed native graph bout budgets (latest addition)

New read-only scripts/analyze_topic4_loop_axis_bout_mechanism.py aligns ALL existing activity episodes and right-censored final episode with actual20ms Zbudgets, 1ms causalR/q and20ms Z/K/G. No thresholds or simulations changed. Sourcecurrent0–120s androtated120s completed; outputaxis_controls/native_runs/bout_mechanism.json/md, posthocdescriptive. Allbudgetidentitiespassed; intervalquietfraction explicitlymeasured because filtered eventgaps neednot befullyquiet. Fourrotatedcomplete~7s bouts losecoreZ to~.56–.64 and buildK~1.4–2.6; subsequent~19–24s gaps are99.9–100percentjointquiet, gaincoreZ~.35–.44, decayK~.02–.04. Onlylastlongboutmeetsentrythreshold. Thusautonomousactivity/recoveryalternation survivesbutinitial/interictalbriefsequence doesnot. Final115.04–120boutrightcensored. scientific_review_live.md updated.

When isotropic120 completes AND native analysis.json refreshes, manually rerun this script once (watcher doesnot yet call it); inspect thirdrow and update interpretation. Snapshotwildcardalreadyincludesnewproducer. No pendingexecsession after completion.


## First structural conditional completed

rotated/z0.75_k2_interictal completed30s; both nativePNGs inspected andagent_visual_review.json written. Final10s allE/core/other0Hz, noevents, dZall+.05,dK−.399996, allregionseligible1. Current-axis pairedrates/meanfields/meancontacts/driftexact; futureinputs exact. Quietpointpreserved, no equal-boundary claim. Structurecomparison1/16, nextrotatedz.25K.02high dispatchedautomatically; re-read live status foritsPID. Remaining15notcomplete. No pendingexecsessions.


## Structural3 reviewed (supersedes first-completion count)

isotropic/z0.75_k2_high andinterictal nowcomplete30s, bothnativePNGpairs actuallyinspected andrecordsadded. Along withrotatedinterictal, all3 areE/core/otherquietfinal10s, noevents, eligibility1,dZall+.05,dK−.399996; pairedcurrentgraphregional/spatialmean/contactmean/driftexact, futureinputs exact. Interictalhistory nowhasall3graphs; rotatedhighstillrunning. Structure3/16; 13remain. Do not generalize one quietcoordinate to equalboundaries. Main scientificreview updated.


## Scheduler capacity update (authoritative over historical4total notes)

New manager3238772, up to6total with atmost4CPU and2GPU0. Original16scientificjobs unchanged; CPU cap remains original4, GPU route alreadyvalidated. Same resourceguards. Previousmanager2852545 alone terminated, all4existingworkerPIDs/creationtimes preserved andadopted. Newlylaunched3238782 isotropicz.25K.02interictal and3238783 rotatedz.95K.02high; total6active. Before/after source/status, job/queuehashes andverifiedhandoff savedexecution_updates/capacity_20260925/. Frozen physics sources untouched. Do not revert to4total merelyfromolderhandoff paragraphs. Re-read live status fornextworkers.


## Structural4 + checkpoint device prioritization (latest)

RotatedZ.75K2high finished andbothPNGs reviewed; all4newcentralconditions complete. All3graphs x2histories Efullyquiettail atZ.75K2, exactpairedmeanreadouts/drifts/futureinputs. Scientificreviewnow4/16.

CPU lowZ/lowK firstsimsecond took~700–770s (statusBUILDING persistsuntilfirst1s, notproofstillinitializing); GPU lowZ progressed5s in~883s includingstartup. ToavoidCPUtailbottleneck, manager nowprioritizeslowZCPU→GPU viafullcheckpointhelper. Manager3265151 live. New helper scripts/topic4_loop_axis_checkpoint_handoff.py, source/identity/jobchecks, freezeonlyownworker, requireALL observationstreams endpoints matchcp orinitialcpwithNO savedblocks, bytebackup thenresume frozenvalidatedroute. Recomputeatmostuncommitted2s, no persisteddatadiscard. Up to6total/4CPU/2GPU0 unchanged.

Two actualtransfers: isotropicZ.25K.02high CPU3218015→GPU3265375 and newlystarted isotropicZ.95K.02high GPU3252774→CPU3265376, bothat initialstep120000 withzero savedblocks. Allotherworkerscontinued. Backups/records backend_handoffs/axis_cpu_gpu/; execution_updates/axis_gpu_checkpoint_handoff_20260925/verification.json PASS_CHECKPOINT_AND_RESUME_IDENTITY; freshprogressPIDs/runtime verified, jobs/queues unchanged. FIRST NEW2sBLOCK CONTINUITY STILL TO CHECK onceproduced; thenfullfinalnativeanalysisrequired. CPUlaunch nowwrites correctruntime_backend.json sooldGPUsidecarcannotmislabelresumedCPU. Helper copied previousruntimeintoarchive.

Next: follow6activeactualPIDs and nativeisotropic120; verifyfirstnewhandoffblock start_step=120000 forboth transferredconditions, nooverlaps/gaps. Recordlaterautomatichandoffs similarly. Do not stopp healthyworkers dueBUILDING alone. Py-spy read-only stackattempt denied; do notescalate, existingprogressandprocliveness adequate. Protocol deadlinesremain~162h, noimminentwallcutoff. Finalreprosnapshotmustincludeaddedhelper/managerandboutbudgetproducer; wildcardalreadycovers them.


## First resumed blocks verified / helper stream correction (latest)

Both firstresumedblocks nowPASS120000→140000, exactsameinputrecords/heldZsummary0..7/Kmean/sampletimes asoriginalgraph correspondingblock; nativeE/Iregionalcountand5msfieldconservationPASS. Newreusable scripts/verify_topic4_loop_axis_handoff_blocks.py writes perhandoff first_block_verification.json andaggregate backend_handoffs/axis_cpu_gpu/first_blocks.json. Rerun onlywhennewhandoffblocksappear oratfinalcloseout; currenttwo PASS, no pendingblocks.

Firstverificationfound helperrequired autonomousz_budget_chunks, which CONDITIONAL CLAMP DOES NOT PRODUCE: frozenConditionalSlow.step recordsconditional_drift_chunks instead. All18originalcompletedconditionalsconfirmthe8streamcontract. Fixedhelperrequiredlisttoeight; extraactualdirectoriesstillchecked. Initialhandoffsremainedvalid(no savedblocks), butunfixedhelperwouldhavepreventedlaternoninitialcheckpointmigration. Manageronlyreloaded3265151→3282780, all6workerPID/creationtimesretained; no physics/workerchanges. execution_updates/axis_gpu_checkpoint_handoff_20260925/conditional_stream_fix.json PASS_MANAGER_RELOAD_ALL6_WORKERS_RETAINED, oldhelperarchived. Thiswas a verification-list mismatch, not missingdata. Currenthelperhash2eac0af593d2556629832ab44dde4b2964b3aba2e0ec86ef16f7ee4d2f442765.


## Fixed isotropic60s prefix review completed

scripts/review_topic4_loop_axis_halfway.py reads exactlycommitted0–60s blocks, allstreamfilenamebounds contiguous, sourcehashmanifest, countfieldconservation andactualZbudgetPASS. Outputaxis_controls/native_runs/prefix_checkpoints/isotropic60s/. Late50–60s28briefs,all28strongdoublecoreandAfirst; initial26briefs20Bfirst. Firstchronological50.18sPNG actuallyviewedandreviewrecordsaved: nearA radialexpansion thenB, sequentialcontacts. Userinlineimagealreadyshown. LateZall/A/Bmeans.762/.750/.762; gain.359689loss.372360delta−.012671 (coresalso slightlynegative), jointquiet.699, qpositivefraction.0096, meanK.02599. No operationalentry/exits infirst60; notpostictalreturn, nostableequilibriumorfeedbacknecessityproof. Main scientificreview updated. Do notrerununchangedprefix. Frozenrunnersunchanged; no pendingexecsessions.


## GPU1 execution-only handoff (latest)

Newmanager3290602; old3282780 exited normally. GPU1 .2s bothhistories fullengine andALL8streamarrays EXACT; qa/axis_cuda_device1_route/gate.json PASS, verifier scripts/verify_topic4_loop_axis_gpu1_route.py. Scientificrunner/wrapper/backend sourcesunchanged. Conditionaltotal6unchanged; GPU0cap2,GPU1cap1,CPUcap4; eachdevice separate1.25GiBstartupreservationsand9.3GiBfreesafeguard. GPU1 onlymigratesexistinglowZ/lowKCPUjobs. Otherjobs/nativeGPU1 untouched.

rotated/z.25K.02interictal CPU3218016→GPU13290615 atcommittedstep540000 (54s). Eightstreamendpoints andwholecheckpointbackup verified; all16jobhashes unchanged,other5conditional PIDs+native3013518 retained. execution_updates/gpu1_checkpoint_capacity_20260925/deployment.json PASS_MANAGER_AND_CHECKPOINT_RESUME. FIRSTNEW54→56sBLOCK PENDING; run scripts/verify_topic4_loop_axis_handoff_blocks.py oncecommitted. ExistingfirsttwohandoffsPASS. No pendingexecsessions afterQAandmanagerdeploymentfinish. No furtherresource/backendchanges needed unlessnewerror.


## GPU1 firstblock verified (supersedes pending above)

rotated/z0.25_k0.02_interictal54→56s firstresumedblockPASS. Enhanced scripts/verify_topic4_loop_axis_handoff_blocks.py verifies everyarray shape andalltimestamp arrays againstpairedoriginalgraphstream inALL8streams, plusnativecount/field/input/Z/Kchecks. Driftstreamfilenamefirststep540200 isitsfirst20msrightendpoint, NOTmissing20ms; actualtime54.02→56 contains100integratedbins andmatchesoriginal. Initialattempttoforcesamefilenameasmainobserver wascorrected aftercheckingoriginalstreamschema; no simulation ordatachanged. AggregatestatusPASS_FIRST_BLOCKS,pending0. execution_updates/gpu1_checkpoint_capacity_20260925/deployment.json PASS_RESUME_AND_FIRST_BLOCK. No pendingexecsessions.


## Structural5 reviewed: isotropic highZ/lowK highhistory

isotropic/z0.95_k0.02_high finished30s, originalworker3265376exitednormally. BothfullcontextandfirsttailbriefPNG actuallyviewedandrecordedagent_visual_review.json; originalsamecoordinatefirstbriefPNGalsoreviewedsidebyside. Tail3briefs110–130ms,all3strongdoublecore,2Afirst/1Bfirst,jointquiet.964,allEmean1.93184Hz vsoriginal5briefs/4strong/.954/1.12685Hz. Firstchronologicaleventboth34.31s; isoexpandingbroadwavefromAtoB vsoriginalmorelocalizedfield. ExampleallE5mspeak105vs31Hz, so fewerbriefsdoesnotmeanlowertotalmeanrate. Samefutureinputexact andheldZmeans; dZall/corepositive,dKnegativeboth. Mostlyquietlabelretainssparseevents; nofullrecurrencestandardsatisfied,nobifurcationorautonomouscredit. Main scientificreviewupdated5/16. Rotatedhighhistoryandotherhistorypending. Reviewmapagainonceall24complete, notforunchangedpartialcount. CurrentactivePIDslistedattop. No pendingexecsessions.


## Structural7 + threegraphhighZcomparison (LATEST SCIENCE)

RotatedZ.95K.02high androtatedZ.25K.02high complete30s, originalPIDs3238783/3209719exitnormal. BothpairsnativePNGsactuallyviewed, recordsAGENT_VISUAL_REVIEWED. LowZlowKrotatedglobalplateau476Hz,dZ−.05,dK+70.5,nobriefs/nojointquiet, notautonomousattractor.

HighZlowKhighhistorynowall3graphs complete. RotatedallE31.857Hz,coreA35.1/B37.57,nojointquiet/nocompletebrief; repeatedstrongcorebursts withnoncorebackgroundandcontinuednativewaves. Original5briefs(4strong),iso3(allstrong),bothmostlyquiet. SamefutureinputEXACT. EligibleZfractioncurrent.98508/rotated.63044/iso.97712 givesdZall+.007016/−.063912/+.005425, bothcoreslikewise. Rawfeedbackfinal10s q0(all1mssamples),identicalGrawmax1.926e−17,K.02; rotatedcausalRmin11.53Hz. Geometryaffectsresource-recoverydirection atcommonZK, notmerelydifferingcurrentG; outdegree/sourceweightconfoundretained. NotcertifiedoscillationorSN/Hopf.

NEWscripts/plot_topic4_loop_axis_event_comparison.py --history high producedaxis_controls/conditional_runs/event_comparison/high/figures/axis_highZ_lowK_propagation.png/pdf/svg, actualPNG+samePDFreviewPASS metadata, humanPENDING. Three rowsrates+6native5msframes, original/isofirstbrief34.31s vsrotatedfixed41.7–42slice(reference41.74,noteventonset). Samefields/cellcounts/geometry/futureinputsasserted. Scientificdiagnostic, notreplacementmainFig5layout. Produceralso writesfeedback_tail.json. MUSTrun --history interictal onceallthreeinterictalZ.95K.02casescomplete, thenPNG/PDFreview. Scriptwildcardinfinalsnapshotincludesproducer.

GPU0slotfreedbyrotatedlowZhigh; manager3290602 automaticallymigratedisotropicZ.25K.02interictal CPU3238782→GPU03341973 atstep560000. Allcommitted8streams/backupverified; firstnew56→58sblockPENDING (aggregateverifierstatusPENDING_BLOCKS,pending1). NewCPUworkers3340500 isotropicZ.95K.02interictal and3341974 rotatedZ.75K8high. OtheractivePIDsattop. Oncefirstblockwritten, runverify_topic4_loop_axis_handoff_blocks.py. No furtherbackend/schedulerchangesneeded. No pendingexecsessions.


## Fourthhandofffirstblock PASS (supersedes pending above)

isotropicZ.25K.02interictal CPU3238782→GPU03341973 at56s nowfirst56→58sblockPASS. All8streams correctpairedfilename/sampletimes/arraydimensions; inputs,Zsummary0..7,Kmean exact; nativeE/Iregion+fieldcountconservation; checkpointbackuphashintact. Aggregateverifierbackend_handoffs/axis_cpu_gpu/first_blocks.json PASS_FIRST_BLOCKS,pending0,4transfers. Do notrepeatunchangedfirstblockchecks unlessnewhandofforfinalaudit. No pendingexecsessions.


## Structural9 reviewed (latest)

IsotropiclowZlowKhigh complete andnativePNGpair viewed: all3graphs highhistory atZ.25K.02global~476Hz,nojointquiet/briefs,dZ−.05,dK+70.5,inputs exact. Pairedspatialmeanfielddifferencevsoriginalrot.001981/iso.001791Hz; notwholeengineconvergence/equalboundaries/autonomousattractors. RotatedZ.75K8high completeandPNGpairviewed: Equiet, sparseI/currentfluctuationsremain; allE dZ+.05,dK−1.6,eligibility1,pairedcurrentmeanreadouts exact. Scientificreview9/16; remaining7. NextinspecthighZlowKinterhistorieswhencomplete, thenrun eventcomparisonproducer --history interictal and PNG+PDFreview. No new simulations or source changes.


## Structural10 reviewed (latest)

IsotropicZ.95K.02interictal complete andbothnativePNGs actuallyviewed/recordsaved. Tail3brief110–130ms,3strongdoublecore,2Afirst1Bfirst,jointquiet.964,allE1.93184,dZ+.005425; same descriptive metrics as highhistory. First72.31s A→outwardexpansion→B→extinction/sequentialcontacts. Sharedinputmeans notindependentreplication; wholeengineconvergencenotchecked. Rotatedcorrespondinginterhistorystillrunning; oncecompletegenerate3grapheventcomparison --history interictal, PNG+PDFinspect. Structuralpendingqueueempty, all6remainingactive. Re-readstatusforlastisoK8interictalPID.


## BothhighZlowKhistoriesreviewed / interictalcomparison complete

Rotatedinterictal highZlowK complete andbothPNGs reviewed. Continuousmovingfield/coreB→A→B,zerojointquiet/briefs,meanE31.637,dZ−.06314,eligible.6343;q0/Graw<7.4e−46sameasothergraphs. Thusbothhistoriesshowcurrent/iso positiveZdrift vsrotatednegative, noformalbranchclaim. Produced plot_topic4_loop_axis_event_comparison.py --history interictal; PNG+samePDF actuallyreviewed, metadata+recordPASS/humanPENDING. BOTHhistoriescomparativefiguresnowDONE, no needrerun. highZ_lowK_history_comparison.json: current/iso histories tail10s E/I/region/fieldcountsexact, full30sdiffer;rotatedtaildiffers. Payloadinputsexact+relativeclockmatched (absoluteclockcolumn0differs38s asintended; firstrawarraycomparisoncaughtclockorigin, correctedselectionmatchesexistinganalyzer). No wholeengineconvergence,independentreplication,orbistabilityclaim. Newsciencein scientific_review_live.md. No pendingexecsessions.


## Structural12 reviewed (latest)

RotatedlowZlowKinterictal COMPLETE30s; fullcontext/fixedtailPNGpairactuallyviewed andrecordsaved. Global476.212Hz,0brief/0quiet,dZ−.05,dK+70.516; notautonomousattractor. Remaining4structurejobs alreadyactive: [{'condition': 'isotropic', 'name': 'z0.25_k0.02_interictal', 'pid': 3341973, 'time_s': 69.0, 'state': 'RUNNING', 'backend': 'cuda_ordered:0'}, {'condition': 'isotropic', 'name': 'z0.75_k8_high', 'pid': 3416565, 'time_s': 20.0, 'state': 'RUNNING', 'backend': 'cuda_ordered:0'}, {'condition': 'rotated', 'name': 'z0.75_k8_interictal', 'pid': 3420739, 'time_s': 66.0, 'state': 'RUNNING', 'backend': 'locality_cpu'}, {'condition': 'isotropic', 'name': 'z0.75_k8_interictal', 'pid': 3440713, 'time_s': 58.0, 'state': 'RUNNING', 'backend': 'locality_cpu'}]. Native {'condition': 'isotropic', 'pid': 3013518, 'time_s': 113.0, 'state': 'RUNNING'}. No pendingexecsessions; BOTHhistoryeventcomparisonsDONE. Finalstructural24pointPNG+PDFstillmustwaitall16newcomplete; isotropicnative120+all3boutbudgetstillpending.


## Delivery navigation prepared (stillpartial)

Newrootdelivery_index.md links actualcandidateA/B/C/D, bothconditionalprimaryfigures, mechanisms/negativeclosure, BOTHhistorystructuralpropagationfigures, runtime/analysis andexistingexecution snapshot. All26localtargets checkedexist. CandidateREADME backlinksindex. Currentindexexplicitlystates4structure+isotropic120/finalmap/finalsnapshotpending; update these atfinalcloseout, thenno stale runningtext. No scientific/worker changes.


## Native all3 complete / newfullcomparison andisotropictailDONE

Bothnew120sgraphcontrols COMPLETE; nativeanalysis rowscurrent4entries3exits2returns,rot1/1/0,iso0/0/0(no priorentry, notpostictalreturn). Iso314completeepisodes70–140ms,final119.96→120rightcensored40ms. Last110–12020brief allstrongdoublecore15A/4B/1same; first112.85visiblefieldbetweencores→outwardexpansion→bothcores, notuniversalcoreorigin. TailZmean.887 butactualgain.179689loss.282022delta−.102334; corelossalso, budgeterror~1e−16. FullZminimumall/A/B .720/.701/.717. No stableattractorclaim.

New scripts/plot_topic4_loop_axis_native_full.py andscripts/review_topic4_loop_axis_isotropic_tail.py executed. Full_comparison/figures/axis_native_120s.png/pdf andpdf_review actuallyviewedafterlegendsmovedoutsidedata; metadata+agentrecordPASS,humanPENDING. Isotropic_completed_tailPNGactuallyviewed andrecordsaved. Nativeisotropic fullcontext+initialPNGpairviewed/recordsaved; isotropic_complete_review.json verifies24frozenphysics+2runners/graphaudit, identity/input/countfield andprocessesexited. RerunboutmechanismalreadyDONEall3; no needrepeat. Newnative/scientific_review.md written, delivery_indexupdated. Finalsnapshotwildcardcapturesnewproducers. Onlyremaining4structureconditions+final24pointmap/finalcloseout left. Currentphysicalworkersunchanged.


## Structural13 reviewed (latest)

RotatedZ.75K8interictal30scomplete andPNGpair actuallyviewed/recordsaved. Equiet,dZall+.05,dK−1.6,eligibility1; sparseI/currentremain. Remaining3areisotropicz.25K.02interictal, z.75K8high/interictal, allhealthy. No pendingexecsessions. Final tasks: inspectremaining3PNGpairs, final24pointstructuralmapPNG+PDF andmetadatareview; finalscientificsynthesis, requirement-by-requirementcompletionaudit, finalreprosnapshot andindexrunningtextupdates. Native120fullcomparativefigure/boutbudgets/tailDONE; do notredo.


## Final audit producer prepared (not yet run)

New scripts/audit_topic4_loop_delivery.py requires --snapshot LABEL and ALL main/native/structural/postprocessing stages COMPLETE with theirPIDsexited. Re-reads34conditionalcases, checks15committedblocks inALL8streams, actualnativecount/field/heldZ checks, commonfutureinput, full30s, resultjob/identity/frozenphysics, everypercase reviewedPNGhash. Re-readsall3native120analyses; verifiescandidate/primary/structure/event/fullnativefigurePNG+PDFmetadataandnegative-rate-gateboundary; verifiesfinalreprosnapshot unchanged. Needsfinal_scientific_review.md anddelivery_index.md. Notrunwhile3branchespending; notanewtestofthefailedrateclosure. Syntaxcompiled; streamfilenameassumptionsconfirmed againstcompletedcase (driftfirst+200). Finalstructuralfiguremetadataagent_latest_render_reviewmustbesetPASSafteractualPNG/PDFview. Newauditwildcardcapturedbyfinalsnapshot. snapshotproducerREADME madevalidforcompletedcampaign too; oldsnapshotsuntouched. No pendingexecsessions.


## Structural14 reviewed (latest)

IsotropicZ.75K8interictal complete30s; PNGpairactuallyviewed andrecordsaved. All3graphsthiscoordinateinterhistoryEquiet,dZ+.05,dK−1.6; sparseI/contact remain. Remaining2GPU0branches isoZ.25K.02interictal~79/80s andisoZ.75K8high~36/42s; rereadstatus. Finalauditproducerpreparedpreviousappendix, no pendingexecsessions.


## Structural15 reviewed (latest)

IsotropiclowZlowKinterictal completed30s andbothPNGsactuallyviewed/recordsaved. All3graphs x2historiesthiscoordinateglobal~476Hz,0brief/0quiet,dZ−.05,dK+70.5,pairedfutureinputexact. OnlyisotropicZ.75K8high remains, PID3416565 healthy at~38/42s; re-read live status. Nextaftercomplete: inspectlastPNGpair; automaticmap24/24 thenPNG+samePDFreview andsetagent_latest_render_reviewPASS. Needrootfinal_scientific_review.md +final_summary.json, updateddeliveryindexandnative/candidateREADME completionwording, finalsnapshotunique label (audit+newnativeplots captured), then audit_topic4_loop_delivery.py --snapshot LABEL. Finalauditnotyetexecuted. Goalnotdoneuntilall verified. No pendingexecsessions.
