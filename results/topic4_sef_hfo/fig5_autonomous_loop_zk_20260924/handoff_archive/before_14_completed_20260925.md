# Active Figure5 goal handoff — 2026-09-25

Goal ACTIVE: complete the already bounded18 native conditional branches +2 autonomous structural120s runs +16 structural conditional branches, analyze/render/review them. Do not mark complete while required simulations remain. Slow simulations are not a block. User explicitly authorized goal; no new subagents, no Codex memory edits. Use /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python. Preserve unrelated jobs/worktrees. Current turn added GPU execution for pending16 (validated), verified existing GPU handoffs, and found a real grid-coverage limitation. No new scientific conditions.

## Current processes (verify live before any action)

As of 2026-09-25T03:09:05.852257+08:00: primary supervisorPID2821859, 6/18complete, 8active, 4pending, failed={}. Maximum8total/up to4GPU0, host>=88GiB/CPU<=65percent/GPU0free>=9.3GiB beforedispatch; startupreservation1.25GiB carriedacrosspolls. Preservehealthyworkers.

Completed: z0.75_k2_high, z0.75_k2_interictal, z0.75_k0.02_high, z0.75_k0.02_interictal, z0.25_k8_high, z0.75_k8_high.

- z0.25_k0.02_high: PID2806152, cuda_ordered:0, absolute t=34.0s, RUNNING.
- z0.25_k0.02_interictal: PID2806153, cuda_ordered:0, absolute t=75.0s, RUNNING.
- z0.25_k2_high: PID2806845, cuda_ordered:0, absolute t=34.0s, RUNNING.
- z0.25_k2_interictal: PID2806960, cuda_ordered:0, absolute t=73.0s, RUNNING.
- z0.25_k8_interictal: PID2640175, locality_cpu, absolute t=75.0s, RUNNING.
- z0.75_k8_interictal: PID2964940, locality_cpu, absolute t=63.0s, RUNNING.
- z0.95_k0.02_high: PID2966643, locality_cpu, absolute t=15.0s, RUNNING.
- z0.95_k0.02_interictal: PID2968246, locality_cpu, absolute t=51.0s, RUNNING.

Absolute highhistory starts12s/ends42s; interictalhistory starts50s/ends80s. Everyconditional duration30s.

Nativeaxis manager2603204, rotatedworker2605911 remainslive (~103s/120s lastchecked); isotropicqueued. Parsednativeanalysis96s has1entry1exit, Zreference93.22, noafterZbriefs; owninitialcomparisonnull becauseinitialbriefs0. Full120stillpending.

Structuralconditionalmanager2852545 WAITING_PREREQUISITES: original18complete/full30s/analysis +bothaxis8sprefix/loadQA +referencerouteQA. No16sciencejobslaunchedyet. Max4, GPU0memoryguard orCPUfallback. Postprocessor2679166 watchescompletedresults/plots only, lastnoerrors. Noactiveunifiedtoolsessions.

## Execution correctness and frozen files

DO NOT EDIT active scientific runners: run_topic4_loop_zk_conditional.py, run_topic4_loop_axis_native.py, run_topic4_loop_axis_conditional.py; original source_hashes/runnerhash in protocols. DO NOT EDIT verified active backends/wrappers: topic4_loop_locality_cpu.py, run_topic4_loop_zk_locality_cpu.py, run_topic4_loop_cuda_override.py, src/topic4_cuda_ordered_scatter.py. New structural GPU wrapper run_topic4_loop_axis_cuda_override.py now FROZEN by gate too.

Original four serial branches moved to localityCPU thenGPU0 through complete checkpoint backup, identical job/identity,24sourcehashes,all8observation endpoints. No persisted observations discarded; atmostone2s uncommittedblockrecomputed. GPU0 backups in backend_handoffs/gpu0/NAME, verification.json now PASS_FULL_STATE_TRANSFER_AND_NEW_COMMITTED_BLOCK (all8streams advanced). OldCPU PIDs2717297/2717302/2718093/2718095 intentionallyterminated for these handoffs; not failures. Earlier serial backups retained separately. GPU0 runtime metadata preserves plannedjob while records actualdevice/backend.

qa/cuda_device0_route/gate.json: original high12s andinter50s0.2s observations/fullengine bitwise against native/serial source.
qa/axis_cuda_device0_route/gate.json: both clampedreference histories0.2s through exactgraphloader+CUDA0 match all8observations AND completeengine of originalserialclamp references. Both QA workers finished normally. New supervisor verify_backend checks originalrunner,wrapper,CUDAbackend andCPUkernel hashes before dispatch. No active tool sessions remain after this turn.

GPU first manager forgot cross-poll startup reservations; initial4allocations briefly left7.31GiB vsnominal8GiB. Fixed2821859, preserved liveworkers, noOOM/otherjobsstopped. FreeGPUmemory varies as otherusersrun; do notclaimreservecontinuouslyheld. CPU/GPUmicrobench speedups are notfulltrajectory performance. Do not keep changing managers/backends absent a concrete error.

## Scientific source and Figure5 candidates already complete

Source /data/hfosp/topic4_sef_hfo/fig5_global_feedback_response_20260923/runs/G30_response0.5_s9108405,240s, four completeautonomous returns. Firstentry9.94,exitobserver16.70,jointlow16.83,return49.31,nextentry59.74. Otherreturns:exit70.31→weak100.86,strong101.42;119.37→~151.8;173.98→203.22. Shortnonreturns63.56/166.47;last221.77→240censored. Pairedsource120s has4entries3exits2returns, not4returns.

Law: τm dV=−V+IE−Z II−ηM M−Z Graw(V−EG)−K(V−EK). EG−17.66285,EK−30;ηM.0005,τM1s,jump1. CausalglobalallErate15ms; q=clip((R−200)/300,0,1). ds=(q−s)/.5s, Graw30s. E spikeKjump.16q;Kdecayτ5s ifcausalR<=5Hz,else.5s. Zτ5s, recoveryeligible Ji=localII+(18−EG)Graw<95.198513;KexcludedfromZload. dmeanZ=(eligiblefraction−meanZ)/5. Fixed statefeedback, noquiettimer orZreset. Gload can delay Zrecovery after ratefalls; Kretention buysquiet, Zrecovers,Kdecays,eventsreturn. This is a new hypothesis inspired by literature, not exactLiouimplementation. Jobk1008 is hypotheticalq1 normalization, not actual100HzsteadyK.

Candidate root HFOsp/results/paper-ready-figure/fig5/candidates/autonomous_loop_20260924. Producer scripts/paper_figures/build_fig5_autonomous_loop.py. A0–60fixed80raster andzooms. BZ/coreK/appliedG/effectiveM. Cnative50msmaps. DoldmeanZ/localH/r andnewZ/K/r actual5msadjacentstates, noforcedclosure. OldH = regional[:,:,6] weighted, NOT genericcurrents[:,2] containingGload. PNG/PDF/SVG/inputNPZ/metadata; sourceA/stateprojection visually reread this turn. HumanPENDING. OldcanonicalA–F unchanged; oldetaMscan/patientenergy not relabeled asnewmodel.

32000E8000I,20fixedcells eachA/B/other/I. Physicalcore radius1.5mm,observer1.75mm. LoweredE781, none raised; observerA754/B786. Original0–8s bitwise no-feedbackreference. Currentproxy fixed15contacts2ms(absIE+absZII+absGcurrent),excludesintrinsicM/K;notHFOorclinicalvalidation. CommonZref lastsample<=8s of references/native_s9108405.npz is7.98s,not fullrun8.000s.

## Latest recovery-mechanism evidence and candidateB update

scripts/analyze_topic4_loop_recovery_budget.py -> recovery_mechanism/analysis.json + scientific_review.md. Reads existing completeddata only, no newtrials. Native20msbudgets sumactual0.1msgain/loss; maxerror~1e-15. FirstGrawcrossesresourceblockthreshold2.66940 at17.551; bothcores netpositive>=1s frombudgetendpoint17.58. BothZreachoriginal7.98sreference23.90;firstcompletebrief49.31. Fourfullreturns have23–25s delayafterZreference. K/gL atfirstZreference2.718,meanKcurrent30.91mV-equiv; G/Mcurrent negligible. FirstbriefK.01691. Firstquiet20–40s K5sexponentialerror1.2e-11 andZidealEulerrecovery agree; no fittedtimer. NotproofKalone deterministicallytriggersfirstevent. Currentstrictfirstbrief timesfourreturns49.31,100.86,151.94,203.46 (earlierapprox151.8/203.22 were notexactacceptedcompletebriefonsets). Verified current spatialrevieweventidentity.

ExistingG30 matchedseed8402/8403 controls: jobdiffonlyname/global_tau_s,identity/futureinputs120sexact,originalrhythmPASS. tauG0 noexits/returns; postentryminimumcausalR~74/85Hz,neverR<=5Kretentionrule. tauG.5 crosseslowregime,each2fullreturns. Feedbackkineticsmatterswithinthismodel;noformalHopf/SNclaim. Statisticunits2pairedseeds,notalltimepoints.

Figure5B producer build_fig5_autonomous_loop.py nowmarks Zreference23.90 andreturnedbrief49.31,25.41sdifference; markersareobservations,notcontrols. All4panelsrerendered; A/C PNGhashesunchanged. Dnowusesobserved16.83lowactivity/23.90Zreference/49.31returnedbrief markers, replacingarbitrary17/25representativetimes; 5msplottedsampletimes explicitlyrecorded. B/DPNGand allsame-statePDFsvisually inspected. Metadata/publicationQA updatedafterview, humanPENDING. PriorB+metadata archivedcandidate/prior_without_recovery_markers. READMEpreservesconditionalmapsection. No runtimephysics changed.

## Original18 conditional design and NEW coverage limitation

3x3 meanZ[.25,.75,.95],meanK[.02,2,8], two endogenoushistories12s/50s. Same20s fullspatialZ/Ktemplate, Zlogitshift andKscale; same50s exogenouscompleteRNG/inputstate withclockshift. V/refractory/delayrings/currents/G/Mhistoriesretained. HoldfullZ/Kfields,allowG/Mdynamic;counterfactualnativedZ/dK every20ms. Three completed: central.75K2 pair bothquiet, plus.75K.02high mixed;pairedfutureinput30s exact. Tail10s descriptions high/brief/quiet/mixed; notattractors, autonomousloops, orcertifiedbifurcations.

IMPORTANT NEW AUDIT: scripts/audit_topic4_loop_grid_coverage.py -> trajectory_grid_coverage.json. Native20msfirst60smeanZrange.1804–1,K0–13.3057. FirstentryZ.741,K.000165;exitobserverZ.213,K12.66;firstjointlowZ.208,K12.19;firstreturnedbriefZ.9986,K.0169. Thus currentninepointgrid does NOT enclose entry/exit/return path. Even full18cannot locate complete transition boundaries. Do not silentlyexpandgrid: executioncontract explicitlybounded18/noadaptiveextras. Complete existingconditionals and reportthegap; observedautonomousreturn itself remainsvalid. Same meanZ/K atdifferentnative times neednot share fullspatialfields/G/M/history. Two carriedhistories different after30s≠provedbistability.

Conditionalmap scripts/paper_figures/build_fig5_conditional_zk.py now includes range-coverage caveat; PNG/PDFvisually inspectedthisturn. Now4/18conditions shown; remainingcrosses. Arrowsdirectiononly conditionalnativedrift, not2Dautonomousfield. HumanPENDING.

## Latest completed conditional response (2026-09-25 ~02:32)

z0.75_k0.02_high completed30s, fulltail32–42s absolute. Classmixed_or_transient, allE127.22Hz/coreA179.46/coreB275.44; highfraction0,jointquietfraction0,briefs0. Fullratecontext +fixedfinal300msnative80raster/5msfield/15contactview visually inspected: persistent axial bands/coreactivity, NOTisolatedinterictalreturn. Conditionalmap now3/18; structuralfourpointmap stays2/24 because thispointisnotin its fourcoordinates. Otherhistory lowK stillrunning; nohistorydependence/bistabilityclaim.

At sameZ.75/highhistory, compareK.02 vsK2: jobdiffonlyname/target_K,identicalnetworkidentity andZstate statscolumns0..7; allcompletedconditions futureinputrecordsexact (analyze_topic4_loop_zk_conditional.py nowcheckscommon_future_input_checks). analysis/central_Z_K_contrast.json. Zcoremeans.735616/.732510. TailrecoveryeligiblefractionallE.32508 vs1;counterfactualmeanZdrift−.08497/s vs+.05/s. Klow permitspersistentmixedactivity,not recoveredbriefs. NativeZrecordcolumn8 isfractionJ>=Ith (thresholdexposure),NOTaZstate orrecoveryeligiblefraction; recoveryeligible=1−column8. Existingaxisbudgetproducer alreadyuses1−column8correctly. Earliercomparison ofall9columns was an analysisassertion mistake, corrected by comparingactualstatecolumns0..7; existing scientificresults unchanged.

Auto spatialreview/summary/map completedwithoutanalysiserrors. NewCPUworkerz.75K8highPID2916952 takes freedslot; do notstartduplicates. Currentprimarymanager unchanged2821859.

## Latest paired-readout analysis and rotated transition (~02:53)

Updated analysis-only producers analyze_topic4_loop_zk_conditional.py and analyze_topic4_loop_axis_conditional.py: preserve last10s regionalrates,400bin meanEfield,15contact mean proxy; compare histories and completedgraphs descriptively. Cell-count-weighted fieldmean independently reconstructs allErate; no equivalence threshold, no propagation-direction claim frommeanfield. Fullspatial eventreview remains required. Existing3 event/rateclassifications unchanged. QuietK2pair haszero regional/spatial/contactmean differences, notproofcompleteengineconvergence.

Corrected actual driftwindow offbyone: driftrecords are preceding20ms averages timestamped at rightendpoint. Select(horizon−10,horizon]500records, notold>=left501records. LowKhigh dZ now−.084932331875/s (old−.084965155/s), directionunchanged. Allscienceworkers/sourcephysics untouched. analysis/paired_readouts_validation.json PASS; oldsummary/scripts archivedanalysis/prior_to_paired_readouts_20260925. Automatedwatcher caught changes without failures. Axisanalyzer manually rerun once, PASS; no pendingunifiedtoolsessions.

Rotatednative newactualsaved94sprefix: entry89.64(confirm89.84), autonomouslowexit90.34(confirm92.34), bothcoresbacktooriginalcommonZref93.22, no briefreturns by94. No full120resultyet. Exactmatchedexternalinput/countfieldintegrityPASS. axis_controls/native_runs/rotated_prefix94_review.json. Currentanalysis.json refreshedto94s; earlier60s noentry is censoredhistoricalobservation, not inabilitytoenter. No newpropagationclaimuntilfullnativeviews. Scientificreview_live updated.

## New fourth conditional completed (~02:55)

z.75K.02_interictal full30s complete, old2640177normalexit; newz.75K8interictalCPUworker2964940autoscheduled. LowKpairbothmixed/persistent,nojointquiet/nobrief last10s. InterhistoryallE126.457/coreA130.037/coreB286.595/other122.237Hz; dZ−.0857037/s,dK−.039996/s. HighhistoryallE127.219/coreA179.461/coreB275.442/other122.101. Inter−high CoreA−49.423Hz; cellweightedmeanfieldabsolute difference33.1123Hz, meanallE−.761Hz. Sameglobalmean/coarselabel hides spatialdifferences; differences couldbephase/transient, notprovenseparateattractors. Fourconditions inputrecords exact; native count/fieldconservationPASS. Newfullratecontext +fixedfinal300msraster/5msfield/contact reviewed: persistent axialbands/coreBactivity,notinterictalreturn. agent_visual_review.json saved. Conditionalmap4/18PNGandPDFbothreviewed,metadatasetPASS_4_OF_18; humanPENDING. Noactivefunctions/unifiedsessions.

## Native axis return estimability correction (~02:58)

Analysis-only analyze_topic4_loop_axis_native.py corrected a concrete scientific reporting issue: rotatedinitial has0isolatedbriefs, so duration/interval/peak similarity toitsowninitial isnotestimable, notfalse. New temporal_return_screen andtotal temporal_returns are null wheninitialfeaturesmissing. Separate brief_recurrence_after_common_Z_screen andbrief_recurrence_episodes_after_common_Z useunchanged existing10events/5span/80percentbrief rule; nodefinitionretuned/noalternatebaseline silentlysubstituted. Currentgraph120s preservesexactevents/exits/oldreturn screensand2returns; rotatedsaved96s has1entry1exit, noafterZbriefs, owninitialcomparisonNOT_ESTIMABLE. ValidationPASS inaxis_controls/native_runs/return_estimability_validation.json. Oldanalysis/script preservedprior_to_return_estimability_20260925. Automaticwatcher detectsproducerchange; nochangefrozenphysics. Do notsummarizerotatedtemporal_returns nullas0failures. Full120trajectorystillpending; newreferencecomparison isnotrequiredwithoutaccepteddesign. Sourcecandidate alreadyhasvalidinitialinterictalevents.

## New fifth/sixth conditional results and regional recovery (~03:06)

Completedz.25K8high andz.75K8high; old2640174/2916952exitednormally, newCPU.95K.02 high2966643 andinter2968246autoscheduled. LowZK8high haspersistentlocalizedsouthwestpatchcontainingcoreA (allE137.116/coreA406.407/coreB0/other133.988Hz), nojointquiet/nobriefs. Counterfactual dZallE+.070729, A−.0470734, B+.123967/s. CoreAheldZ.235367 andeligibility~0, whileallEeligibility.603645. SamehighhistoryK8 atZ.75 isquietallE; allcoresdZpositive. analysis/K8_Z_spatial_recovery_contrast.json verifiesjobsdiffonlyZ/name, identitysame, completeheldKarraysbitwiseequal, ZhigherineveryE, futureinputsidentical. This isconditionalinteraction/recoveryevidence, notautonomousexit/bifurcation.

analyze_topic4_loop_zk_conditional.py nowrecordsheldregionalZ andexactreconstructedeligibility=heldZ+tauZ*counterfactualdZ (sameobservermasks verifiedfromnativecode). Floatingroundoffcan give1+2e−15; tolerance1e−12, notphysicalovershoot. Addsregionalrate/dZtable soallEmean cannot hideactivecoreconsumption. Existing6count/field/tailchecksPASS. Newtwofullcontext/final300msnativeviews visuallyinspected; agent_visual_review.json saved. Conditionalmapupdated6/18, PNGandPDFreviewed; footnotespecifiesmean/core driftscanopposesigns. HumanPENDING. Otherphysics/scriptsfrozenunchanged.

## Reproducibility snapshot (~03:23)

snapshot_topic4_loop_reproducibility.py read-onlyarchivedcode/config/environment, noengine/GPUworkerstarted. Outputreproducibility/execution_20260925:174files,1.70MB;25frozenprotocolsourcesverifiedandallcopiesreadbackSHAchecked. Includesactualimportedprojectmodules, enginepyhelperdirectories, fixedjobs/protocols/QAroutegates, runtimewrappers/analysis/figureproducers. environment.json hasPython/packageversions/condabuilds/GPUdriver, nofullenvironmentorcredentialURLs. Largeimmutable source states/graphs/rawobservations remainatprotocolpaths: thisisnotstandalonerelocatabledataarchive. Snapshotbeforefinalanalysis; rerunwithnewlabelatfinaldelivery, neveroverwriteexistinglabel. Noactiveunifiedtoolsessions.

## Seventh/eighth conditional complete (~03:29)

Now8/18complete: addedz.25K.02_interictal (oldGPU2806153normalexit) andz.75K8_interictal(oldCPU2964940normalexit). LowZlowKinter tailallE476.212/coreA476.696/coreB476.567Hz, globalhighfraction1,quiet0,brief0; allfielduniformhigh. Nativecounterfactual dZall−.05/s, dK+70.5155/s, recoveryeligibility~0. PrescribedlowK blockswould-beaccumulation: donotcallthisfullautonomousattractor. Otherhistorysamestatepointstillrunning. CentralZK8inter quiet(allrates0), dZall+.05, dK−1.599984/s; exacttailregional/spatial/contactmeansmatchhighhistory,notcertifiedwholeengineconvergence. ComparelowZK8active dK−15.9984 vsquiet−1.599984consistentwith0.5s/5s retentionrule, onlycounterfactualdirection.

Bothnewfull30ratecontexts/fixed300msnativefield/raster/contactviewsinspected; agent_visual_review.json stored. All8futureinputrecords pairedexact. Conditionalmap8/18PNG+PDFreviewed, metadataPASS_8_OF_18; humanPENDING. Scientificreviewupdated. Structuralconditionalmapnow5/24currentcases only; new16stillwaiting. Native rotated~114s/120lastchecked, fullrun notyetcomplete; isotropicnotstarted. Noactiveunifiedtoolhandles. Newslotjobsz.95K2highPID2982900localityCPU, nextmanagerdispatched.95K2inter(checklivePID). Oldprocesssection6/18isearlier; refreshstatusbeforeacting.

## Rate closure gate: failed this round

Simple effective-input/timecompressiong012/12,g>019/36. Frozen oneattempt staticcalibration1024Sobolx512train,heldout256x1024:241/256strict,254/256broad; two actual13–14Hz conditionspredicted29–39Hz,g6.39/9.61. STATIC_VALIDATION_FAIL. No validationrefit/thresholdrelaxation. See conductance_response/static_calibration_v1 and report_topic4_loop_conductance_gate.py. Transient/nativecorrespondence notrun becauseprereqfailed;formalSN/Hopf/limitcycleNOTESTABLISHED. Contract permitsnativeconditional/driftfallback. Do not force olderSN orrestartoldunrelatedkineticdensityproject. SparsecheckpointgG+gKmax13.3023<32isnotfulltime/jointµσcoverage.

## Axis controls and limitations

Validgraphs axis_controls/angular_reassignment/{rotated,isotropic}; STATIC_CONTROL_PASS. Hungarian sourcepartnerreassignment within target xsourceobserverregion xexactdelay preservesincomingdegree/fullweightmultiset andallnonEE/thresholds/noise. Outgoingdegree/sourceidentity changed. Originalcentralangle148.41deg/AR1.936;rot56.75deg/AR1.577(88.3degrotation,18.5percentweakeranisotropy);isoAR1.028. Failedweight-onlygraphs rejected/preserved, DONOTRUN.

Additionalaudit outgoing_and_excitability_audit.json:781loweredEoutgoingmean4415.3→4170.94rot(−5.5%)→3810.18iso(−13.7%); amongloweredtargetsloweredsourcefraction30.02%→29.21%→27.84%. These are compositegeometry controls, notpureorientationcausaltests. Actualsource substrate from topic4_historical_manual_z_common.OUT/substrate.npz (notbase.old.OUT).

Native2x120s samecoldstartseed9108405/fixedlawunclamped. Rotatedcurrentcompletedprefix60s:0operationalentries/exits, censored/runinprogress. Fullsource paired120s4entries3exits2returns. Earlyrotatedcores still~400Hzbursts, butno jointquietgaps;0.05–7.43s becomesone mergedepisode. Fixeddiagnostic.5–6s beforeG/Kactivation: original25shortevents/50.7%quiet,rot0short/0%quiet;meanE19.46vs46.23;Zeligiblefraction.7543vs.5749;native dZ−.02848vs−.04956/s,endpointslopesagree. This explains fasterZdepletion withresidualactivity, notlossofcorebursts. Cannotisolateorientationfromsourceoutdegreeconfounds. Firstfeedbackoriginal9.8736,rot6.4821. PNG/PDFprefixreviewdonepreviously. Fixedwindowkey fixed_0p5_6s; checkisotropicbefore_added_feedback ratherthanassuming.

Structural16 contract: fourZKpoints(.25,.02),(.95,.02),(.75,2),(.75,8)x2historiesx2graphs, reuse8originalcases. Graphswitchonly atbranchstart; preservepopulatedolddelayrings(past35.8ms)/currents/V/ref/G/M/RNG, sameZKtemplate/futureinput. Tail10s after20stransient. Preparedcompleteinitialstates/graphmetadataQA passed. Fixed30s, noautonomouscredit, nocompleteboundaries. OriginalCPUreferencerouteQA in axis_controls/conditional_runs/reference/route_qa.json, GPUreferencegate above.

## Automated review and next actions

review_topic4_loop_native_states.py: completechunkcount/fieldconservation, core/surroundrates, native5msfields,fixed80raster,15contactcurrentreadout. Autonomousinitial.5–min8/onset;postexitafterbothcores reachsharedZref. Firstchronologicalcompletebrief only, nevercherrypick. Ifnone, fixed300ms diagnostic(first300msinitial,last300mselse), explicitlynotanevent. Source initial36/31strongcore;fourreturns43/34,42/36,43/34,42/34. Postexit3first100.86weaknoncore shownhonestly. Directiondistributionnotclaimedrecovered. Lastpostexit7zeroevents butfixedquietview; quietE canstillhavecurrentbackground/Ispikes. Contactscalescommonwithinexample butdifferentbetweenexamples. OldreturnN aliases superseded/unlisted.

Otherproducers: analyze_topic4_loop_zk_conditional.py, analyze_topic4_loop_axis_native.py, analyze_topic4_loop_axis_conditional.py; plot_topic4_loop_axis_responses.py gives24comparisonpoints(8reference+16new), only2completecurrently. Nointerpolation/fakebasins. Allauto afterdurableresults. FinalstructuralPNG/PDFneedsvisualreview whenactualresultsarrive. fitz absent; usepdftoppm forPDF. No moretests/managerrewritesneeded unless newfailures.

Next: verify liveworkers andactualnewresults, allowexistingjobs tofinish, reviewmap/nativeeventshapes/structuralcontrols, updatefinalscientificreview/candidatebundle. Do notinfer fullreturn from quiet,coreorigin fromcoreparticipation,formaldivision fromfinite-windowpoints,orcompletegridcoverage. Allrequired18+2+16stillpending,so goalnotcomplete. Do notstartunboundedfollowupscans.

Memory citation final: MEMORY.md:277-297|note=[Native correspondence required before bifurcation interpretation];rolloutIDs01a09eae-c163-7cf2-8f2d-f11d43bdeaaf and01a0add1-6bb8-78f1-8af3-98dc7b0724b2. Do noteditCodexmemory. Priorhandoffarchived in handoff_archive/ for executionhistory.
