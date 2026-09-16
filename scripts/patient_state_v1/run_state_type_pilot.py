"""Whole-interval held-out OU inference and conditional seizure-type readout.

Each outer fold excludes the complete interval ending at one qualified seizure
from dynamics fitting, and its seizure label from readout fitting. Test marks
are predicted before assimilation. Future training intervals are allowed:
this is retrospective interval transfer, not prospective seizure forecasting.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import sys, json, time, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN, write_json
from scripts.patient_state_v1.model import fit, filter_adf
from scripts.patient_state_v1.round2 import ewma_fit_predict, ewma
from scripts.patient_state_v1.preseizure import state_at, covered

OUT = ROOT / "results/topic5_patient_state_inference/e1146_state_type_pilot_v2_20260910"
NODES, WEIGHTS = hermgauss(32)
WEIGHTS = WEIGHTS / np.sqrt(np.pi)
READOUTS = ("constant", "ou_state", "recent_memory", "recent_rate", "record_time", "ou_state_rate")
COLORS = {"constant":"#888888", "ou_state":"#6a3d9a", "recent_memory":"#008577", "recent_rate":"#c17c21", "record_time":"#999933", "ou_state_rate":"#2166ac"}


def inputs():
    data = dict(np.load(RUN / "observations.npz"))
    events = pd.read_csv(RUN / "events.csv")
    inv = json.loads((RUN / "seizures.json").read_text())
    wins = pd.read_csv(RUN / "frozen_seizure_windows.csv")
    wins = wins[wins.window.eq("pre15")].sort_values("sz").reset_index(drop=True)
    ex = pd.read_csv(RUN / "exposure.csv")
    assert len(events) == len(data["y"]) and np.all(data["n"] == 1)
    assert np.all(events.end_epoch.to_numpy()[:-1] <= events.start_epoch.to_numpy()[1:])
    for row in wins.itertuples():
        p = ROOT / f"results/topic5_preseizure_template_association_broadband/cohort_20260909/per_subject/epilepsiae_1146/per_seizure/seizure_{row.sz-1:03d}.json"
        label = json.loads(p.read_text())
        assert label["qualified_source_label"] == row.label
        assert label["seizure_id"] == inv[row.sz-1]["seizure_id"]
    return data, events, inv, wins, ex


def select_training(data, mask):
    d = {k: np.array(v[mask], copy=True) for k, v in data.items() if np.ndim(v) > 0}
    d["reset"] = np.r_[True, np.diff(d["epoch"]) != 0]
    d["dt"] = np.r_[0., np.diff(d["t"])]
    d["dt"][d["reset"]] = 0
    assert np.all(d["dt"][~d["reset"]] > 0)
    return d


def exposure_hours(lo, hi, ex):
    return np.maximum(0., np.minimum(hi, ex.end_epoch.to_numpy()) - np.maximum(lo, ex.start_epoch.to_numpy())).sum() / 3600


def memory_queries(times, data, events, theta, inv):
    tau, strength, p0 = theta["tau_hours"], theta["strength"], theta["p0"]
    aa = np.empty(len(events)); bb = np.empty(len(events))
    a = b = 0.
    for i in range(len(events)):
        if data["reset"][i]:
            a = b = 0.
        else:
            decay = np.exp(-data["dt"][i] / tau)
            a *= decay; b *= decay
        a += data["y"][i]; b += 1-data["y"][i]
        aa[i], bb[i] = a, b
    idx = np.searchsorted(events.end_epoch.to_numpy(), times, side="right") - 1
    epochs = np.searchsorted([r["offset"] for r in inv], times, side="right")
    out = np.full(len(times), p0)
    valid = idx >= 0
    valid[valid] &= data["epoch"][idx[valid]] == epochs[valid]
    ix = idx[valid]
    decay = np.exp(-(times[valid]-events.start_epoch.to_numpy()[ix])/3600/tau)
    out[valid] = (aa[ix]*decay+strength*p0)/((aa[ix]+bb[ix])*decay+strength)
    return out


def feature_queries(times, data, events, filt, theta, memory, inv, ex):
    m, v, p, _ = state_at(np.asarray(times), theta, data, events, filt, inv)
    mem = memory_queries(np.asarray(times), data, events, memory, inv)
    rows = []
    for t, mu, var, pp, pm in zip(times, m, v, p, mem):
        epoch = np.searchsorted([r["offset"] for r in inv], t, side="right")
        start = inv[epoch-1]["offset"] if epoch > 0 else float(data["origin_epoch"])
        lo = max(start, t-900)
        count = int(((events.start_epoch >= lo) & (events.end_epoch <= t)).sum())
        hours = exposure_hours(lo, t, ex)
        history = int(((events.start_epoch >= start) & (events.end_epoch <= t)).sum())
        rows.append(dict(time=t, state_mean=mu, state_var=var, p_ied_tb=pp,
                         memory_p_tb=pm, memory_state=logit(pm)-logit(memory["p0"]),
                         log_rate=np.log1p(count/hours) if hours > 0 else np.nan,
                         rate=count/hours if hours > 0 else np.nan,
                         record_hours=(t-float(data["origin_epoch"]))/3600,
                         recent_events=count, recent_observed_hours=hours,
                         interval_history_events=history))
    return pd.DataFrame(rows)


def design(features, model, center=None):
    n = len(features)
    arrays = []
    meta = dict(center or {})
    if model in ("ou_state", "ou_state_rate"):
        # The readout intercept absorbs b; residual x=s-b retains fixed log-odds units.
        arrays.append(features.state_mean.to_numpy()[:,None] + np.sqrt(2*features.state_var.to_numpy())[:,None]*NODES)
    elif model == "recent_memory":
        arrays.append(np.repeat(features.memory_state.to_numpy()[:,None], len(NODES), axis=1))
    if model in ("recent_rate", "ou_state_rate"):
        x = features.log_rate.to_numpy()
        if center is None:
            meta["rate_center"] = float(np.nanmean(x)); meta["rate_scale"] = max(float(np.nanstd(x)), 1.)
        x = (np.nan_to_num(x, nan=meta["rate_center"])-meta["rate_center"])/meta["rate_scale"]
        arrays.append(np.repeat(x[:,None], len(NODES), axis=1))
    if model == "record_time":
        x = features.record_hours.to_numpy()/24
        if center is None: meta["time_center"] = float(x.mean())
        arrays.append(np.repeat((x-meta["time_center"])[:,None], len(NODES), axis=1))
    return np.stack([np.ones((n,len(NODES))), *arrays], axis=2), meta


def readout_fit(features, labels, model, prior_scale=2.):
    X, meta = design(features, model)
    labels = np.asarray(labels, float)
    if model == "constant":
        p = (labels.sum()+.5)/(len(labels)+1)
        return dict(theta=[float(logit(p))], center=meta, success=True, constant=p)
    # Fixed weak intercept pseudo-counts and Gaussian slopes, no tuning on test labels.
    def objective(beta):
        node_p = expit(np.einsum("nqd,d->nq", X, beta))
        p = np.clip(node_p@WEIGHTS, 1e-12, 1-1e-12)
        loss = -np.sum(labels*np.log(p)+(1-labels)*np.log1p(-p))
        gradp = np.einsum("nq,nqd,q->nd", node_p*(1-node_p), X, WEIGHTS)
        grad = np.sum(((p-labels)/(p*(1-p)))[:,None]*gradp, axis=0)
        loss += .5*np.logaddexp(0,-beta[0])+.5*np.logaddexp(0,beta[0]) + .5*np.sum((beta[1:]/prior_scale)**2)
        grad[0] += expit(beta[0])-.5
        grad[1:] += beta[1:]/prior_scale**2
        return loss, grad
    start = np.r_[logit((labels.sum()+.5)/(len(labels)+1)), np.zeros(X.shape[2]-1)]
    bounds = [(-12.,12.)] + [((-8.,8.) if model == "record_time" else (0.,8.))]*(len(start)-1)
    result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds, options={"ftol":1e-12,"gtol":1e-7,"maxiter":500})
    return dict(theta=result.x.tolist(), center=meta, success=bool(result.success), objective=float(result.fun),
                prior_scale=prior_scale, message=str(result.message))


def readout_predict(features, model, fitted):
    X, _ = design(features, model, fitted["center"])
    return expit(np.einsum("nqd,d->nq", X, np.asarray(fitted["theta"])))@WEIGHTS


def worker(sz):
    start_time = time.time()
    target = OUT / "folds" / f"sz{sz:02d}.json"
    if target.exists(): return json.loads(target.read_text())
    data, events, inv, wins, ex = inputs()
    test = data["epoch"] == sz-1
    train = select_training(data, ~test)
    assert not np.any(train["epoch"] == sz-1)
    baseline = float(train["y"].mean())
    fits = [fit(train, "ou", np.r_[logit(baseline), np.log(tau), np.log(.6)], maxiter=150) for tau in (.1, 1., 6.)]
    valid = [f for f in fits if f["success"]]
    if not valid: raise RuntimeError(f"SZ{sz}: all dynamics optimizations failed")
    best = max(valid, key=lambda f:f["loglik"])
    theta = np.asarray(best["theta"])
    filt = filter_adf(theta, data, order=32)
    _, memory = ewma_fit_predict(train, len(train["y"]))
    mem_pred = ewma(data["y"],data["n"],data["dt"],data["reset"],memory["tau_hours"],memory["strength"],memory["p0"])
    # Earliest EEG/clinical onset, before any current ictal observations.
    times = np.array([inv[int(i)-1]["onset"]-1e-6 for i in wins.sz])
    features = feature_queries(times, data, events, filt, theta, memory, inv, ex)
    features["sz"] = wins.sz.to_numpy(); features["label"] = wins.label.to_numpy()
    hold = features.sz.eq(sz).to_numpy()
    y = features.label.eq("TB").astype(int).to_numpy()
    assert hold.sum() == 1
    rows, readouts = [], {}
    for model in READOUTS:
        r = readout_fit(features.loc[~hold], y[~hold], model)
        assert r["success"], (sz,model,r)
        pp = float(readout_predict(features.loc[hold],model,r)[0])
        readouts[model] = r
        rows.append(dict(sz=sz,label=features.loc[hold,"label"].iloc[0],model=model,p_tb=pp,
                         n_train_ta=int((1-y[~hold]).sum()),n_train_tb=int(y[~hold].sum()),
                         log_score=float(y[hold][0]*np.log(pp)+(1-y[hold][0])*np.log1p(-pp))))
    # Sensitivity is reported separately, never used to choose the principal result.
    sensitivity = []
    for scale in (1.,4.):
        for model in ("ou_state", "recent_memory"):
            r = readout_fit(features.loc[~hold], y[~hold], model, scale)
            sensitivity.append(dict(sz=sz,model=model,prior_scale=scale,p_tb=float(readout_predict(features.loc[hold],model,r)[0]),success=r["success"]))
    ix = np.flatnonzero(test)
    marks = pd.DataFrame(dict(index=ix,sz=sz,time=events.start_epoch.to_numpy()[ix],y=data["y"][ix],
                              ou_p_tb=filt["predict_tb"][ix],memory_p_tb=mem_pred[ix],constant_p_tb=baseline))
    # Query at real times: only completed events are assimilated by state_at.
    prev = inv[sz-2]["offset"]
    ts = np.unique(np.r_[np.arange(max(prev,times[hold][0]-3600),times[hold][0],10.),times[hold][0]])
    trace = feature_queries(ts, data, events, filt, theta, memory, inv, ex)
    trace["minutes"] = (trace.time-inv[sz-1]["onset"])/60
    trace["covered"] = covered(ts, ex)
    for model in READOUTS:
        trace["seizure_p_"+model] = readout_predict(trace, model, readouts[model])
    # Verify no current-label consumption in prediction and isolate clinical epochs.
    max_own = max_other = 0.
    if len(ix):
        mutated = dict(data); mutated["y"] = data["y"].copy()
        middle = int(ix[len(ix)//2]); mutated["y"][middle] = 1-mutated["y"][middle]
        counter = filter_adf(theta, mutated)
        max_own = float(np.max(abs(counter["predict_tb"][:middle+1]-filt["predict_tb"][:middle+1])))
        mutated["y"][ix] = 1-data["y"][ix]
        counter = filter_adf(theta, mutated)
        max_other = float(np.max(abs(counter["mean"][~test]-filt["mean"][~test])))
        assert max_own < 1e-12 and max_other < 1e-12
    features.to_csv(target.with_name(f"sz{sz:02d}_features.csv"),index=False)
    marks.to_csv(target.with_name(f"sz{sz:02d}_ied_predictions.csv.gz"),index=False)
    trace.to_csv(target.with_name(f"sz{sz:02d}_trace.csv"),index=False)
    result = dict(status="COMPLETE",sz=sz,elapsed_seconds=time.time()-start_time,
                  dynamics=best,all_dynamics_starts=fits,memory=memory,
                  n_train_events=len(train["y"]),n_test_events=int(test.sum()),
                  heldout_epoch=sz-1,train_seizure_ids=features.loc[~hold,"sz"].tolist(),
                  endpoint=features.loc[hold].to_dict("records")[0],
                  readouts=readouts,predictions=rows,prior_sensitivity=sensitivity,
                  causal_checks=dict(current_and_earlier_prediction_change=max_own,other_epoch_state_change=max_other))
    write_json(target,result)
    return result


def summarize():
    _,events,inv,wins,ex = inputs()
    folds = [json.loads((OUT/"folds"/f"sz{sz:02d}.json").read_text()) for sz in wins.sz]
    predictions = pd.DataFrame([r for f in folds for r in f["predictions"]])
    predictions.to_csv(OUT/"seizure_predictions.csv",index=False)
    sensitivity = pd.DataFrame([r for f in folds for r in f["prior_sensitivity"]])
    sensitivity.to_csv(OUT/"prior_sensitivity.csv",index=False)
    endpoints = pd.DataFrame([f["endpoint"] for f in folds])
    endpoints.to_csv(OUT/"endpoint_states.csv",index=False)
    scores=[]
    for model,g in predictions.groupby("model",sort=False):
        y=g.label.eq("TB").astype(int).to_numpy();p=g.p_tb.to_numpy();guess=p>=.5
        scores.append(dict(model=model,n=len(g),n_tb=int(y.sum()),mean_log_score=g.log_score.mean(),brier=np.mean((p-y)**2),
                           pooled_auc_diagnostic=roc_auc_score(y,p),balanced_accuracy=balanced_accuracy_score(y,guess),
                           correct_ta=int(np.sum((y==0)&~guess)),correct_tb=int(np.sum((y==1)&guess)),
                           tb_probabilities=g.loc[g.label.eq("TB"),"p_tb"].tolist()))
    write_json(OUT/"seizure_scores.json",scores)
    marks=pd.concat([pd.read_csv(OUT/"folds"/f"sz{sz:02d}_ied_predictions.csv.gz") for sz in wins.sz],ignore_index=True)
    assert not marks["index"].duplicated().any()
    marks.to_csv(OUT/"ied_predictions.csv.gz",index=False)
    ms=[]
    for model,column in [("ou","ou_p_tb"),("recent_memory","memory_p_tb"),("constant","constant_p_tb")]:
        p=marks[column].to_numpy();y=marks.y.to_numpy();ll=y*np.log(p)+(1-y)*np.log1p(-p)
        ms.append(dict(model=model,n_events=len(marks),mean_log_score=float(ll.mean()),brier=float(np.mean((p-y)**2))))
    pd.DataFrame(ms).to_csv(OUT/"ied_scores.csv",index=False)
    # Case-level paired scores; no IED pseudo-replication for seizure evidence.
    pivot=predictions.pivot(index="sz",columns="model",values="log_score")
    rng=np.random.default_rng(20260910);unc=[]
    for baseline in ("constant","recent_memory","recent_rate","record_time"):
        delta=(pivot.ou_state-pivot[baseline]).to_numpy()
        boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
        unc.append(dict(baseline=baseline,mean_gain=delta.mean(),lower=np.quantile(boot,.025),upper=np.quantile(boot,.975),
                        note="Exploratory case resampling; shared CV fits and temporal dependence not resolved"))
    write_json(OUT/"seizure_paired_uncertainty.json",unc)
    plot_results(predictions,endpoints,scores,marks)
    write_json(OUT/"status.json",dict(status="COMPLETE",n_folds=len(folds),n_ied_predictions=len(marks),
             n_seizures=len(wins),n_tb=2,retrospective_interval_transfer=True,templates="fixed development record",
             parameters_exclude_target_interval=True,target_type_excluded_from_readout=True,
             test_marks_used_only_after_prediction=True,earliest_eeg_clinical_onset=True,
             readout_slope_prior_sd=2,readout_direction="positive TB association, fixed before running",
             no_ied_reset=True,cross_seizure_boundary="independent stationary prior; statistical convention",
             initial_state_uncertainty="Gaussian ADF, dynamics point estimate; no parameter uncertainty",
             empty_test_intervals=endpoints.loc[endpoints.interval_history_events.eq(0),"sz"].tolist(),
             fits_success=all(f["dynamics"]["success"] for f in folds),
             pooled_auc_caution="LOO training type priors differ between TA and TB targets; pooled AUC is not a primary separation metric",
             human_visual_review="pending"))
    print(pd.DataFrame(scores).to_string(index=False),flush=True)
    print(pd.DataFrame(ms).to_string(index=False),flush=True)


def plot_results(predictions,endpoints,scores,marks):
    fp=OUT/"figures";fp.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={"width_ratios":[1.7,1]})
    order=endpoints.sz.tolist();xx=np.arange(len(order))
    for model,offset in [("constant",-.18),("recent_memory",0),("ou_state",.18)]:
        g=predictions[predictions.model.eq(model)].set_index("sz").loc[order]
        axes[0].scatter(xx+offset,g.p_tb,color=COLORS[model],s=40,label=model)
    axes[0].set_xticks(xx,[f"SZ{r.sz}\n{r.label}" for r in endpoints.itertuples()])
    for tick,label in zip(axes[0].get_xticklabels(),endpoints.label):tick.set_color("#2166ac" if label=="TB" else "#b2182b")
    axes[0].axhline(.5,c="gray",lw=.7,ls="--");axes[0].set_ylim(0,1)
    axes[0].set_ylabel("P(TB-type seizure | seizure occurs now)")
    axes[0].set_title("Every held-out seizure; updates use preceding IEDs only")
    axes[0].legend(fontsize=9)
    s=pd.DataFrame(scores).set_index("model")
    for i,model in enumerate(READOUTS):axes[1].barh(i,s.loc[model,"mean_log_score"]-s.loc["constant","mean_log_score"],color=COLORS[model])
    axes[1].set_yticks(np.arange(len(READOUTS)),READOUTS);axes[1].axvline(0,c="gray",lw=.7)
    axes[1].set_xlabel("Mean log-score gain over type baseline")
    axes[1].set_title("12 seizures: 10 TA, 2 TB")
    fig.suptitle("E1146 | Whole-interval holdout, shared OU state and seizure-type readout",fontsize=13)
    fig.tight_layout(rect=(0,.04,1,.93));fig.text(.02,.01,"Retrospective transfer; fixed development labels. SZ11/SZ13 have no IEDs in their held-out interval. No seizure-time prediction.",fontsize=9)
    save(fig,fp/"heldout_seizure_types")
    fig,axs=plt.subplots(3,2,figsize=(13,9),sharex="col")
    for col,sz in enumerate((19,22)):
        trace=pd.read_csv(OUT/"folds"/f"sz{sz:02d}_trace.csv")
        onset=inv_time(sz);window=marks[marks.sz.eq(sz)&marks.time.ge(onset-3600)]
        for y,color,label in [(0,"#b2182b","TA"),(1,"#2166ac","TB")]:
            part=window[window.y.eq(y)]
            axs[0,col].scatter((part.time-onset)/60,np.full(len(part),y),marker="|",s=22,color=color)
        axs[0,col].set_yticks([0,1],["TA","TB"]);axs[0,col].set_ylim(-.5,1.5)
        axs[0,col].set_title(f"SZ{sz} | actual seizure: TB")
        for column,label,color in [("ou_p_tb","OU: before current label",COLORS["ou_state"]),("memory_p_tb","Recent memory",COLORS["recent_memory"])]:
            axs[1,col].plot((window.time-onset)/60,window[column],".",ms=2.5,alpha=.6,color=color,label=label)
        for model in ("ou_state","recent_memory","constant"):
            pp=trace["seizure_p_"+model].copy();pp[~trace.covered]=np.nan
            axs[2,col].plot(trace.minutes,pp,color=COLORS[model],label=model,lw=1.5)
        q=predictions[predictions.sz.eq(sz)&predictions.model.eq("ou_state")].p_tb.iloc[0]
        axs[2,col].scatter([0],[q],c=COLORS["ou_state"],s=50,zorder=5)
        axs[2,col].annotate(f"At onset: {q:.3f}",(0,q),xytext=(-6,12),textcoords="offset points",ha="right")
        for row in (1,2):axs[row,col].set_ylim(0,1)
        for ax in axs[:,col]:ax.set_xlim(-60,0);ax.axvline(-15,c="gray",ls=":",lw=.7)
        axs[2,col].set_xlabel("Minutes before earliest EEG / clinical onset")
    axs[0,0].set_ylabel("Observed IED labels")
    axs[1,0].set_ylabel("P(TB IED | current event)")
    axs[2,0].set_ylabel("P(TB seizure | occurs now)")
    axs[1,0].legend(fontsize=8);axs[2,0].legend(fontsize=8)
    fig.suptitle("Two held-out TB seizures | Online state updates and two distinct type probabilities",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96));save(fig,fp/"tb_cases_online_predictions")
    fig,axs=plt.subplots(3,4,figsize=(14,9),sharex=True,sharey=True)
    for ax,row in zip(axs.flat,endpoints.itertuples()):
        trace=pd.read_csv(OUT/"folds"/f"sz{row.sz:02d}_trace.csv")
        mean=trace.state_mean.copy();mean[~trace.covered]=np.nan;sd=np.sqrt(trace.state_var)
        ax.plot(trace.minutes,mean,c=COLORS["ou_state"])
        ax.fill_between(trace.minutes,mean-1.96*sd,mean+1.96*sd,color=COLORS["ou_state"],alpha=.15)
        ax.axhline(0,c="gray",lw=.7);ax.set_title(f"SZ{row.sz}: {row.label} | IED n={row.interval_history_events}")
        ax.set_xlim(-60,0)
    for ax in axs[-1]:ax.set_xlabel("Minutes before onset")
    for ax in axs[:,0]:ax.set_ylabel("Residual state x = s - b")
    fig.suptitle("All held-out intervals: causal state reconstruction with fixed dynamics per fold\n95% Gaussian state intervals; dynamics and readout parameter uncertainty not included",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,fp/"all_heldout_states")
    (fp/"README.md").write_text(
        "### heldout_seizure_types.png / heldout_seizure_types.pdf\n\n完整留出每次发作之前的间期段，并排除该发作标签后预测类型。左图逐例显示概率，右图比较条件类型评分。\n\n**关注点**：TB 仅两例；区分类型概率、排序与 0.5 阈值判断，当前为固定开发标签下的回顾性迁移。\n\n"
        "### tb_cases_online_predictions.png / tb_cases_online_predictions.pdf\n\n两次 TB 发作分别展示真实 IED 标签、先预测后更新的 IED 概率，以及同一状态的条件发作类型概率。灰色竖线只是 −15 min 参考，不冻结后续更新。\n\n**关注点**：模型可持续吸收已经完成的真实事件，不能使用当前未揭盲标签或起始后的发作波形更新起始前状态。\n\n"
        "### all_heldout_states.png / all_heldout_states.pdf\n\n12 个留出间期的顺序状态重建及高斯 95% 状态范围。没有 IED 的 SZ11/SZ13 保留先验，缺覆盖位置断线。\n\n**关注点**：不同折的动力学由各自训练集得到，阴影不包含参数不确定性；待用户目视检查。\n",encoding="utf-8")


def inv_time(sz):
    return json.loads((RUN/"seizures.json").read_text())[sz-1]["onset"]


def save(fig,path):
    for ext in ("png","pdf"):fig.savefig(path.with_suffix("."+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=6);ap.add_argument("--summarize-only",action="store_true")
    args=ap.parse_args();(OUT/"folds").mkdir(parents=True,exist_ok=True)
    if not args.summarize_only:
        _,_,_,wins,_=inputs()
        write_json(OUT/"status.json",dict(status="RUNNING",n_folds=len(wins),started_unix=time.time(),workers=args.workers))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(worker,int(sz)) for sz in wins.sz]):
                r=f.result();print(json.dumps(dict(sz=r["sz"],status=r["status"],seconds=r["elapsed_seconds"],n_ied=r["n_test_events"],predictions=r["predictions"])),flush=True)
    summarize()
