import pandas as pd, numpy as np, sys
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from itertools import combinations
d = pd.read_csv(sys.argv[1])
print("=== J distribution ===")
for (run), g in d.groupby("run"):
    J=g.J
    print(f"{run:5s} n={len(g):4d} meanJ={J.mean():.3f} q10={J.quantile(.1):.3f} q25={J.quantile(.25):.3f} med={J.median():.3f} J<0.5={100*(J<.5).mean():.1f}% J<0.7={100*(J<.7).mean():.1f}%")
for (run,r), g in d.groupby(["run","reason"]):
    J=g.J; print(f"  {run}/{r:18s} n={len(g):4d} meanJ={J.mean():.3f} med={J.median():.3f} J<0.5={100*(J<.5).mean():.1f}% J<0.7={100*(J<.7).mean():.1f}%")
# concentration by video
t = d[d.run.isin(["E12","E14"])]
bad = t[t.J<.7]; print("bad frames E12+E14:", len(bad), "videos:", bad.video.nunique(), "top5 videos share:", round(bad.video.value_counts().head(5).sum()/len(bad),2))
print(bad.video.value_counts().head(6).to_dict())
sig = ["logit","pred_iou","area_ratio","log_area_dev","stab1","stab2","stab3","stab_min","cut_dist","since_cut","hsv_dist","div_maxcos"]
print("=== AUC (good=J>=0.7), raw (>0.5 => higher value = better) ===")
for name, g in [("E12+E14", t), ("E12", d[d.run=="E12"]), ("E14", d[d.run=="E14"]), ("E0r+E10a", d[d.run.isin(["E0r","E10a"])])]:
    y=(g.J>=.7).astype(int); s=[]
    for c in sig:
        m=g[c].notna()&np.isfinite(g[c])
        if m.sum()<20 or y[m].nunique()<2: continue
        s.append(f"{c}={roc_auc_score(y[m],g[c][m]):.3f}(n{m.sum()})")
    print(name, f"good={y.sum()} bad={(1-y).sum()}"); print("   "," ".join(s))
# combos: logistic, grouped CV by video, on E12+E14
y=(t.J>=.7).astype(int).values
feat = {"logit":t.logit,"pred_iou":t.pred_iou,"log_area_dev":t.log_area_dev.fillna(0),"stab1":t.stab1.fillna(0),"stab3":t.stab3.fillna(0),"stab_min":t.stab_min.fillna(0),"cut_dist":np.log1p(t.cut_dist.clip(upper=200)),"since_cut":np.log1p(t.since_cut)}
X=pd.DataFrame(feat); X["logit"]=X.logit.clip(-10,15)
gkf=GroupKFold(5)
def cv(cols):
    p=np.zeros(len(y))
    for tr,te in gkf.split(X,y,t.video):
        mdl=LogisticRegression(max_iter=1000).fit(X.iloc[tr][cols],y[tr]); p[te]=mdl.predict_proba(X.iloc[te][cols])[:,1]
    return roc_auc_score(y,p)
print("=== combos (logistic, 5-fold CV by video, E12+E14) ===")
res=[]
for k in (1,2,3):
    for c in combinations(X.columns,k): res.append((cv(list(c)),c))
for a,c in sorted(res,reverse=True)[:12]: print(f"{a:.3f}",c)
for c in [("logit",),("stab1",),("pred_iou",)]: print("single",c,f"{cv(list(c)):.3f}")
