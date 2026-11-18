import pandas as pd, numpy as np, sys
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
d=pd.read_csv(sys.argv[1]); t=d[d.run.isin(["E12","E14"])].copy(); y=(t.J>=.7).values
print("pred_iou quantiles", t.pred_iou.quantile([.05,.1,.25,.5]).round(3).to_dict(), "logit q", t.logit.quantile([.05,.1,.25,.5]).round(2).to_dict())
# rank-average combos (no fitting)
def r(x): return rankdata(x.fillna(x.median()))
for cols in [["pred_iou","logit"],["pred_iou","logit","area_ratio"],["pred_iou","logit","stab3"],["pred_iou","area_ratio"]]:
    s=sum(r(t[c]) for c in cols); print("rank-avg",cols,round(roc_auc_score(y,s),3))
# within-video AUC (videos with both classes)
a=[]
for v,g in t.groupby("video"):
    yy=g.J>=.7
    if yy.nunique()==2: a.append((roc_auc_score(yy,g.pred_iou),len(g)))
print("pred_iou within-video AUC mean", round(np.mean([x for x,_ in a]),3), "n videos", len(a))
a=[]
for v,g in t.groupby("video"):
    yy=g.J>=.7
    if yy.nunique()==2: a.append(roc_auc_score(yy,g.logit))
print("logit within-video AUC mean", round(np.mean(a),3))
# gate grid
G,B=y.sum(),(~y).sum(); rows=[]
for pi in np.r_[0, np.round(np.arange(0.70,0.98,0.01),2)]:
    for lg in [-99,0,1,2,3,4,5,6,7,8]:
        keep=(t.pred_iou>=pi)&(t.logit>=lg)
        mg=(y&~keep).sum(); bb=(~y&~keep).sum()
        rows.append((pi,lg,mg,bb,keep.sum(), t.J[keep].mean()))
R=pd.DataFrame(rows,columns=["pi","lg","good_missed","bad_blocked","kept","meanJ_kept"])
R["score"]=R.bad_blocked-R.good_missed
print(f"good={G} bad={B}")
print(R.sort_values("score",ascending=False).head(8).to_string(index=False))
print("--- operating points: max bad_blocked with good_missed <= x ---")
for x in [22,45,90]:
    print(x, R[R.good_missed<=x].sort_values(["bad_blocked","good_missed"],ascending=[False,True]).head(2).to_string(index=False,header=False))
print("--- single pred_iou only ---")
print(R[R.lg==-99].iloc[::3].to_string(index=False))
