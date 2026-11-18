"""Independently verify *actual* matched CPU predictions (never read test)."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import f1_score,confusion_matrix,precision_score,recall_score
ROOT=Path("/tmp/hr_runs")
ARM=("knowledge_v1","hr_evidence")
def get(arm):
    paths=list(ROOT.rglob(f"knowledge_run_{arm}_s42.json"))
    assert len(paths)==1,paths
    info=json.loads(paths[0].read_text())
    rows=list(ROOT.rglob(f"val_predictions_{arm}_s42.csv"))
    assert len(rows)==1,rows
    df=pd.read_csv(rows[0])
    weights=list(ROOT.rglob(f"M0_L512_{arm}_s42/weights.pt"))
    assert len(weights)==1,weights
    digest=hashlib.sha256(weights[0].read_bytes()).hexdigest()
    assert digest==info["weights_sha256"],f"weights SHA mismatch {arm}"
    for k in ("row_uid","label","pred","p_no","p_potential","p_good"):
        assert k in df, (k,df.columns)
    assert len(df)==1162 and df.row_uid.is_unique
    p=df[["p_no","p_potential","p_good"]].to_numpy(dtype=float)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1,atol=2e-5)
    assert np.array_equal(p.argmax(1),df.pred.to_numpy())
    y=df.label.to_numpy().astype(int)
    z=df.pred.to_numpy().astype(int)
    cm=confusion_matrix(y,z,labels=[0,1,2])
    acc=100*np.mean(y==z)
    f1=100*f1_score(y,z,average="macro",labels=[0,1,2],zero_division=0)
    gr=100*recall_score(y,z,labels=[2],average="macro",zero_division=0)
    gp=100*precision_score(y,z,labels=[2],average="macro",zero_division=0)
    pr=100*recall_score(y,z,labels=[1],average="macro",zero_division=0)
    for k,v in [("accuracy",acc),("macro_f1",f1),("good_fit_recall",gr),("good_fit_precision",gp),("potential_fit_recall",pr)]:
        assert abs(info[k]-v)<1e-5,(arm,k,info[k],v)
    assert cm.tolist()==info["confusion"]
    return info,df,dict(accuracy=acc,macro_f1=f1,good_fit_recall=gr,good_fit_precision=gp,potential_fit_recall=pr,confusion=cm.tolist(),checkpoint_sha256=digest)
base,b,mb=get(ARM[0]);trt,t,mt=get(ARM[1])
assert b.row_uid.tolist()==t.row_uid.tolist()
assert b.label.tolist()==t.label.tolist()
y=b.label.to_numpy();bp=b.pred.to_numpy();tp=t.pred.to_numpy()
gain={k:round(mt[k]-mb[k],6) for k in ("accuracy","macro_f1","good_fit_recall","good_fit_precision","potential_fit_recall")}
good=y==2
passes={
 "accuracy_gain_ge_1pp":gain["accuracy"]>=1,
 "good_fit_recall_gain_ge_2pp":gain["good_fit_recall"]>=2,
 "macro_f1_drop_no_more_than_1pp":gain["macro_f1"]>=-1,
 "good_fit_precision_drop_no_more_than_1pp":gain["good_fit_precision"]>=-1
}
verdict="PROMISING_SINGLE_SEED_NEEDS_8_SEED_CONFIRMATION" if all(passes.values()) else "REJECT_AS_JOINT_OBJECTIVE_IMPROVEMENT"
out={"n_validation":int(len(y)),"seeds":[42],"evaluated_on":"validation only","data_is_same":True,
"control":mb,"treatment":mt,"changes_pp":gain,"acceptance":passes,"verdict":verdict,
"good_fit_transitions":{"previously_missed_now_correct":int(((bp!=2)&(tp==2)&good).sum()),
"previously_correct_now_missed":int(((bp==2)&(tp!=2)&good).sum()),"count":int(good.sum())},
"note":"Do not use final test set to tune or select a new model. Never claim Hams HR weights were available in public dataset."}
dest=Path("verified_comparison");dest.mkdir(exist_ok=True)
(dest/"results.json").write_text(json.dumps(out,indent=2))
pd.DataFrame([dict(arm="knowledge_v1",**{k:v for k,v in mb.items() if k not in ("confusion","checkpoint_sha256")}),dict(arm="hr_evidence",**{k:v for k,v in mt.items() if k not in ("confusion","checkpoint_sha256")})]).to_csv(dest/"comparison.csv",index=False)
print(json.dumps(out,indent=2))
