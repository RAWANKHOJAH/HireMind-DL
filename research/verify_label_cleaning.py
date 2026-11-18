"""Strict source-of-truth audit of trained label-cleaning ablations."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
ROOT=Path("/tmp/model_runs")
AUDIT=Path("/tmp/label_audit")
arms=("original","pseudo","weight")
metrics={};frames={}
def recalc(a):
    fs=list(ROOT.rglob(f"metrics_{a}.json"));assert len(fs)==1,fs
    obj=json.loads(fs[0].read_text())
    preds=list(ROOT.rglob(f"val_predictions_{a}.csv"));assert len(preds)==1,preds
    wpaths=list(ROOT.rglob(f"weights_{a}.pt"));assert len(wpaths)==1,wpaths
    assert hashlib.sha256(wpaths[0].read_bytes()).hexdigest()==obj["checkpoint_sha256"]
    df=pd.read_csv(preds[0])
    assert len(df)==1162 and df.row_uid.is_unique and len(df.job_group.unique())==55
    assert set(df.label)=={0,1,2}
    ps=df[["p_no","p_potential","p_good"]].to_numpy()
    assert np.isfinite(ps).all() and np.allclose(ps.sum(1),1,atol=2e-5)
    y=df.label.to_numpy().astype(int);z=df.pred.to_numpy().astype(int)
    assert np.array_equal(ps.argmax(1),z)
    cm=confusion_matrix(y,z,labels=[0,1,2])
    classification=classification_report(y,z,labels=[0,1,2],output_dict=True,zero_division=0)
    ms={
        "accuracy":100*accuracy_score(y,z),
        "macro_f1":100*f1_score(y,z,average="macro",labels=[0,1,2],zero_division=0),
        "good_fit_recall":100*classification["2"]["recall"],
        "good_fit_precision":100*classification["2"]["precision"],
        "potential_fit_recall":100*classification["1"]["recall"],
        "no_fit_recall":100*classification["0"]["recall"],
        "confusion":cm.tolist(),"checkpoint_sha256":obj["checkpoint_sha256"],
        "best_step":obj["best_step"],"n_pseudo_changed":obj["experimental_pseudo_label_changes"],
        "n_downweighted":obj["downweighted"]}
    for k in ("accuracy","macro_f1","good_fit_recall","good_fit_precision","potential_fit_recall"):
        assert abs(ms[k]-obj[k])<1e-4,(a,k,ms[k],obj[k])
    assert ms["confusion"]==obj["confusion"]
    return df,ms

for a in arms:
    frames[a],metrics[a]=recalc(a)
for a in arms[1:]:
    assert frames[a].row_uid.tolist()==frames["original"].row_uid.tolist()
    assert frames[a].label.tolist()==frames["original"].label.tolist()
audit_path=AUDIT/"AUDIT_COUNTS_AND_PROTOCOL.json"
assert audit_path.is_file()
audit=json.loads(audit_path.read_text())
assert audit["train_rows_oof"]==4993 and audit["folds"]==5
assert metrics["pseudo"]["n_pseudo_changed"]==audit["strict_pseudo_change_count"]
orig=frames["original"]
gf=orig.label.to_numpy()==2
out={"evaluation":"FROZEN VALIDATION ONLY","seed":42,
     "all_4993_train_rows_have_oof_prediction":True,
     "verified_human_label_corrections":0,
     "strict_pseudo_changes":audit["strict_pseudo_change_count"],
     "cleanlab_suspected":audit["cleanlab_suspected_count"],
     "arms":metrics,"differences_vs_original_pp":{},"decisions":{}}
acceptance={"accuracy":1.0,"good_fit_recall":2.0,"macro_f1":-1.0,"good_fit_precision":-1.0,"potential_fit_recall":-3.0}
for a in arms[1:]:
    diffs={k:round(metrics[a][k]-metrics["original"][k],6) for k in acceptance}
    passes={k:bool((diffs[k]>=v)) for k,v in acceptance.items()}
    z=frames[a].pred.to_numpy();oz=orig.pred.to_numpy()
    out["differences_vs_original_pp"][a]=diffs
    out["decisions"][a]={
      "passes_each_gate":passes,
      "passes_joint_objective":all(passes.values()),
      "result":"SEED42_ONLY_PROMISING_NEEDS_MULTI_SEED_REPLICATION" if all(passes.values()) else "NO_EVIDENCE_OF_JOINT_IMPROVEMENT",
      "GF_was_missed_now_correct":int((gf&(oz!=2)&(z==2)).sum()),
      "GF_was_correct_now_missed":int((gf&(oz==2)&(z!=2)).sum())}
dst=Path("final_label_cleaning_report");dst.mkdir(exist_ok=True)
(dst/"FINAL_EXPERIMENT_DECISION.json").write_text(json.dumps(out,indent=2))
rows=[]
for a in arms:
    row={"arm":a}
    for k in ("accuracy","macro_f1","good_fit_recall","good_fit_precision","potential_fit_recall","no_fit_recall"):
        row[k]=metrics[a][k]
    row["pseudo_changes"]=metrics[a]["n_pseudo_changed"]
    row["downweighted"]=metrics[a]["n_downweighted"]
    rows.append(row)
pd.DataFrame(rows).to_csv(dst/"FULL_4993_LABEL_CLEANING_RESULTS.csv",index=False)
print(json.dumps(out,indent=2),flush=True)
