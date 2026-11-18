"""Aggregate OOF predictions for ALL 4993 TRAIN labels and rank possible issues.

None of the output is a verified error. No automatic "corrected ground truth"
exists here. The STRICT_PSEUDO column is an experimental train-only treatment
based on specified rules, and MUST NOT be imported as human-approved labels.
Human-reviewed, independent corrections remain a separate task.
"""
import json,os
from pathlib import Path
import numpy as np,pandas as pd
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from collections import Counter
ROOT=Path("/tmp/oof_runs")
files=list(ROOT.rglob("oof_fold_*.csv"))
assert len(files)==5,files
frames=[]
for f in files:
    df=pd.read_csv(f)
    assert len(df)>0 and "fold" in df and df.fold.nunique()==1
    expected=int(f.stem.rsplit("_",1)[1])
    assert int(df.fold.iloc[0])==expected
    frames.append(df)
df=pd.concat(frames,ignore_index=True)
assert len(df)==4993 and df.row_uid.is_unique and set(df.fold)==set(range(5))
# Frozen split is an independent trusted index; no raw text required.
manifest=pd.read_csv("/tmp/HireMind-DL/splits/splits.csv")
m=manifest[manifest.split=="train"].copy()
assert len(m)==4993
d=m[["row_uid","label","label_id","cv_id","job_id","job_group"]].merge(df,on="row_uid",validate="one_to_one",suffixes=("_frozen","_oof"),sort=False)
assert len(d)==4993
for c in ["label","label_id","cv_id","job_id","job_group"]:
    assert d[f"{c}_frozen"].astype(str).tolist()==d[f"{c}_oof"].astype(str).tolist(),c
assert set(d.row_uid)==set(m.row_uid)
d=d.rename(columns={f"{c}_frozen":c for c in ["label","label_id","cv_id","job_id","job_group"]})
d=d.drop(columns=[f"{c}_oof" for c in ["label","label_id","cv_id","job_id","job_group"]])
y=d.label_id.to_numpy().astype(int)
P=d[["p_no","p_potential","p_good"]].to_numpy(dtype=float)
assert np.isfinite(P).all() and np.allclose(P.sum(1),1,atol=2e-5)
pmax=P.max(1);pred=P.argmax(1)
selfconf=P[np.arange(len(y)),y]
alt=np.where(P==pmax[:,None],-np.inf,P).max(1)
margin=pmax-selfconf
issues=find_label_issues(labels=y,pred_probs=P,filter_by="prune_by_noise_rate",verbose=False)
assert np.shape(issues)==(4993,)
scores=get_label_quality_scores(labels=y,pred_probs=P,method="self_confidence")
assert np.shape(scores)==(4993,)
d["oof_pred"]=pred
d["oof_max_probability"]=np.round(pmax,7)
d["oof_prob_given_label"]=np.round(selfconf,7)
d["oof_margin_over_given"]=np.round(margin,7)
d["label_quality_score"]=np.round(scores,7)
d["cleanlab_suspected"]=issues
# Strict experimental pseudo relabel: 2 pieces of independent evidence
# (OOF disagreement and cleanlab flag). These are NOT human truths.
candidate=(issues&(pred!=y)&(pmax>=0.85)&(selfconf<=0.10)&(margin>=0.65))
# Extra protection against extreme ordinal flips (Good <-> No Fit)
extreme=(abs(pred-y)==2)
candidate&=(~extreme)|(pmax>=0.97)
# Cap 2% of full training, and 2% of each original label; never depend on val.
selected=np.zeros(len(y),bool)
rank=sorted(np.where(candidate)[0],key=lambda i:(-margin[i],-pmax[i],int(i)))
caps={cls:max(1,int((y==cls).sum()*0.02)) for cls in range(3)}
counts=Counter()
for i in rank:
    orig=int(y[i])
    if sum(counts.values())>=99:break
    if counts[orig]>=caps[orig]:continue
    selected[i]=True;counts[orig]+=1
d["strict_pseudo_candidate"]=selected
d["strict_pseudo_label_id"]=np.where(selected,pred,y)
d["review_status"]="NOT_HUMAN_REVIEWED"
d["verified_label_error"]=False
d["source_method"]="5fold_group_heldout_oof+cleanlab_screening"
assert int((d.strict_pseudo_label_id.to_numpy()!=y).sum())==int(selected.sum())
assert set(d.loc[selected,"row_uid"]).issubset(set(m.row_uid))
dest=Path("audit_4993");dest.mkdir(exist_ok=True)
# Preserve frozen manifest order and no text; all 4993 records.
d.to_csv(dest/"ALL_4993_OOF_LABEL_AUDIT.csv",index=False)
changed=d.loc[selected,["row_uid","label","label_id","strict_pseudo_label_id","oof_pred","oof_max_probability",
                         "oof_prob_given_label","label_quality_score","fold","job_group"]].copy()
changed["label_status"]="UNVERIFIED_EXPERIMENTAL_PSEUDO_LABEL_NOT_APPROVED"
changed.to_csv(dest/"STRICT_PSEUDO_LABEL_OVERLAY_EXPERIMENT_ONLY.csv",index=False)
review=d.sort_values(["cleanlab_suspected","label_quality_score"],ascending=[False,True]).head(400).copy()
review["reviewer_1_label"]=""
review["reviewer_2_label"]=""
review["independent_evidence_notes"]=""
review["adjudication"]="PENDING"
review.to_csv(dest/"TOP_400_FOR_BLIND_HUMAN_REVIEW_NOT_APPROVED.csv",index=False)
# Conservative downweight: rank cleanlab-suspected examples by lower confidence;
# max 5% per class, no hard label rewrite. The sample-weight arm uses 0.25.
weight=np.ones(len(d),np.float32)
for cls in range(3):
    candidates=np.where((y==cls)&issues)[0]
    candidates=sorted(candidates,key=lambda i:(selfconf[i],i))
    n=min(len(candidates),int((y==cls).sum()*.05))
    weight[candidates[:n]]=.25
d["experimental_train_weight"]=weight
d[["row_uid","experimental_train_weight"]].to_csv(dest/"EXPERIMENTAL_DOWNWEIGHT.csv",index=False)
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
meta={
 "train_rows_oof":len(d),"folds":5,
 "unique_job_groups":int(d.job_group.nunique()),
 "test_rows_used":0,"validation_rows_used_to_select_labels":0,
 "all_original_labels_preserved_as_frozen":True,
 "out_of_fold_accuracy_against_orig_pct":round(float(100*accuracy_score(y,pred)),4),
 "out_of_fold_macro_f1_against_orig_pct":round(float(100*f1_score(y,pred,average="macro",zero_division=0)),4),
 "out_of_fold_confusion_against_original":confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
 "cleanlab_suspected_count":int(issues.sum()),
 "strict_pseudo_change_count":int(selected.sum()),
 "strict_pseudo_transition_counts":dict(Counter(f"{int(a)}->{int(b)}" for a,b in zip(y[selected],pred[selected]))),
 "experimental_downweighted_count":int((weight<1).sum()),
 "caps_per_original_class":caps,
 "confidence_thresholds":{"p_pred_min":.85,"p_original_max":.1,"margin_min":.65,"extreme_flip_pred_min":.97},
 "corrections_human_approved":0,
 "warning":"Model-disagreed and Cleanlab-suspected labels are NOT verified errors. Pseudo overlay is for ablation only."
}
(dest/"AUDIT_COUNTS_AND_PROTOCOL.json").write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2),flush=True)
