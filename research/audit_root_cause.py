"""Privacy-preserving post-hoc failure diagnosis, validation only. No labels changed."""
import os,sys,json,hashlib,collections,random,re
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,"/tmp/Senior-2/evaluation/knowledge")
import knowledge as orig
import knowledge_quality_overlay as overlay
sys.path.insert(0,"/tmp/HireMind-DL/src")
from hiremind_dl import load_config
from hiremind_dl.data import download_raw, read_raw, split_frame, load_manifest
OUT=ROOT/"root_audit_results";OUT.mkdir(exist_ok=True)
BASE="google/bert_uncased_L-2_H-128_A-2"
REV="30b0a37ccaaa32f332884b96992754e246e48c5f"
WEIGHTS=Path("/tmp/final-results/final_model/M0_L512_final_s42/weights.pt")
EXPECTED="cdd6b38503d708ecb9acd8a4be387893367013675e445b1693c56f9a6ada4d8b"
sha=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
assert sha==EXPECTED,(sha,EXPECTED)
cfg=load_config("/tmp/HireMind-DL/configs/experiment.json")
raw={"train":read_raw(download_raw(cfg,"train"))}
man=load_manifest(cfg)
tr=split_frame(raw,man,"train",4993)
va=split_frame(raw,man,"validation",1162)
assert tr.job_group.nunique()>0 and va.job_group.nunique()>0
assert set(tr.job_group).isdisjoint(set(va.job_group))
ref=pd.read_csv("/tmp/final-results/test-results/val_predictions_final_s42.csv")
assert va.row_uid.tolist()==ref.row_uid.tolist()
assert va.label_id.tolist()==ref.label.tolist()
tok=AutoTokenizer.from_pretrained(BASE,revision=REV)
model=AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(BASE,revision=REV,num_labels=3),attn_implementation="eager")
model.load_state_dict(torch.load(WEIGHTS,map_location="cpu",weights_only=True),strict=True)
model.eval(); torch.set_num_threads(4)
def pair_input(cv,jd):
    return orig.knowledge_line(orig.features(cv,jd))+"\n"+jd
def infer(cvs, jds):
    probs=[]
    with torch.inference_mode():
        for i in range(0,len(cvs),24):
            enc=tok(cvs[i:i+24],jds[i:i+24],truncation="longest_first",max_length=512,padding="max_length",return_tensors="pt")
            probs.append(model(**enc).logits.softmax(-1).cpu().numpy())
    return np.concatenate(probs)
cvs=va.resume_text.tolist()
jobs=va.job_description_text.tolist()
aug=[pair_input(a,b) for a,b in zip(cvs,jobs)]
p=infer(cvs,aug)
pred=np.argmax(p,axis=1)
agreement=float(np.mean(pred==ref.pred.to_numpy()))
assert agreement==1.0,(agreement,int((pred!=ref.pred.to_numpy()).sum()))
job_groups=sorted(va.job_group.unique().tolist())
first_job={g:va.loc[va.job_group==g,"job_description_text"].iloc[0] for g in job_groups}
swap_map={g:job_groups[(i+1)%len(job_groups)] for i,g in enumerate(job_groups)}
swapped=[first_job[swap_map[g]] for g in va.job_group]
swap_aug=[pair_input(cv,jd) for cv,jd in zip(cvs,swapped)]
p_swap=infer(cvs,swap_aug)
pred_swap=np.argmax(p_swap,axis=1)
GF=(va.label_id.to_numpy()==2)
miss=GF & (pred!=2)
hits=GF &(pred==2)
def pct(flag):return round(float(np.mean(flag))*100,2) if len(flag)>0 else None
def summ(a):
    arr=np.asarray(a);return dict(n=int(len(arr)),median=round(float(np.median(arr)),2) if len(arr) else None,mean=round(float(np.mean(arr)),2) if len(arr) else None)
# Actual retained tokens and literal skill evidence after pair truncation
rows=[]
for i,(cv,jd,aj) in enumerate(zip(cvs,jobs,aug)):
    enc=tok(cv,aj,truncation="longest_first",max_length=512,padding=False)
    ids=np.array(enc["input_ids"],dtype=int); types=np.array(enc["token_type_ids"],dtype=int)
    keepcv=int((types==0).sum()-2);keepjd=int((types==1).sum()-1)
    fullcv=len(tok.encode(cv,add_special_tokens=False))
    fulljd=len(tok.encode(aj,add_special_tokens=False))
    vis_cv=tok.decode(ids[(types==0)][1:-1],skip_special_tokens=True)
    vis_jd=tok.decode(ids[(types==1)][:-1],skip_special_tokens=True)
    fs=orig.features(cv,jd)
    fv=overlay.features(cv,jd)
    orig_line=orig.knowledge_line(fs)
    revised_line=overlay.knowledge_line(fv)
    found_full=set(fs["found"])
    found_visible=found_full & orig.skills_in(vis_cv)
    rows.append(dict(row_uid=va.row_uid.iloc[i],job_group=va.job_group.iloc[i],label=int(va.label_id.iloc[i]),pred=int(pred[i]),
       full_cv_tokens=fullcv,full_job_plus_knowledge_tokens=fulljd,retained_cv_tokens=keepcv,retained_job_tokens=keepjd,
       retained_cv_fraction=keepcv/max(1,fullcv),retained_job_fraction=keepjd/max(1,fulljd),
       n_found_full=len(found_full),n_found_visible=len(found_visible),
       skill_evidence_lost_count=len(found_full-found_visible),
       knowledge_line_preserved=bool(re.search(r"skills\s+check\s*:",vis_jd[:120],re.I)),
       new_knowledge_line_differs=orig_line!=revised_line,
       old_n_job_skills=fs["n_job_skills"],new_n_job_skills=fv["n_job_skills"],
       old_n_found=fs["n_found"],new_n_found=fv["n_found"],
       skipped_required_words_within_last_half=bool(fulljd>keepjd),
       job_length_over_input_budget=bool(fullcv+fulljd+3>512)))
diag=pd.DataFrame(rows)
def group_sum(sub):
    return dict(n=int(len(sub)),cv_retained_median_pct=round(float(sub.retained_cv_fraction.median()*100),2),jd_retained_median_pct=round(float(sub.retained_job_fraction.median()*100),2),
        lost_any_full_matched_skill_pct=pct(sub.skill_evidence_lost_count>0),
        knowledge_prefix_preserved_pct=pct(sub.knowledge_line_preserved),
        line_changed_pct=pct(sub.new_knowledge_line_differs),
        no_full_matched_skill_pct=pct(sub.n_found_full==0),
        median_full_cv_tokens=float(sub.full_cv_tokens.median()),
        median_full_jd_tokens=float(sub.full_job_plus_knowledge_tokens.median()),
        median_retained_cv_tokens=float(sub.retained_cv_tokens.median()),
        median_retained_jd_tokens=float(sub.retained_job_tokens.median()))
groups=[]
for g,x in diag[diag.label==2].groupby("job_group"):
    groups.append(dict(job_group=g,n_good=int(len(x)),miss=int((x.pred!=2).sum()),tp=int((x.pred==2).sum()),
      any_skill_evidence_lost=int((x.skill_evidence_lost_count>0).sum()),
      changed_knowledge_lines=int(x.new_knowledge_line_differs.sum()),
      median_cv_frac=round(float(x.retained_cv_fraction.median()),3)))
pd.DataFrame(groups).sort_values("miss",ascending=False).to_csv(OUT/"good_fit_group_aggregate.csv",index=False)
# Training impact of overlay: no raw CV, only aggregate
changed=collections.Counter()
for v in tr.itertuples(index=False):
    a=orig.knowledge_line(orig.features(v.resume_text,v.job_description_text))
    b=overlay.knowledge_line(overlay.features(v.resume_text,v.job_description_text))
    if a!=b:changed[str(v.label)]+=1
# Indexed validation features have no raw text, to preserve candidate privacy.
diag.to_csv(OUT/"val_evidence_statistics_by_row_NO_RAW_TEXT.csv",index=False)
report={
 "checkpoint_sha256":sha,"data_train_count":int(len(tr)),"data_val_count":int(len(va)),
 "train_unique_job_groups":int(tr.job_group.nunique()),"val_unique_job_groups":int(va.job_group.nunique()),
 "job_group_overlap_train_val":int(len(set(tr.job_group)&set(va.job_group))),
 "validation_original_prediction_reproduction_fraction":agreement,
 "GF_count":int(GF.sum()),"GF_missed":int(miss.sum()),"GF_correct":int(hits.sum()),
 "GF_missed_Potential":int((miss&(pred==1)).sum()),"GF_missed_No":int((miss&(pred==0)).sum()),
 "all_validation":group_sum(diag),"missed_good_fit":group_sum(diag.loc[miss]),"correct_good_fit":group_sum(diag.loc[hits]),
 "knowledge_line_changed_validation_count":int(diag.new_knowledge_line_differs.sum()),
 "knowledge_line_changed_training_by_label":dict(changed),
 "job_swap_counterfactual":{"n":int(len(pred)),"different_job_group_per_row":True,
    "overall_changed_prediction_pct":pct(pred_swap!=pred),
    "among_missed_GF_changed_prediction_pct":pct(pred_swap[miss]!=pred[miss]),
    "among_hit_GF_changed_prediction_pct":pct(pred_swap[hits]!=pred[hits]),
    "overall_mean_abs_delta_pgood":round(float(np.mean(np.abs(p_swap[:,2]-p[:,2]))),4),
    "among_missed_mean_abs_delta_pgood":round(float(np.mean(np.abs(p_swap[miss,2]-p[miss,2]))),4),
    "counts_pred_on_original":np.bincount(pred,minlength=3).tolist(),
    "counts_pred_on_swapped_job":np.bincount(pred_swap,minlength=3).tolist(),
    "warning":"No accuracy after job swap: original ground truth labels no longer apply."
 },
 "interpretation_guardrails":"Observed association, not causal demonstration of label noise or of truncation impact; candidate labels not modified; no test read; no CV/JD raw text exported."
}
(OUT/"root_diagnosis_summary.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
