"""Predeclare and quantify whether proposed input packing retains job skill evidence.
Uses frozen validation for diagnostics, NEVER edits labels / opens test / fits model.
No raw CV or job text ever written to output.
"""
import os,sys,json,hashlib
from pathlib import Path
import pandas as pd,numpy as np
from transformers import AutoTokenizer

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,"/tmp/Senior-2/evaluation/knowledge")
from knowledge import features,skills_in,knowledge_line
sys.path.insert(0,str(root/"research"))
from job_conditioned_cv import make_cv,words
sys.path.insert(0,"/tmp/HireMind-DL/src")
from hiremind_dl import load_config
from hiremind_dl.data import download_raw,read_raw,split_frame,load_manifest
config=load_config("/tmp/HireMind-DL/configs/experiment.json")
raw={"train":read_raw(download_raw(config,"train"))}
manifest=load_manifest(config)
frame=split_frame(raw,manifest,"validation",1162)
pred=pd.read_csv("/tmp/final-results/test-results/val_predictions_final_s42.csv")
assert pred.row_uid.tolist()==frame.row_uid.tolist()
assert pred.label.tolist()==frame.label_id.tolist()
tok=AutoTokenizer.from_pretrained(config["student"]["name"],revision=config["student"]["revision"])
outdir=root/"pilot_preflight";outdir.mkdir(exist_ok=True)

def visible_cv(cv, job_aug):
    e=tok(cv,job_aug,max_length=512,truncation="longest_first",padding=False)
    typ=e.get("token_type_ids")
    if typ is None: raise ValueError("tokenizer must supply token_type_ids")
    indexes=[i for i,t in enumerate(typ) if t==0]
    # includes CLS and first SEP
    ids=[e["input_ids"][i] for i in indexes[1:-1]]
    return tok.decode(ids,skip_special_tokens=True)

rows=[]
for i,v in enumerate(frame.itertuples(index=False)):
    cv=v.resume_text;job=v.job_description_text
    new=make_cv(cv,job,tok)
    aug=knowledge_line(features(cv,job))+"\n"+job
    full_found=set(features(cv,job)["found"])
    v0=visible_cv(cv,aug);v1=visible_cv(new,aug)
    base=skills_in(v0)&full_found
    trt=skills_in(v1)&full_found
    terms=words(job[:2400])
    w0=words(v0);w1=words(v1)
    rows.append({
      "row_uid":v.row_uid, "job_group":v.job_group, "label":v.label_id, "baseline_pred":int(pred.pred.iloc[i]),
      "cv_changed":cv!=new, "full_found":len(full_found),
      "found_visible_baseline":len(base),"found_visible_treatment":len(trt),
      "job_terms_visible_baseline":len(w0&terms),"job_terms_visible_treatment":len(w1&terms),
      "cv_tokens_original":len(tok.tokenize(cv)),"cv_tokens_new":len(tok.tokenize(new))
    })
df=pd.DataFrame(rows)
df.to_csv(outdir/"preflight_no_raw_text.csv",index=False)

def group(g):
    if len(g)==0:return {}
    return {
       "n":int(len(g)),
       "changed_cv_pct":round(g.cv_changed.mean()*100,2),
       "full_match_any_pct":round((g.full_found>0).mean()*100,2),
       "more_visible_job_terms_pct":round((g.job_terms_visible_treatment>g.job_terms_visible_baseline).mean()*100,2),
       "fewer_visible_job_terms_pct":round((g.job_terms_visible_treatment<g.job_terms_visible_baseline).mean()*100,2),
       "mean_job_terms_visible_base":round(g.job_terms_visible_baseline.mean(),2),
       "mean_job_terms_visible_new":round(g.job_terms_visible_treatment.mean(),2),
       "skill_evidence_gain_count":int((g.found_visible_treatment>g.found_visible_baseline).sum()),
       "skill_evidence_loss_count":int((g.found_visible_treatment<g.found_visible_baseline).sum()),
       "skill_evidence_gain_on_missed":None
    }
report={
  "intervention":"Job-conditioned CV evidence sentence frontloading (CV text only; knowledge line unchanged).",
  "frozen_split_sha256":config["dataset"]["split_sha256"],
  "total_validation":group(df),
  "missed_good_fit":group(df[(df.label==2)&(df.baseline_pred!=2)]),
  "correct_good_fit":group(df[(df.label==2)&(df.baseline_pred==2)]),
  "nongood_fit":group(df[df.label!=2]),
  "original_validation_predictions_reused":"Hams Seed 42 final checkpoint, unmodified",
  "warning":"Text coverage diagnosis only; cannot infer accuracy gains before actual training"
}
(outdir/"preflight_summary.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
