"""Data-scoped preflight for one HR-category knowledge intervention.
No labels used for extracting features. No validation-driven rule fitting.
Test is never fetched. Raw CV/JD text never saved to artifacts."""
import sys,json
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hr_evidence import evidence,hr_evidence_line
from transformers import AutoTokenizer
sys.path.insert(0,"/tmp/HireMind-DL/src")
from hiremind_dl import load_config
from hiremind_dl.data import download_raw,read_raw,split_frame,load_manifest
cfg=load_config("/tmp/HireMind-DL/configs/experiment.json")
raw={"train":read_raw(download_raw(cfg,"train"))}
manifest=load_manifest(cfg)
tok=AutoTokenizer.from_pretrained(cfg["student"]["name"],revision=cfg["student"]["revision"])
assert hr_evidence_line("Bachelor's degree, CPA, accountant.","Senior accountant job. Bachelor's degree, CPA preferred.").startswith("HR evidence")
assert hr_evidence_line("","No occupation stated.")==""
stats={}
for split in ("train","validation"):
    f=split_frame(raw,manifest,split,cfg["dataset"]["rows"][split])
    all_n=len(f);nonnull=0;too_long=0
    lengths=[];degrees=0;certs=0;roles=0;no_support=0
    for cv,job in zip(f.resume_text,f.job_description_text):
        e=evidence(cv,job)
        line=hr_evidence_line(cv,job)
        nonnull+=bool(line)
        degrees+=bool(e["job_degree_mentioned"])
        certs+=bool(e["job_certs_mentioned"])
        roles+=bool(e["job_role_categories"])
        no_support+=bool(e["job_role_categories"] and not e["cv_job_role_categories"])
        if line:
            length=len(tok.tokenize(line))
            lengths.append(length)
            too_long+=int(length>65)
    stats[split]={"rows":all_n,"knowledge_line_added_pct":round(100*nonnull/all_n,2),
       "job_mentions_degree_pct":round(100*degrees/all_n,2),
       "job_mentions_certification_pct":round(100*certs/all_n,2),
       "job_role_detected_pct":round(100*roles/all_n,2),
       "job_role_not_detected_in_cv_pct":round(100*no_support/all_n,2),
       "new_line_median_wordpieces":sorted(lengths)[len(lengths)//2] if lengths else 0,
       "added_lines_over_65_wordpieces":too_long}
out=Path("preflight_hr");out.mkdir(exist_ok=True)
(out/"summary.json").write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2),flush=True)
