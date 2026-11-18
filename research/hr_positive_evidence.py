"""Controlled ablation of HR-category augmentation: POSITIVE EVIDENCE ONLY.

Hypothesis: negative/unknown mentions in the earlier HR augmentation were
misleading. This version emits a compact cue ONLY when both job and CV text
support an overlapping credential, degree tier, or occupation. Never treats
missing CV text as proof of lack. Original Hams Knowledge v1 is unchanged.

This is *not* real HR-scoring-policy conditioning: public training data lacks HR
policy weights. This is a research-only input feature. No test/labels used.
"""
from hr_evidence import evidence

_LEVEL={"diploma":1,"bachelors":2,"masters":3,"doctorate":4}

def hr_evidence_line(cv:str,job:str)->str:
    f=evidence(cv,job)
    chunks=[]
    jd=f["job_degree_mentioned"];cd=f["cv_degree_mentioned"]
    if jd is not None and cd is not None:
        # State documented educational tier, never claim it satisfies HR policy.
        chunks.append("Education mentions JD:"+jd+", CV:"+cd)
    if f["cv_job_certs_mentioned"]:
        chunks.append("Credential overlap:"+"/".join(f["cv_job_certs_mentioned"][:3]))
    if f["cv_job_role_categories"]:
        chunks.append("Occupation evidence:"+"/".join(f["cv_job_role_categories"][:3]))
    if not chunks:
        return ""
    return "Documented CV and job evidence: "+"; ".join(chunks)+"."
