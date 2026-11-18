"""HireMind: prototype HR-category evidence augmentation (validation research only).

MUST NOT be called "HR-guided weights" or "HR-selected scoring": the frozen public
training dataset does NOT contain per-job HR scoring policies or human HR ratings.

The only intervention is adding a compact CV-vs-JD evidence line before the JD.
The original Hams Knowledge v1 line stays unchanged. Never infer Good/No/Potential labels.
No historical HR feedback, outcomes, assessment scores, or validation/test labels used.

Conservative observed mentions only; absence is 'not documented', NOT proof
that the candidate lacks a qualification. All examples are anonymized in artifacts.
"""
import re

_DEGREES = [
    ("doctorate", re.compile(r"\b(?:ph\.?\s?d\.?|doctorate|doctoral\s+degree)\b",re.I)),
    ("masters", re.compile(r"\b(?:master'?s?\s+degree|master\s+of|m\.?\s?sc\.?|mba\b)\b",re.I)),
    ("bachelors", re.compile(r"\b(?:bachelor'?s?\s+degree|bachelor\s+of|b\.?\s?sc\.?|b\.?\s?a\.?\s+degree)\b",re.I)),
    ("diploma", re.compile(r"\b(?:diploma|associate'?s?\s+degree)\b",re.I)),
]
# Long-form, defensible official certifications only; no general soft-skill guessing.
_CERTS = [
    ("CPA", re.compile(r"\b(?:CPA|certified\s+public\s+accountant)\b",re.I)),
    ("CFA", re.compile(r"\b(?:CFA|chartered\s+financial\s+analyst)\b",re.I)),
    ("PMP", re.compile(r"\b(?:PMP|project\s+management\s+professional)\b",re.I)),
    ("CMA", re.compile(r"\b(?:CMA|certified\s+management\s+accountant)\b",re.I)),
    ("CISSP", re.compile(r"\bCISSP\b",re.I)),
    ("CCNA", re.compile(r"\bCCNA\b",re.I)),
    ("SHRM", re.compile(r"\bSHRM(?:-CP|-SCP)?\b",re.I)),
]
# Occupation equivalences are evidence hints, NOT mandatory requirement checks.
_ROLE = {
    "accounting":r"\b(?:accountant|accounting|bookkeep|auditor|audit|financial\s+report)\w*\b",
    "salesforce":r"\b(?:salesforce|apex\s+development|sales\s+cloud|service\s+cloud)\b",
    "data":r"\b(?:data\s+(?:analyst|scientist|engineer)|business\s+intelligence|analytics)\b",
    "software":r"\b(?:software\s+(?:engineer|developer|development)|programmer|web\s+developer)\b",
    "nursing":r"\b(?:nurs(?:e|ing)|registered\s+nurse|rn\b)\b",
    "construction":r"\b(?:construction|civil\s+engineer|site\s+engineer)\b",
    "marketing":r"\b(?:marketing|brand\s+management|digital\s+campaign)\b",
    "sales":r"\b(?:sales\s+(?:representative|manager|executive)|account\s+executive)\b",
    "HR":r"\b(?:human\s+resources|recruitment|talent\s+acquisition)\b",
    "mechanical":r"\b(?:mechanical\s+engineer|mechanical\s+design)\b",
}
_ROLE_PAT = {k:re.compile(v,re.I) for k,v in _ROLE.items()}
_WS=re.compile(r"\s+")

def _highest_degree(s):
    for name,pat in _DEGREES:
        if pat.search(s):
            return name
    return None

def evidence(cv:str, job:str)->dict:
    jd=_highest_degree(job)
    cd=_highest_degree(cv)
    jc=[name for name,pat in _CERTS if pat.search(job)]
    cc=[name for name,pat in _CERTS if pat.search(cv)]
    roles=[name for name,pat in _ROLE_PAT.items() if pat.search(job)]
    cvroles=[name for name,pat in _ROLE_PAT.items() if pat.search(cv)]
    return {
       "job_degree_mentioned":jd,
       "cv_degree_mentioned":cd,
       "job_certs_mentioned":jc,
       "cv_job_certs_mentioned":[x for x in jc if x in cc],
       "job_role_categories":roles,
       "cv_job_role_categories":[x for x in roles if x in cvroles],
    }

def hr_evidence_line(cv:str, job:str)->str:
    f=evidence(cv,job)
    parts=[]
    if f["job_degree_mentioned"]:
        # Do not claim it is required: we only establish text mentions.
        parts.append("Education mentioned JD:"+f["job_degree_mentioned"]+
          ", CV:"+(f["cv_degree_mentioned"] or "not stated"))
    if f["job_certs_mentioned"]:
        parts.append("Certificates mentioned JD:"+"/".join(f["job_certs_mentioned"][:3])+
          ", CV evidence:"+("/".join(f["cv_job_certs_mentioned"][:3]) if f["cv_job_certs_mentioned"] else "not stated"))
    if f["job_role_categories"]:
        parts.append("Occupation evidence JD:"+"/".join(f["job_role_categories"][:3])+
          ", CV mentions:"+("/".join(f["cv_job_role_categories"][:3]) if f["cv_job_role_categories"] else "not stated"))
    # No new line when these well-defined signals do not exist, preserve original
    # entirely and avoid wasting tokens on "unknown" boilerplate.
    if not parts:return ""
    return "HR evidence (text mentions only): "+"; ".join(parts)+"."
