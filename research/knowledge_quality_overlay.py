"""Experimental, conservative evidence-quality overlay for Hams Knowledge v1.

DO NOT use for deployment without matched validation. Only changes the knowledge line:
- explicit negation near a CV skill does not count as positive evidence;
- clearly optional/preferred JD skills are separately summarized.
No labels or architecture are modified. The pinned dataset/test split is untouched.
"""
import re
import knowledge_orig as old

NEG_PRE = re.compile(
    r"(?:\bno\s+(?:relevant\s+|prior\s+|hands-on\s+)?"
    r"(?:experience|knowledge|background|skills?|proficiency|familiarity)\s+"
    r"(?:in|with|of|using)\s+"
    r"|\bnot\s+(?:familiar|experienced|proficient|skilled)\s+(?:in|with)\s+"
    r"|\bwithout\s+(?:any\s+)?(?:experience|knowledge|skills?)\s+(?:in|with)\s+"
    r"|\b(?:lack|lacking)\s+(?:experience|knowledge|skills?)\s+(?:in|with)\s+"
    r"|\b(?:do\s+not|don't)\s+know\s+)"
    r"(?:[\w+#.-]+\s+){0,3}$",
    flags=re.I
)
OPTIONAL = re.compile(r"\b(?:preferred|optional|desirable|nice[\s-]+to[\s-]+have|bonus)\b|\ba\s+plus\b", re.I)
MANDATORY = re.compile(r"\b(?:required|mandatory|must[\s-]+have|essential|non[\s-]?negotiable)\b", re.I)

def _occurrences(text, skill):
    pat = old._PATTERNS[skill]
    return list(pat.finditer(text))

def _cv_negated(text, skill):
    occ = _occurrences(text, skill)
    if not occ:
        return False
    negated = []
    for m in occ:
        left = text[max(0, m.start()-125):m.start()]
        left = re.split(r"[.;\n!?]", left)[-1]
        negated.append(bool(NEG_PRE.search(left)))
    return all(negated)

def _is_optional_skill(job, skill):
    occ = _occurrences(job, skill)
    if not occ:
        return False
    flags = []
    for m in occ:
        lo = max(job.rfind("\n", 0, m.start()), job.rfind(".", 0, m.start()), job.rfind(";", 0, m.start()))
        hi_candidates = [p for p in (job.find("\n", m.end()), job.find(".", m.end()), job.find(";", m.end())) if p >= 0]
        hi = min(hi_candidates) if hi_candidates else len(job)
        clause = job[lo+1:hi]
        flags.append(bool(OPTIONAL.search(clause)) and not bool(MANDATORY.search(clause)))
    return all(flags)

def features(cv, job):
    f = old.features(cv, job)
    js = set(f["job_skills"])
    recognized = set(old.skills_in(cv))
    neg = {s for s in js & recognized if _cv_negated(cv, s)}
    optional = {s for s in js if _is_optional_skill(job, s)}
    mandatory = js - optional
    evidenced = recognized - neg
    found = mandatory & evidenced
    f["job_skills"] = sorted(mandatory)
    f["found"] = sorted(found)
    f["missing"] = sorted(mandatory - evidenced)
    f["n_job_skills"] = len(mandatory)
    f["n_found"] = len(found)
    f["match_ratio"] = len(found)/len(mandatory) if mandatory else None
    f["optional_job_skills"] = sorted(optional)
    f["optional_found"] = sorted(optional & evidenced)
    f["cv_negated"] = sorted(neg)
    f["changed_from_v1"] = bool(optional or neg)
    return f

def knowledge_line(f):
    if not f.get("changed_from_v1"):
        return old.knowledge_line(f)
    parts = [
        f"Skills check: {f['n_found']} of {f['n_job_skills']} non-optional job skills evidenced",
        "found: " + (", ".join(f["found"][:10]) if f["found"] else "none")
    ]
    if f["missing"]:
        parts.append("not evidenced: " + ", ".join(f["missing"][:10]))
    if f["optional_job_skills"]:
        parts.append("preferred/optional: " + ", ".join(f["optional_job_skills"][:10]))
    if f["cv_negated"]:
        parts.append("explicitly negated in CV: " + ", ".join(f["cv_negated"][:10]))
    if f["meets_years"] is not None:
        parts.append(f"experience {f['cv_years']} years, job asks {f['job_years']}")
    return "; ".join(parts) + "."
