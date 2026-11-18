"""Conservative, deterministic, label-free, job-conditioned CV evidence packing.

ONE changed factor: the order of CV sentences in the text presented to BERT-Tiny.
Keep the existing Hams Knowledge v1, job description, model, labels and training
parameters untouched. No inference on labels or test data.
"""
import re
from functools import lru_cache

STOP = frozenset("""
a an the and or in on at of for from to by with as into be is are was were that which
where you your our their they it this these those who we us candidates candidate role
job position work working looking seeking excellent strong ability abilities skills skill
knowledge experience experiences have has years year including such must should will
can any all required preferred minimum plus etc good communication project team
""".split())
WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#.-]{2,}")
SPLIT = re.compile(r"(?:\r?\n)+|(?<=[.!?])\s+(?=[A-Z0-9])|(?<=;)\s+")
EDU = re.compile(r"\b(?:degree|bachelor|master|diploma|certification|certificate|licensed|certified|cpa|cfa|gaap|phd|mba|university|graduate)\b", re.I)
EXP = re.compile(r"\b(?:years?\s+(?:of\s+)?experience|experience\s+(?:with|in)|employment|worked|employment|implemented|developed|managed|built|led|designed)\b", re.I)

def words(t):
    return {x.lower() for x in WORD.findall(t) if x.lower() not in STOP}

def segments(cv):
    bits=[x.strip() for x in SPLIT.split(cv) if x.strip()]
    # Long single paragraphs can dominate. Word-boundary chunks are deterministic.
    ret=[]
    for b in bits:
        toks=b.split()
        for i in range(0,len(toks),42):
            ret.append(" ".join(toks[i:i+42]))
    return [x for x in ret if len(x)>=8]

def make_cv(cv, job, tok):
    """Bring high-scoring relevant, ORIGINAL spans to the front of long CVs.

    Exact text copied, not generated; all baseline full text preserved behind
    the evidence prefix. Scoring uses job text and CV text only.
    """
    if not cv or not job:
        return cv
    try:
        orig_n=len(tok.tokenize(cv))
    except Exception:
        return cv
    if orig_n<=280:
        return cv
    parts=segments(cv)
    if len(parts)<4:
        return cv
    # The first job line is usually the title; avoid generic job boilerplate.
    title=words(job[:220])
    jobterms=words(job[:4500])
    # Prefer uncommon, substantive overlap. No label or model predictions used.
    ranked=[]
    for idx,piece in enumerate(parts):
        w=words(piece)
        if not w:
            continue
        titlehits=len(w&title)
        jobhits=len(w&jobterms)
        score=6.0*titlehits + 1.4*jobhits + 1.0*bool(EDU.search(piece)) + 1.0*bool(EXP.search(piece))
        if score>1:
            ranked.append((-score,idx,piece))
    if not ranked:
        return cv
    ranked.sort()
    chosen=[];used=0
    for _,idx,piece in ranked:
        # Use only 130 WordPiece tokens of the likely ~255 CV tokens available.
        n=len(tok.tokenize(piece))
        if n>75 or used+n>130:
            continue
        chosen.append((idx,piece))
        used+=n
        if used>=110:
            break
    if not chosen:
        return cv
    # Original text follows, so we do not discard evidence irreversibly.
    chosen.sort(key=lambda x:x[0])
    return "CV evidence relevant to job:\n"+"\n".join(p for _,p in chosen)+"\nOriginal CV:\n"+cv
