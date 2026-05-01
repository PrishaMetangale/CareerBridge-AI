import re, io, json, datetime, pickle, os, logging
import numpy as np
import requests
import pdfplumber
import anthropic
from sklearn.metrics.pairwise import cosine_similarity
from skills_config import SKILLS, JOBS, JOB_DESCRIPTIONS, extract_skills
from careerbridge_config import (
    CLAUDE_MODEL, SBERT_MODEL, SBERT_FINETUNED,
    ACTION_VERBS, W_SKILL, W_SBERT, W_EXP,
    W_XGB, W_YEARS, W_CLAUDE, YEARS_CAP, ADZUNA_RESULTS_PER_ROLE,
)

log   = logging.getLogger(__name__)
ROLES = list(JOBS.keys())

# ── Resume reader ─────────────────────────────────────────────────────────────

def read_resume(file_obj, filename):
    raw = file_obj.read()
    if filename.endswith(".docx"):
        from docx import Document
        return " ".join(p.text for p in Document(io.BytesIO(raw)).paragraphs)
    if filename.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    pages = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            t = page.extract_text(x_tolerance=2, y_tolerance=2)
            if t:
                pages.append(t)
    return " ".join(pages)

# ── Live Adzuna descriptions ──────────────────────────────────────────────────

def fetch_live_description(role, aid, akey, country="gb"):
    try:
        r = requests.get(
            f"https://api.adzuna.com/v1/api/jobs/{country}/search/1",
            params={"app_id": aid, "app_key": akey, "what": role,
                    "results_per_page": ADZUNA_RESULTS_PER_ROLE}, timeout=10,
        )
        r.raise_for_status()
        text = " ".join(j.get("description", "") for j in r.json().get("results", [])).strip()
        if len(text) > 200:
            return text
    except Exception as e:
        log.warning("Adzuna failed for %s: %s", role, e)
    return JOB_DESCRIPTIONS[role]

def get_live_descriptions(aid, akey, country="gb"):
    return {r: fetch_live_description(r, aid, akey, country) for r in ROLES}

# ── Six signals ───────────────────────────────────────────────────────────────

def sig_skill(skill_set, role):
    req = JOBS[role]
    return len([s for s in req if s in skill_set]) / len(req) if req else 0.0

def sig_sbert(text, model, live):
    vecs = model.encode([text] + [live[r] for r in ROLES])
    sims = cosine_similarity([vecs[0]], vecs[1:])[0]
    return {r: float(np.clip(sims[i], 0, 1)) for i, r in enumerate(ROLES)}

def sig_exp(text, skill_set):
    # TODO: improve window size - 400 chars works but could be smarter
    # tried 200 chars first but it missed too many matches
    apat = re.compile(r'\b(' + '|'.join(re.escape(v) for v in ACTION_VERBS) + r')\b', re.I)
    apos = [m.start() for m in apat.finditer(text)]
    scores = {}
    for role in ROLES:
        backed = sum(
            1 for s in JOBS[role] if s in skill_set and
            any(abs(sp - ap) < 400
                for sp in [m.start() for m in re.finditer(re.escape(s), text, re.I)]
                for ap in apos)
        )
        scores[role] = backed / len(JOBS[role]) if JOBS[role] else 0.0
    return scores

def sig_xgb(skill_set):
    scores = {r: 0.0 for r in ROLES}
    if not os.path.exists("xgb_model.pkl"):
        return scores, False
    saved = pickle.load(open("xgb_model.pkl", "rb"))
    vec   = np.array([[1 if s in skill_set else 0 for s in saved["skills"]]])
    probs = saved["model"].predict_proba(vec)[0]
    for i, cls in enumerate(saved["classes"]):
        if cls in scores:
            scores[cls] = float(probs[i])
    mx = max(scores.values())
    if mx > 1e-9:
        scores = {r: v / mx for r, v in scores.items()}
    return scores, True

def sig_years(text):
    now = datetime.datetime.now()
    MO  = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
           "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
           "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,
           "sep":9,"oct":10,"nov":11,"dec":12}

    def parse(s):
        s = s.strip().lower()
        if s in ("present","current","now","till date"): return now
        m = re.match(r'([a-z]+)\s+(\d{4})', s)
        if m and m.group(1) in MO: return datetime.datetime(int(m.group(2)), MO[m.group(1)], 1)
        m = re.match(r'(\d{4})$', s)
        if m: return datetime.datetime(int(m.group(1)), 6, 1)
        return None

    pat = re.compile(
        r'((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|'
        r'aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}|\d{4})'
        r'\s*[–\-—to]+\s*'
        r'((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|'
        r'aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}|\d{4}'
        r'|present|current|now)', re.I)

    intervals = sorted(
        [(s, e) for m in pat.finditer(text)
         for s, e in [(parse(m.group(1)), parse(m.group(2)))]
         # max 10 years per interval - filters out project data ranges like "2010 to 2022"
         # min 1 month - filters out same-month ranges
         if s and e and e > s and 1 <= (e.year-s.year)*12+(e.month-s.month) <= 120],
        key=lambda x: x[0]
    )
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])

    total = sum((e.year-s.year)*12+(e.month-s.month) for s,e in merged) / 12
    return round(total, 1), round(min(total / YEARS_CAP, 1.0), 4)

def sig_claude(text, role, api_key):
    # using Claude as a judge - gives a holistic score that pure ML cant capture
    # only runs for top 3 roles to save API calls
    try:
        msg = anthropic.Anthropic(api_key=api_key).messages.create(
            model=CLAUDE_MODEL, max_tokens=120,
            messages=[{"role":"user","content":
                f"Expert recruiter. Score resume for {role} role 0-10.\n\n"
                f"Resume:\n{text[:3000]}\n\n"
                f'Reply ONLY: {{"score":<0-10>,"reason":"<one sentence>"}}'}])
        data = json.loads(re.sub(r"```(?:json)?|```","",msg.content[0].text).strip())
        return round(min(max(float(data["score"])/10,0),1),4), data.get("reason","")
    except Exception:
        return 0.5, ""

# ── Main match ────────────────────────────────────────────────────────────────

def match_resume(text, skills, sbert_model, live, api_key):
    # main function - runs all 6 signals and blends them
    # weights were tuned by testing on multiple resumes and checking if results made sense
    skill_set = set(skills)
    s1 = {r: sig_skill(skill_set, r) for r in ROLES}
    s2 = sig_sbert(text, sbert_model, live)
    s3 = sig_exp(text, skill_set)
    s4, xgb_ok = sig_xgb(skill_set)
    total_years, yrs = sig_years(text)
    s5 = {r: yrs for r in ROLES}

    pre  = {r: W_SKILL*s1[r] + W_SBERT*s2[r] for r in ROLES}
    top3 = sorted(pre, key=pre.get, reverse=True)[:3]
    s6   = {r: 0.5 for r in ROLES}
    for r in top3:
        s6[r], _ = sig_claude(text, r, api_key)

    blended = {r: min(W_SKILL*s1[r]+W_SBERT*s2[r]+W_EXP*s3[r]+
                      W_XGB*s4[r]+W_YEARS*s5[r]+W_CLAUDE*s6[r], 1.0) for r in ROLES}
    best = max(blended, key=blended.get)
    return best, blended, {"skill":s1,"sbert":s2,"exp":s3,"xgb":s4,
                           "years":s5,"claude":s6,"xgb_ok":xgb_ok}, total_years

# ── Simulation & gap ──────────────────────────────────────────────────────────

def simulate_after_learning(text, skills, role, gap, sbert_model, live, api_key):
    _, blended, _, _ = match_resume(text, list(set(skills)|set(gap)), sbert_model, live, api_key)
    return blended[role]

def compute_gap(skills, role, live):
    skill_set  = set(skills)
    static_gap = [s for s in JOBS[role] if s not in skill_set]
    live_skills = set(extract_skills(live.get(role,""), infer_implied=False))
    # Only show live gaps that appear in at least 2 roles' skill definitions
    # This removes role-irrelevant noise (e.g. "accessibility" in a Data Analyst posting)
    skill_freq  = {s: sum(1 for r in ROLES if s in JOBS[r]) for s in live_skills}
    live_gap = sorted(s for s in (live_skills - skill_set - set(JOBS[role]))
                      if skill_freq.get(s, 0) >= 2)
    return static_gap, live_gap