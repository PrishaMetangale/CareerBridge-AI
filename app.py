# app.py - CareerBridge AI
# main streamlit app with 5 tabs

import os, json
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from skills_config import JOBS, extract_skills
from careerbridge_config import (
    SBERT_FINETUNED, SBERT_MODEL,
    W_SKILL, W_SBERT, W_EXP, W_XGB, W_YEARS, W_CLAUDE, YEARS_CAP
)
from core import (
    read_resume, get_live_descriptions, match_resume,
    compute_gap, simulate_after_learning, sig_claude
)

load_dotenv()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
ADZUNA_ID     = os.getenv("ADZUNA_ID")
ADZUNA_KEY    = os.getenv("ADZUNA_KEY")
YOUTUBE_KEY   = os.getenv("YOUTUBE_KEY")
ROLES         = list(JOBS.keys())
LOW_DATA      = {"ML Engineer", "Cloud Engineer"}
CODES         = {"India":"in","UK":"gb","USA":"us","Australia":"au","Canada":"ca"}

st.set_page_config(page_title="CareerBridge AI", page_icon="🎯", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family:'Inter',sans-serif !important; }
.stApp { background-color:#0a0c10; }
.skills-box {
    background:linear-gradient(135deg,#0d2818,#0f3020);
    border:1px solid #1e5c38; border-radius:12px;
    padding:16px 20px; font-size:0.85rem;
    color:#86efac; line-height:2.1;
}
.gap-box {
    background:linear-gradient(135deg,#1a0a0a,#200d0d);
    border:1px solid #5c1e1e; border-radius:12px;
    padding:16px 20px; font-size:0.85rem;
    color:#fca5a5; line-height:2.1;
}
.score-number {
    font-size:3.6rem; font-weight:800;
    color:#fff; line-height:1.1; margin-bottom:12px;
}
.match-badge {
    display:block; padding:10px 20px; border-radius:10px;
    font-weight:600; font-size:0.9rem;
    margin-bottom:20px; text-align:center;
}
.section-heading {
    font-size:1.2rem; font-weight:700;
    color:#f9fafb; margin:22px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# stop if any API key is missing
for k, v in {"ANTHROPIC_API_KEY": ANTHROPIC_KEY,
             "ADZUNA_ID": ADZUNA_ID,
             "ADZUNA_KEY": ADZUNA_KEY}.items():
    if not v:
        st.error(f"Missing environment variable: {k}")
        st.stop()

# ── Cached functions ──────────────────────────────────────────────────────────

@st.cache_resource
def load_sbert():
    if os.path.exists(SBERT_FINETUNED):
        return SentenceTransformer(SBERT_FINETUNED)
    return SentenceTransformer(SBERT_MODEL)

@st.cache_data(ttl=3600, show_spinner="Fetching live job postings...")
def load_live(country):
    return get_live_descriptions(ADZUNA_ID, ADZUNA_KEY, country)

@st.cache_data(ttl=3600)
def fetch_listings(role, country):
    try:
        r = requests.get(
            f"https://api.adzuna.com/v1/api/jobs/{country}/search/1",
            params={"app_id": ADZUNA_ID, "app_key": ADZUNA_KEY,
                    "what": role, "results_per_page": 5},
            timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return []

def get_video(query):
    if not YOUTUBE_KEY:
        return None, None
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "snippet", "q": query,
                    "key": YOUTUBE_KEY, "maxResults": 5, "type": "video"},
            timeout=10)
        r.raise_for_status()
        items = r.json().get("items", [])
        for v in items:
            vid   = v.get("id", {}).get("videoId", "")
            title = v.get("snippet", {}).get("title", "")
            if vid and title:
                return vid, title
        # if no items returned at all
        print(f"YouTube returned {len(items)} items for: {query}")
    except Exception:
        pass
    return None, None

@st.cache_data(show_spinner=False)
def get_prep(skill, role):
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content":
                f"Senior {role} interviewer. Candidate needs: {skill}.\n"
                f"Reply EXACTLY:\n"
                f"EASY\n1) [q] | Hint: [h]\n2) [q] | Hint: [h]\n\n"
                f"MEDIUM\n3) [q] | Hint: [h]\n4) [q] | Hint: [h]\n\n"
                f"HARD\n5) [q] | Hint: [h]\n\n"
                f"STUDY\n1) [t]\n2) [t]\n3) [t]\n\n"
                f"MISTAKE\n[one sentence]"}])
        return msg.content[0].text.strip()
    except Exception:
        return None

# ── Helper functions ──────────────────────────────────────────────────────────

def show_prep(text):
    if not text:
        st.warning("Could not generate prep.")
        return
    section = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line in ("EASY", "MEDIUM", "HARD"):
            section = line
            bg = {"EASY": "#0d2818", "MEDIUM": "#1a1600", "HARD": "#1a0000"}[line]
            em = {"EASY": "🟢", "MEDIUM": "🟡", "HARD": "🔴"}[line]
            st.markdown(
                f'<div style="background:{bg};border-radius:8px;'
                f'padding:5px 14px;margin:8px 0 4px;font-weight:600;">'
                f'{em} {line}</div>', unsafe_allow_html=True)
        elif line == "STUDY":
            section = "STUDY"
            st.markdown("**📚 Study This Week**")
        elif line == "MISTAKE":
            section = "MISTAKE"
            st.markdown("**⚠️ Common Mistake**")
        elif section in ("EASY", "MEDIUM", "HARD") and "|" in line:
            q, hint = line.split("|", 1)
            st.markdown(f"**{q.strip()}**")
            st.caption(hint.strip())
        elif section == "STUDY":
            st.markdown(f"- {line.lstrip('123456789) .')}")
        elif section == "MISTAKE":
            st.error(line)

def get_badge(score):
    if score >= 0.80:
        return "🟢", "Excellent Match", "#0d2818", "#1e5c38", "#86efac"
    elif score >= 0.60:
        return "🟡", "Strong Match", "#1a1600", "#4a3800", "#fde68a"
    elif score >= 0.40:
        return "🟠", "Good Match", "#1a0f00", "#4a2800", "#fdba74"
    else:
        return "🔴", "Partial Match", "#1a0000", "#5c1e1e", "#fca5a5"

# ── App layout ────────────────────────────────────────────────────────────────

st.title("🎯 CareerBridge AI")
st.caption("Upload your resume · match against live job postings · close your skill gaps.")
st.divider()

country_name = st.sidebar.selectbox("Job Market", list(CODES.keys()))
country_code = CODES[country_name]
live  = load_live(country_code)
sbert = load_sbert()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Analysis", "Before vs After", "Interview Prep", "Jobs", "Model"
])

# ── TAB 1: Analysis ───────────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

    if uploaded:
        try:
            with st.spinner("Analysing your resume — please wait..."):

                # step 1: read and extract
                text   = read_resume(uploaded, uploaded.name)
                if not text.strip():
                    st.error("Could not extract text."); st.stop()
                skills = extract_skills(text)
                if not skills:
                    st.warning("No skills found."); st.stop()

                # step 2: run 6-signal match
                best, blended, breakdown, yrs = match_resume(
                    text, skills, sbert, live, ANTHROPIC_KEY)

                # step 3: compute gaps and claude reason
                s_gap, l_gap = compute_gap(skills, best, live)
                all_gaps     = list(dict.fromkeys(s_gap + l_gap))
                _, reason    = sig_claude(text, best, ANTHROPIC_KEY)

                # step 4: pre-generate all interview questions
                prep_data = {}
                for skill in all_gaps:
                    prep_data[skill] = get_prep(skill, best)

                # step 5: pre-fetch YouTube courses
                course_data = {}
                for skill in all_gaps:
                    course_data[skill] = get_video(f"{skill} full course tutorial")

            # clear any old cached data before saving new results
            for key in list(st.session_state.keys()):
                if key.startswith("open_") or key == "courses":
                    del st.session_state[key]

            # save everything to session state
            st.session_state["text"]      = text
            st.session_state["skills"]    = skills
            st.session_state["best"]      = best
            st.session_state["blended"]   = blended
            st.session_state["breakdown"] = breakdown
            st.session_state["yrs"]       = yrs
            st.session_state["gap"]       = all_gaps
            st.session_state["reason"]    = reason
            st.session_state["prep"]      = prep_data
            st.session_state["courses"]   = course_data
            st.session_state["sel"]       = best

        except Exception as e:
            st.error(f"Analysis failed: {e}"); st.stop()

    if "blended" in st.session_state:
        text      = st.session_state["text"]
        skills    = st.session_state["skills"]
        best      = st.session_state["best"]
        blended   = st.session_state["blended"]
        breakdown = st.session_state["breakdown"]
        yrs       = st.session_state["yrs"]
        reason    = st.session_state["reason"]

        sel = st.selectbox("Explore a different role?",
                           ROLES, index=ROLES.index(best))
        st.session_state["sel"] = sel

        # cache gap per role so compute_gap doesnt rerun on every click
        # cache gap per role so compute_gap doesnt rerun on every click
        gap_key = f"gap_{sel}"
        if gap_key not in st.session_state:
            s_gap, l_gap = compute_gap(skills, sel, live)
            st.session_state[gap_key] = list(dict.fromkeys(s_gap + l_gap))
        gap = st.session_state[gap_key]

        # regenerate prep and courses for this role if not cached
        prep_key    = f"prep_{sel}"
        courses_key = f"courses_{sel}"
        if prep_key not in st.session_state:
            with st.spinner(f"Generating prep for {sel}..."):
                st.session_state[prep_key]    = {s: get_prep(s, sel) for s in gap}
                st.session_state[courses_key] = {s: get_video(f"{s} full course tutorial") for s in gap}
        st.session_state["prep"]    = st.session_state[prep_key]
        st.session_state["courses"] = st.session_state[courses_key]
        st.session_state["gap"]     = gap

        matched = [s for s in JOBS[sel] if s in set(skills)]

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown('<div class="section-heading">Extracted Skills</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="skills-box">{" | ".join(sorted(skills))}</div>',
                unsafe_allow_html=True)

            st.markdown('<div class="section-heading">Top 5 Job Matches</div>',
                        unsafe_allow_html=True)
            for role, score in sorted(blended.items(), key=lambda x: -x[1])[:5]:
                sfx = " ⚠️" if role in LOW_DATA else ""
                st.markdown(
                    f'<p style="margin:12px 0 4px;font-weight:600;'
                    f'font-size:0.9rem;color:#e5e7eb;">'
                    f'{role}{sfx}: {int(score*100)}%</p>',
                    unsafe_allow_html=True)
                st.progress(round(min(score, 1.0), 2))

        with right:
            em, label, bg, border, fg = get_badge(blended[sel])
            st.markdown(
                f'<div class="section-heading">Best Match: {best}</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:0.78rem;color:#6b7280;'
                'text-transform:uppercase;margin-bottom:4px;">'
                'BEST FIT SCORE</div>', unsafe_allow_html=True)
            w = int(blended[sel] * 100)
            d = int((blended[sel] * 10000) % 100)
            st.markdown(
                f'<div class="score-number">{w}.{d:02d}%</div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="match-badge" style="background:{bg};'
                f'border:1px solid {border};color:{fg};">'
                f'{em} {label}</div>', unsafe_allow_html=True)

            if sel in LOW_DATA:
                st.warning(f"⚠️ Limited training data for **{sel}**.")

            # skill gaps
            if gap:
                st.markdown('<div class="section-heading">Skill Gaps</div>',
                            unsafe_allow_html=True)
                s_gap_display = [s for s in JOBS[sel] if s not in set(skills)]
                l_gap_display = [s for s in gap if s not in JOBS[sel]]
                if s_gap_display:
                    st.markdown(
                        f'<div class="gap-box"><b>Required by role:</b><br>'
                        f'{" | ".join(sorted(s_gap_display))}</div>',
                        unsafe_allow_html=True)
                if l_gap_display:
                    st.markdown(
                        f'<div class="gap-box" style="margin-top:8px;">'
                        f'<b>Also seen in live postings:</b><br>'
                        f'{" | ".join(sorted(l_gap_display))}</div>',
                        unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="background:#0d2818;border:1px solid #1e5c38;'
                    'border-radius:12px;padding:14px 20px;color:#86efac;'
                    'font-weight:600;">✅ No skill gaps!</div>',
                    unsafe_allow_html=True)

            # score breakdown
            s1 = breakdown["skill"][sel]
            s2 = breakdown["sbert"][sel]
            s3 = breakdown["exp"][sel]
            s4 = breakdown["xgb"][sel]
            s5 = breakdown["years"][sel]
            s6 = breakdown["claude"][sel]

            st.markdown(" ")
            st.markdown("**Score Breakdown**")
            st.divider()
            for lbl, val in [
                (f"Skill Overlap ({int(W_SKILL*100)}%)",       f"{int(s1*100)}%"),
                (f"SBERT Semantic ({int(W_SBERT*100)}%)",      f"{int(s2*100)}%"),
                (f"Experience Proof ({int(W_EXP*100)}%)",      f"{int(s3*100)}%"),
                (f"Years of Experience ({int(W_YEARS*100)}%)", f"{yrs} yrs → {int(s5*100)}%"),
                (f"Claude LLM Score ({int(W_CLAUDE*100)}%)",   f"{int(s6*10)}/10"),
                (f"XGBoost ({int(W_XGB*100)}%)",               f"{int(s4*100)}%"),
            ]:
                c1, c2 = st.columns(2)
                c1.write(lbl)
                c2.write(f"**{val}**")
            st.divider()
            if reason:
                st.caption(reason)

# ── TAB 2: Before vs After ────────────────────────────────────────────────────
with tab2:
    if "blended" not in st.session_state:
        st.info("Upload your resume in the Analysis tab first.")
    else:
        sel = st.session_state.get("sel", st.session_state["best"])
        gap = st.session_state.get(f"gap_{sel}",
              st.session_state.get("gap", []))
        cur = st.session_state["blended"][sel]

        st.subheader(f"Score Projection — {sel}")
        st.caption("See how much your score improves if you learn all gap skills.")

        if not gap:
            st.success("No skill gaps — already at maximum score!")
        else:
            # cache simulation so it doesnt rerun on every interaction
            sim_key = f"sim_{sel}"
            if sim_key not in st.session_state:
                with st.spinner("Simulating projected score..."):
                    st.session_state[sim_key] = simulate_after_learning(
                        st.session_state["text"],
                        st.session_state["skills"],
                        sel, gap, sbert, live, ANTHROPIC_KEY)
            proj = st.session_state[sim_key]

            c1, c2, c3 = st.columns(3)
            c1.metric("Current Score",   f"{int(cur*100)}%")
            c2.metric("Projected Score", f"{int(proj*100)}%",
                      delta=f"+{int((proj-cur)*100)}%")
            c3.metric("Skills to Learn", str(len(gap)))

            st.divider()
            st.markdown("**Gap skills that would boost your score:**")
            for skill in gap:
                st.markdown(f"- `{skill}`")
            st.info(
                "💡 This is the theoretical maximum. Real improvement also "
                "depends on demonstrating these skills with action verbs "
                "and measurable results.")

# ── TAB 3: Interview Prep ─────────────────────────────────────────────────────
with tab3:
    if "prep" not in st.session_state:
        st.info("Upload your resume in the Analysis tab first.")
    else:
        prep    = st.session_state["prep"]
        courses = st.session_state["courses"]
        gap     = st.session_state.get("gap", [])
        sel     = st.session_state.get("sel", st.session_state["best"])

        if not gap:
            st.success("No skill gaps — you are ready to apply!")
        else:
            st.info(f"**{len(gap)} skill gap(s)** detected for **{sel}**")
            c1, c2 = st.columns(2, gap="large")

            with c1:
                st.subheader("Interview Questions")
                for skill in gap:
                    open_key = f"open_{skill}"
                    if open_key not in st.session_state:
                        st.session_state[open_key] = False

                    # skill box with arrow button
                    arrow = "∧" if st.session_state[open_key] else "∨"
                    col_a, col_b = st.columns([6, 1])
                    with col_a:
                        st.markdown(
                            f'<div style="border:1px solid #444;'
                            f'border-radius:8px;padding:12px 16px;'
                            f'font-weight:600;">{skill.title()}</div>',
                            unsafe_allow_html=True)
                    with col_b:
                        st.markdown(
                            "<div style='height:8px'></div>",
                            unsafe_allow_html=True)
                        if st.button(arrow, key=f"btn_{skill}"):
                            st.session_state[open_key] = not st.session_state[open_key]

                    # show questions if open - already pre-generated, instant
                    if st.session_state[open_key]:
                        show_prep(prep.get(skill))
                    st.markdown(" ")

            with c2:
                st.subheader("Free Courses")
                for skill in gap:
                    vid, title = courses.get(skill, (None, None))
                    if vid:
                        st.success(
                            f"**{skill.title()}:** "
                            f"[{title}](https://youtube.com/watch?v={vid})")
                    else:
                        st.caption(f"No course found for {skill.title()}")

# ── TAB 4: Jobs ───────────────────────────────────────────────────────────────
with tab4:
    if "best" not in st.session_state:
        st.info("Upload your resume in the Analysis tab first.")
    else:
        sel = st.session_state.get("sel", st.session_state["best"])
        st.subheader(f"Live {sel} Listings — {country_name}")
        st.caption(
            "Apply to these opportunities.")
        with st.spinner("Fetching listings..."):
            jobs = fetch_listings(sel, country_code)
        if jobs:
            for job in jobs:
                title   = job.get("title", "N/A")
                company = job.get("company", {}).get("display_name", "N/A")
                loc     = job.get("location", {}).get("display_name", "N/A")
                desc    = job.get("description", "")[:150]
                url     = job.get("redirect_url", "#")
                st.markdown(f"**{title}** — {company}")
                st.write(f"📍 {loc}")
                st.write(f"{desc}…")
                st.markdown(f"[View & Apply]({url})")
                st.markdown("---")
        else:
            st.warning("No listings found. Try a different country.")

# ── TAB 5: Model ──────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Model Evaluation & Accuracy Metrics")

    ft = os.path.exists(SBERT_FINETUNED)
    if ft:
        st.success("✅ Fine-tuned SBERT active")
    else:
        st.warning("Base SBERT in use. Run finetune_sbert.py.")

    if os.path.exists("cv_results.json"):
        cv = json.load(open("cv_results.json"))
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("XGBoost 5-Fold CV",
                  f"{cv['xgb_mean']:.1f}%", delta=f"± {cv['xgb_std']:.1f}%")
        c2.metric("Logistic Regression CV",
                  f"{cv['lr_mean']:.1f}%",  delta=f"± {cv['lr_std']:.1f}%")
        c3.metric("XGBoost Test Accuracy",
                  f"{cv.get('xgb_test_acc', 0):.1f}%")
        c4.metric("Training Samples (real only)",
                  str(cv.get("n_real_train", "—")))
        st.caption(
            f"Evaluated on {cv.get('n_test','?')} held-out real resumes. "
            f"Synthetic: {', '.join(cv.get('synthetic_roles',[]))} (⚠️)")

        if "per_class" in cv:
            st.divider()
            st.markdown("**Per-Role F1 Scores (real data only)**")
            cols = st.columns(5)
            for i, (role, f1) in enumerate(cv["per_class"].items()):
                cols[i % 5].metric(role, f"{f1:.2f}")
    else:
        st.info("Run train.py to generate cv_results.json")

    st.divider()
    st.markdown("**Signal Weight Justification**")
    for lbl, wt, rat in [
        ("Skill Overlap",       f"{int(W_SKILL*100)}%",  "Direct keyword coverage"),
        ("SBERT Semantic",      f"{int(W_SBERT*100)}%",  "Embeddings vs live Adzuna descriptions"),
        ("Experience Proof",    f"{int(W_EXP*100)}%",    "Skills within 400 chars of action verbs"),
        ("Years of Experience", f"{int(W_YEARS*100)}%",  f"Normalised over {YEARS_CAP} years"),
        ("Claude LLM Score",    f"{int(W_CLAUDE*100)}%", "Holistic recruiter judgement"),
        ("XGBoost",             f"{int(W_XGB*100)}%",    "Learned classifier"),
    ]:
        c1, c2, c3 = st.columns([3, 1, 4])
        c1.write(lbl)
        c2.write(f"**{wt}**")
        c3.write(rat)

    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png",
                 caption="XGBoost Confusion Matrix")
    if os.path.exists("shap_bar.png"):
        st.image("shap_bar.png",
                 caption="SHAP Feature Importance — Top 20 Skills")