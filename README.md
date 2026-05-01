# 🎯 CareerBridge AI — Resume Matching & Skill Gap Analysis System

## The Business Problem

Hiring teams spend thousands of hours manually screening resumes — and still miss strong candidates. At the same time, job seekers apply to roles they're underqualified for without knowing exactly what's holding them back. The result: wasted time on both sides, high rejection rates, and no actionable feedback for candidates.

**CareerBridge AI solves both sides of this problem** — automatically matching resumes to job roles with a 6-signal ensemble model, identifying skill gaps against live job postings, and generating personalised interview prep for every gap detected.

---

## What I Built

A production-grade resume intelligence system with a Streamlit web app that:

1. **Parses** resumes (PDF, DOCX, TXT) and extracts 134 skills using regex + alias mapping
2. **Scores** each resume against 9 tech roles using a 6-signal ensemble (XGBoost + SBERT + Claude LLM + 3 custom signals)
3. **Fetches live job postings** via Adzuna API and compares skill gaps against what employers actually ask for right now
4. **Projects score improvement** — shows exactly how much your match score increases if you learn each gap skill
5. **Generates interview prep** — easy/medium/hard questions with hints for every skill gap, powered by Claude API
6. **Recommends YouTube courses** for every gap skill via YouTube Data API

---

## Business Impact

- **68.67% classification accuracy** on 517 unseen real resumes across 9 tech roles
- **6-signal ensemble** outperforms single-model approaches by combining semantic, keyword, experiential, and LLM signals
- Saves a recruiter **~3–4 hours per hiring cycle** by automating first-pass resume screening
- Gives candidates a **specific, actionable learning path** instead of a generic rejection

---

## Technical Architecture

```
careerbridge-ai/
├── app.py                        ← Streamlit app (5 tabs)
├── config.py                     ← Model weights, API constants
├── core.py                       ← 6-signal scoring engine
├── models/
│   ├── train.py                  ← XGBoost training + cross-validation
│   └── finetune_sbert.py         ← SBERT fine-tuning on resume pairs
├── pipelines/
│   ├── scraper.py                ← HuggingFace resume dataset downloader
│   └── skill_extraction.py      ← Batch skill extraction pipeline
├── data/
│   └── sample_resumes/           ← Test resumes for demo
└── requirements.txt
```

---

## The 6 Signals

| Signal | Weight | What It Measures |
|---|---|---|
| Skill Overlap | 30% | Direct keyword match against role requirements |
| SBERT Semantic | 20% | Embedding similarity vs live Adzuna job descriptions |
| Experience Proof | 10% | Skills appearing within 400 chars of action verbs |
| Years of Experience | 15% | Date-range parsing, merged intervals, normalised over 8 years |
| Claude LLM Score | 15% | Holistic recruiter judgement via Claude API |
| XGBoost Classifier | 10% | Learned patterns from 500+ real resumes |

---

## Stack

`Python` `XGBoost` `SBERT (sentence-transformers)` `Claude API (Anthropic)` `Streamlit` `Adzuna API` `YouTube Data API` `scikit-learn` `pdfplumber` `pandas` `numpy`

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/careerbridge-ai.git
cd careerbridge-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and add your API keys:
```
ANTHROPIC_API_KEY=your_key_here
ADZUNA_ID=your_id_here
ADZUNA_KEY=your_key_here
YOUTUBE_KEY=your_key_here
```

### 4. (Optional) Train the XGBoost model
```bash
python pipelines/scraper.py        # download resumes from HuggingFace
python pipelines/skill_extraction.py  # extract skills
python models/train.py             # train + evaluate XGBoost
python models/finetune_sbert.py    # fine-tune SBERT
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## Model Evaluation

- **XGBoost Test Accuracy:** 68.67% on 517 unseen real resumes
- **5-Fold Cross-Validation:** Stratified CV on real resumes only (no data leakage from synthetic augmentation)
- **Synthetic augmentation** used only for training on underrepresented roles (ML Engineer, Cloud Engineer) — clearly flagged in evaluation output
- Confusion matrix and per-class F1 scores visible in the app's Model tab

---

## App Screenshots

> Upload resume → instant 6-signal match → skill gap breakdown → interview prep → live job listings

*[Add screenshots here after deployment]*

---

## Get API Keys

| API | Free Tier | Link |
|---|---|---|
| Anthropic (Claude) | Pay-per-use | [console.anthropic.com](https://console.anthropic.com) |
| Adzuna | 250 calls/month free | [developer.adzuna.com](https://developer.adzuna.com) |
| YouTube Data API | 10,000 units/day free | [console.cloud.google.com](https://console.cloud.google.com) |

---

*Built by Prisha Metangale — [LinkedIn](https://www.linkedin.com/in/prisha-metangale)*
