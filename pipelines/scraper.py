# scraper.py — download resumes from HuggingFace
import os,hashlib, json, logging
from datasets import load_dataset
from dotenv import load_dotenv

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
load_dotenv()


RESUME_KEYS   = ["experience", "education", "skills", "email", "phone", "summary", "objective"]
HF_CATEGORIES = [
    "Data Science", "DevOps", "Database", "DotNet Developer", "ETL Developer",
    "Information Technology", "Java Developer", "Network Security Engineer",
    "Python Developer", "React Developer", "SAP Developer", "SQL Developer",
    "Testing", "Web Designing", "Business Analyst", "Blockchain",
]

resumes, seen = [], set()

def is_resume(text):
    t = text.lower()
    return sum(1 for k in RESUME_KEYS if k in t) >= 4 and len(text) > 300


# HuggingFace
logging.info("Collecting from HuggingFace...")
dataset = load_dataset("ahmedheakl/resume-atlas", split="train")
for row in dataset:
    if row.get("Category") not in HF_CATEGORIES:
        continue
    text = row.get("Text", "")
    h    = hashlib.md5(text.encode()).hexdigest()
    if h not in seen and is_resume(text):
        seen.add(h)
        fname = f"resume_{len(resumes)+1}.txt"
        resumes.append({"text": text, "source": "huggingface",
                        "hf_category": row["Category"], "file": fname})

#  Save 
manifest = []
for entry in resumes:
    path = os.path.join("resumes_clean", entry["file"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(entry["text"])
    manifest.append({"file": entry["file"], "source": entry["source"],
                     "hf_category": entry["hf_category"]})

json.dump(manifest, open("resume_manifest.json", "w"), indent=2)
print(f"Done! Total: {len(resumes)} resumes saved to resumes_clean/")