# skill_extraction.py — extract skills from every resume and save to JSON

import os, json, logging
from skills_config import extract_skills

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if os.path.exists("extracted_skills.json"):
    logging.info("extracted_skills.json already exists. Delete it to re-run.")
    raise SystemExit(0)

manifest = json.load(open("resume_manifest.json"))
results  = []

logging.info("Processing %d resumes...", len(manifest))
for entry in manifest:
    path = os.path.join("resumes_clean", entry["file"])
    try:
        text   = open(path, encoding="utf-8", errors="ignore").read()
        skills = extract_skills(text)
        if skills:
            results.append({
                "file":        entry["file"],
                "source":      entry["source"],
                "hf_category": entry["hf_category"],
                "skills":      skills,
            })
    except FileNotFoundError:
        logging.warning("Missing: %s", path)
    except Exception as e:
        logging.error("Error %s: %s", entry["file"], e)

json.dump(results, open("extracted_skills.json", "w"), indent=2)
logging.info("Done! %d resumes → extracted_skills.json", len(results))