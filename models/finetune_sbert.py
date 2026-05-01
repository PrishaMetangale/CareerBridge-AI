import os
import json
import random
import math
from collections import defaultdict
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from skills_config import JOBS, JOB_DESCRIPTIONS, HF_TO_APP

# Force CPU — MPS runs out of memory on this task
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]        = ""

SBERT_MODEL     = "all-MiniLM-L6-v2"
SBERT_FINETUNED = "sbert_finetuned"
EPOCHS          = 2
BATCH_SIZE      = 16      # small batch — avoids OOM
MAX_PER_ROLE    = 20      # 20 resumes per role is plenty

random.seed(42)

# ── Step 1: Load resumes ──────────────────────────────────────────────────────
print("Loading resumes...")

manifest   = json.load(open("resume_manifest.json"))
role_texts = defaultdict(list)

for entry in manifest:
    role = HF_TO_APP.get(entry["hf_category"])
    if role is None:
        continue
    path = os.path.join("resumes_clean", entry["file"])
    if not os.path.exists(path):
        continue
    text = open(path, encoding="utf-8", errors="ignore").read().strip()
    if len(text) > 100:
        role_texts[role].append(text)

for role in role_texts:
    print(f"  {role}: using {min(len(role_texts[role]), MAX_PER_ROLE)}")

# ── Step 2: Build pairs ───────────────────────────────────────────────────────
print("Building training pairs...")

all_roles      = list(JOBS.keys())
train_examples = []

for role in all_roles:
    job_desc = JOB_DESCRIPTIONS[role]
    texts    = role_texts[role]

    if len(texts) > MAX_PER_ROLE:
        texts = random.sample(texts, MAX_PER_ROLE)

    for resume in texts:
        # positive
        train_examples.append(InputExample(texts=[resume, job_desc], label=1.0))
        # negative
        neg_role = random.choice([r for r in all_roles if r != role])
        train_examples.append(InputExample(texts=[resume, JOB_DESCRIPTIONS[neg_role]], label=0.0))

# job description boundary pairs
for role_a in all_roles:
    for role_b in all_roles:
        label = 1.0 if role_a == role_b else 0.0
        train_examples.append(InputExample(
            texts=[JOB_DESCRIPTIONS[role_a], JOB_DESCRIPTIONS[role_b]], label=label
        ))

print(f"Total pairs: {len(train_examples)}")

# ── Step 3: Split ─────────────────────────────────────────────────────────────
random.shuffle(train_examples)
split      = int(len(train_examples) * 0.9)
train_data = train_examples[:split]
val_data   = train_examples[split:]
print(f"Train: {len(train_data)}  Val: {len(val_data)}")

# ── Step 4: Evaluator ─────────────────────────────────────────────────────────
evaluator = evaluation.EmbeddingSimilarityEvaluator(
    [e.texts[0] for e in val_data],
    [e.texts[1] for e in val_data],
    [e.label    for e in val_data],
    name="val",
)

# ── Step 5: Train on CPU ──────────────────────────────────────────────────────
print("Loading model (CPU mode)...")
model    = SentenceTransformer(SBERT_MODEL, device="cpu")
train_dl = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
loss_fn  = losses.CosineSimilarityLoss(model)
warmup   = math.ceil(len(train_dl) * EPOCHS * 0.1)

print(f"Training: {len(train_dl) * EPOCHS}  should finish in 5-10 mins")

model.fit(
    train_objectives=[(train_dl, loss_fn)],
    evaluator=evaluator,
    epochs=EPOCHS,
    warmup_steps=warmup,
    evaluation_steps=len(train_dl),
    output_path=SBERT_FINETUNED,
    save_best_model=True,
    show_progress_bar=True,
)

print(f"\nDone! Saved to: {SBERT_FINETUNED}/")
print("Run: streamlit run app.py")