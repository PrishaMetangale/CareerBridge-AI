import json, pickle, logging, random
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from skills_config import SKILLS, JOBS, HF_TO_APP

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
random.seed(42); np.random.seed(42)

MIN_REAL = 40

def make_synthetic(role, n):
    # only needed for ML Engineer which had 0 real resumes in the dataset
    # synthetic = randomly pick 70-100% of required skills + some noise
    # not ideal but better than having no data at all for that role
    req   = JOBS[role]
    other = [s for s in SKILLS if s not in req]
    rows  = []
    for _ in range(n):
        vec  = [0] * len(SKILLS)
        for s in random.sample(req, k=random.randint(int(len(req)*0.7), len(req))):
            vec[SKILLS.index(s)] = 1
        for s in random.sample(other, k=random.randint(0, 4)):
            vec[SKILLS.index(s)] = 1
        rows.append(vec)
    return rows

# ── Real data ─────────────────────────────────────────────────────────────────
resumes = json.load(open("extracted_skills.json"))
X_real, y_real = [], []
for r in resumes:
    role = HF_TO_APP.get(r["hf_category"])
    if not role: continue
    X_real.append([1 if s in set(r["skills"]) else 0 for s in SKILLS])
    y_real.append(role)

logging.info("Real samples: %s", dict(Counter(y_real).most_common()))

# important: fit encoder on ALL roles not just ones with data
#got a KeyError for ML Engineer first time
le = LabelEncoder()
le.fit(list(JOBS.keys()))
X_r, y_r = np.array(X_real), le.transform(y_real)

X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42, stratify=y_r)
logging.info("Real train: %d  |  Real test: %d", len(X_train), len(X_test))

# ── Synthetic augmentation (training only) ────────────────────────────────────
X_tr, y_tr = list(X_train), list(y_train)
synthetic_roles = []
for role in JOBS:
    curr = Counter(le.inverse_transform(y_tr)).get(role, 0)
    if curr < MIN_REAL:
        n = MIN_REAL - curr
        X_tr.extend(make_synthetic(role, n))
        y_tr.extend([le.transform([role])[0]] * n)
        synthetic_roles.append(role)
        logging.info("  %s +%d synthetic", role, n)

X_tr, y_tr = np.array(X_tr), np.array(y_tr)
logging.info("Total training (real+synthetic): %d", len(y_tr))

# ── Train ─────────────────────────────────────────────────────────────────────
w  = compute_sample_weight("balanced", y_tr)
lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr, y_tr)
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="mlogloss", verbosity=0,
                    use_label_encoder=False).fit(X_tr, y_tr, sample_weight=w)

logging.info("LR  test: %.2f%%", accuracy_score(y_test, lr.predict(X_test))*100)
xgb_test = accuracy_score(y_test, xgb.predict(X_test))*100
logging.info("XGB test: %.2f%%", xgb_test)

# Report on real test labels only
labels_in_test = sorted(set(y_test))
names_in_test  = le.inverse_transform(labels_in_test)
report = classification_report(y_test, xgb.predict(X_test),
                                labels=labels_in_test, target_names=names_in_test,
                                zero_division=0, output_dict=True)
print(classification_report(y_test, xgb.predict(X_test),
                             labels=labels_in_test, target_names=names_in_test, zero_division=0))
per_class_f1 = {r: round(report[r]["f1-score"],3) for r in names_in_test if r in report}

# ── Cross-validation (real only) ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv, xgb_cv = [], []
for fold, (ti, vi) in enumerate(cv.split(X_r, y_r), 1):
    Xf, yf, Xv, yv = X_r[ti], y_r[ti], X_r[vi], y_r[vi]
    wf  = compute_sample_weight("balanced", yf)
    fle = LabelEncoder().fit(yf)          # contiguous labels per fold
    yfe, yve = fle.transform(yf), fle.transform(yv)
    lrf  = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xf, yfe)
    xgbf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="mlogloss", verbosity=0,
                         use_label_encoder=False).fit(Xf, yfe, sample_weight=wf)
    lr_cv.append(accuracy_score(yve, lrf.predict(Xv)))
    xgb_cv.append(accuracy_score(yve, xgbf.predict(Xv)))
    logging.info("Fold %d — LR: %.2f%%  XGB: %.2f%%", fold, lr_cv[-1]*100, xgb_cv[-1]*100)

xgb_mean, xgb_std = float(np.mean(xgb_cv)*100), float(np.std(xgb_cv)*100)
lr_mean,  lr_std  = float(np.mean(lr_cv)*100),  float(np.std(lr_cv)*100)
logging.info("XGB CV: %.2f%% ± %.2f%%", xgb_mean, xgb_std)
logging.info("LR  CV: %.2f%% ± %.2f%%", lr_mean,  lr_std)

# ── Save results ──────────────────────────────────────────────────────────────
json.dump({"xgb_mean":xgb_mean,"xgb_std":xgb_std,"lr_mean":lr_mean,"lr_std":lr_std,
           "xgb_test_acc":xgb_test,"n_real_train":len(X_train),"n_test":len(X_test),
           "synthetic_roles":synthetic_roles,"per_class":per_class_f1},
          open("cv_results.json","w"), indent=2)

cm = confusion_matrix(y_test, xgb.predict(X_test), labels=labels_in_test)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=names_in_test, yticklabels=names_in_test)
plt.title(f"XGBoost — Real Test: {xgb_test:.1f}%  |  CV: {xgb_mean:.1f}% ± {xgb_std:.1f}%")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig("confusion_matrix.png", bbox_inches="tight")

pickle.dump({"model":xgb,"classes":le.classes_,"skills":SKILLS}, open("xgb_model.pkl","wb"))
np.save("X_train.npy", X_tr); np.save("X_test.npy", X_test)
logging.info("Done — xgb_model.pkl, cv_results.json, confusion_matrix.png")