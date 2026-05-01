# careerbridge_config.py

CLAUDE_MODEL            = "claude-sonnet-4-20250514"
SBERT_MODEL             = "all-MiniLM-L6-v2"
SBERT_FINETUNED         = "sbert_finetuned"
ADZUNA_RESULTS_PER_ROLE = 5   # live postings fetched per role for matching

# Weights must sum to 1.0
W_SKILL  = 0.30
W_SBERT  = 0.20
W_EXP    = 0.10
W_XGB    = 0.10
W_YEARS  = 0.15
W_CLAUDE = 0.15

assert abs(W_SKILL + W_SBERT + W_EXP + W_XGB + W_YEARS + W_CLAUDE - 1.0) < 1e-9

YEARS_CAP = 8

ACTION_VERBS = [
    "developed", "built", "designed", "implemented", "led", "created",
    "migrated", "optimised", "deployed", "managed", "reduced", "improved",
    "automated", "delivered", "shipped", "launched", "architected", "integrated",
    "maintained", "refactored", "scaled", "streamlined", "engineered",
    "optimized", "configured", "established", "spearheaded", "executed",
]