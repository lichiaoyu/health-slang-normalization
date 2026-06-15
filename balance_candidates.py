import pandas as pd
import ast
import re

INPUT = "data_raw/youtube_candidate_slang_cleaned.csv"
OUTPUT = "data_raw/youtube_candidate_slang_balanced.csv"

# 每個 slang 最多保留幾筆，避免 glow up 淹掉 dataset
MAX_PER_SLANG = {
    "glow up": 80,
    "brain fog": 80,
    "girl dinner": 80,
    "hits different": 80,
    "knocked me out": 80,
    "fighting for my life": 80,
    "cooked": 80,
    "dying": 80,
    "dead": 40,
    "locked in": 60,
    "bussin": 60,
    "fr": 40,
    "lowkey": 40,
    "no cap": 40,
    "snatched": 40,
    "slay": 40,
    "delulu": 40,
}

# 每個 health domain 最多保留幾筆，避免只剩 weight-loss/glow-up
MAX_PER_DOMAIN = {
    "medication_side_effects": 100,
    "supplements": 100,
    "diet_weight_loss": 100,
    "fitness_gym": 100,
    "gut_health": 100,
    "sleep_fatigue": 100,
    "mental_cognitive": 100,
    "body_image_transformation": 100,
    "other_health": 50,
}

def parse_hits(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []

def safe_text(row):
    parts = []
    for col in ["query", "title", "video_title", "description", "text"]:
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]).lower())
    return " ".join(parts)

def classify_domain(text):
    if any(k in text for k in [
        "ozempic", "glp1", "mounjaro", "semaglutide", "wegovy",
        "zepbound", "side effect", "nausea", "medication"
    ]):
        return "medication_side_effects"

    if any(k in text for k in [
        "protein powder", "creatine", "pre workout", "fat burner",
        "greens powder", "ashwagandha", "magnesium", "supplement"
    ]):
        return "supplements"

    if any(k in text for k in [
        "calorie deficit", "what i eat", "weight loss", "keto",
        "fasting", "meal prep", "girl dinner", "diet"
    ]):
        return "diet_weight_loss"

    if any(k in text for k in [
        "gymtok", "gym", "workout", "body recomp", "bulking",
        "cutting", "leg day", "fitness"
    ]):
        return "fitness_gym"

    if any(k in text for k in [
        "gut health", "bloating", "probiotic", "ibs",
        "constipation", "digestion", "detox"
    ]):
        return "gut_health"

    if any(k in text for k in [
        "sleep", "melatonin", "insomnia", "fatigue",
        "energy crash", "knocked me out"
    ]):
        return "sleep_fatigue"

    if any(k in text for k in [
        "brain fog", "anxiety", "burnout", "cortisol",
        "hormone", "pcos", "mood"
    ]):
        return "mental_cognitive"

    if any(k in text for k in [
        "glow up", "snatched", "body transformation",
        "before and after", "summer body"
    ]):
        return "body_image_transformation"

    return "other_health"

def priority_score(row):
    """
    Higher score = more useful for annotation.
    Prioritize:
    - comments over video titles
    - ambiguous health expressions
    - longer context
    """
    text = safe_text(row)
    source = str(row.get("source_file", ""))

    score = 0

    if "comments" in source:
        score += 3

    useful_terms = [
        "brain fog", "fighting for my life", "knocked me out",
        "cooked", "dying", "hits different", "girl dinner",
        "messed me up", "wrecked me", "took me out"
    ]

    for term in useful_terms:
        if term in text:
            score += 2

    # longer comments often give better context
    if len(text) > 150:
        score += 1
    if len(text) > 400:
        score += 1

    # penalize likely noisy transformation spam
    if "glow up" in text and "weight loss" not in text and "fitness" not in text:
        score -= 2

    return score

df = pd.read_csv(INPUT)

hit_col = "clean_slang_hits" if "clean_slang_hits" in df.columns else "slang_hits"
df["parsed_hits"] = df[hit_col].apply(parse_hits)
df["primary_slang"] = df["parsed_hits"].apply(lambda xs: xs[0] if xs else "unknown")
df["combined_text"] = df.apply(safe_text, axis=1)
df["health_domain"] = df["combined_text"].apply(classify_domain)
df["priority_score"] = df.apply(priority_score, axis=1)

# 先按 slang balance
slang_balanced_parts = []
for slang, group in df.groupby("primary_slang"):
    max_n = MAX_PER_SLANG.get(slang, 50)
    group = group.sort_values("priority_score", ascending=False)
    slang_balanced_parts.append(group.head(max_n))

slang_balanced = pd.concat(slang_balanced_parts, ignore_index=True)

# 再按 domain balance
domain_balanced_parts = []
for domain, group in slang_balanced.groupby("health_domain"):
    max_n = MAX_PER_DOMAIN.get(domain, 50)
    group = group.sort_values("priority_score", ascending=False)
    domain_balanced_parts.append(group.head(max_n))

balanced = pd.concat(domain_balanced_parts, ignore_index=True)
balanced = balanced.sort_values(
    ["health_domain", "primary_slang", "priority_score"],
    ascending=[True, True, False]
)

balanced.to_csv(OUTPUT, index=False)

print("Input cleaned rows:", len(df))
print("Balanced rows:", len(balanced))

print("\nBalanced slang distribution:")
print(balanced["primary_slang"].value_counts())

print("\nBalanced health domain distribution:")
print(balanced["health_domain"].value_counts())

print(f"\nSaved: {OUTPUT}")