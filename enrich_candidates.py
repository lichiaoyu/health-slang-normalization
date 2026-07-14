import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


INPUT_FILE = Path("data_raw/youtube_candidate_slang_cleaned.csv")
OUTPUT_FILE = Path("data_raw/youtube_candidate_slang_enriched.csv")


analyzer = SentimentIntensityAnalyzer()


SLANG_CATEGORIES = {
    # Health / symptom-like informal expressions
    "brain fog": "informal_symptom",
    "bloated": "informal_symptom",
    "energy crash": "informal_symptom",
    "food noise": "informal_symptom",
    "wired": "informal_symptom",
    "jittery": "informal_symptom",
    "shaky": "informal_symptom",

    # Severe reaction / discomfort
    "dying": "exaggerated_discomfort",
    "dead": "exaggerated_discomfort",
    "fighting for my life": "exaggerated_discomfort",
    "took me out": "exaggerated_discomfort",
    "knocked me out": "exaggerated_discomfort",
    "had me dying": "exaggerated_discomfort",
    "cooked": "exhaustion_or_impairment",

    # Evaluation / approval
    "bussin": "positive_evaluation",
    "hits different": "evaluation_or_intensity",
    "slay": "positive_evaluation",
    "ate": "positive_evaluation",

    # Body image / transformation
    "glow up": "body_transformation",
    "snatched": "body_image",
    "girl dinner": "diet_culture",

    # Stance / emphasis
    "fr": "agreement_or_emphasis",
    "fr fr": "agreement_or_emphasis",
    "no cap": "truth_or_emphasis",
    "lowkey": "stance_or_intensity",
    "highkey": "stance_or_intensity",
    "locked in": "motivation_or_focus",
    "sent me": "humorous_reaction",
}


TARGET_KEYWORDS = {
    "medication": [
        "ozempic", "wegovy", "mounjaro", "zepbound", "semaglutide",
        "tirzepatide", "glp-1", "glp1", "medication", "drug", "dose"
    ],
    "supplement": [
        "supplement", "protein powder", "creatine", "pre workout",
        "pre-workout", "magnesium", "melatonin", "ashwagandha",
        "greens powder", "vitamin"
    ],
    "food_or_meal": [
        "food", "meal", "recipe", "dinner", "breakfast", "lunch",
        "protein", "snack", "calorie", "diet"
    ],
    "body_or_appearance": [
        "body", "weight", "waist", "fat", "skinny", "snatched",
        "glow up", "transformation", "before and after"
    ],
    "fitness_or_workout": [
        "gym", "workout", "exercise", "lifting", "cardio",
        "training", "leg day", "gymtok"
    ],
    "symptom_or_side_effect": [
        "side effect", "nausea", "vomit", "pain", "fatigue",
        "brain fog", "bloating", "constipation", "diarrhea",
        "headache", "dizzy", "anxiety"
    ],
    "creator_or_video": [
        "you", "your video", "this video", "creator", "she", "he",
        "they", "girl", "guy"
    ],
}


DOMAIN_KEYWORDS = {
    "medication_side_effects": [
        "ozempic", "wegovy", "mounjaro", "zepbound", "semaglutide",
        "tirzepatide", "glp1", "glp-1", "side effect", "medication"
    ],
    "supplements": [
        "supplement", "protein powder", "creatine", "pre workout",
        "pre-workout", "magnesium", "melatonin", "ashwagandha",
        "greens powder"
    ],
    "diet_weight_loss": [
        "weight loss", "calorie deficit", "diet", "keto",
        "fasting", "what i eat", "girl dinner", "meal prep"
    ],
    "fitness_gym": [
        "gym", "gymtok", "workout", "fitness", "lifting",
        "leg day", "bulking", "cutting", "body recomp"
    ],
    "gut_health": [
        "gut health", "bloating", "bloated", "ibs", "constipation",
        "digestion", "probiotic", "stomach"
    ],
    "sleep_fatigue": [
        "sleep", "melatonin", "insomnia", "fatigue", "tired",
        "energy crash", "knocked me out"
    ],
    "mental_cognitive": [
        "brain fog", "anxiety", "burnout", "mental health",
        "focus", "adhd", "mood"
    ],
    "body_image_transformation": [
        "glow up", "snatched", "body transformation",
        "before and after", "summer body"
    ],
}


def parse_list(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    text = text.strip("[]")
    if not text:
        return []

    return [
        item.strip().strip("'\"")
        for item in text.split(",")
        if item.strip()
    ]


def row_text(row):
    fields = [
        "text",
        "title",
        "video_title",
        "description",
        "query",
    ]

    values = []

    for field in fields:
        if field in row and pd.notna(row[field]):
            values.append(str(row[field]))

    return " ".join(values)


def extract_hashtags(text):
    return re.findall(r"(?<!\w)#([A-Za-z0-9_]+)", text)


def sentiment_scores(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return pd.Series({
        "sentiment_negative": scores["neg"],
        "sentiment_neutral": scores["neu"],
        "sentiment_positive": scores["pos"],
        "sentiment_compound": compound,
        "sentiment_label": label,
    })


def classify_target(text):
    lower = text.lower()
    scores = {}

    for target, keywords in TARGET_KEYWORDS.items():
        scores[target] = sum(keyword in lower for keyword in keywords)

    best_target = max(scores, key=scores.get)

    if scores[best_target] == 0:
        return "unclear_or_general"

    return best_target


def classify_domain(text):
    lower = text.lower()
    scores = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(keyword in lower for keyword in keywords)

    best_domain = max(scores, key=scores.get)

    if scores[best_domain] == 0:
        return "other_health"

    return best_domain


def classify_slang_category(hits):
    categories = []

    for term in hits:
        normalized = str(term).lower().strip()
        category = SLANG_CATEGORIES.get(normalized, "other_informal_expression")
        categories.append(category)

    return sorted(set(categories))


def get_first_existing(row, names, default=np.nan):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


df = pd.read_csv(INPUT_FILE)

hit_column = (
    "clean_slang_hits"
    if "clean_slang_hits" in df.columns
    else "slang_hits"
)

df["parsed_slang_hits"] = df[hit_column].apply(parse_list)
df["primary_slang"] = df["parsed_slang_hits"].apply(
    lambda values: values[0] if values else "unknown"
)

df["combined_context"] = df.apply(row_text, axis=1)

sentiment_df = df["combined_context"].apply(sentiment_scores)
df = pd.concat([df, sentiment_df], axis=1)

df["sentiment_target"] = df["combined_context"].apply(classify_target)
df["health_domain_enriched"] = df["combined_context"].apply(classify_domain)

df["slang_categories"] = df["parsed_slang_hits"].apply(
    classify_slang_category
)

df["hashtags"] = df["combined_context"].apply(extract_hashtags)
df["hashtag_count"] = df["hashtags"].apply(len)

# Engagement fields: handles different possible column names
df["like_count_enriched"] = df.apply(
    lambda row: get_first_existing(
        row,
        ["like_count", "likes", "comment_like_count", "video_like_count"],
        0,
    ),
    axis=1,
)

df["view_count_enriched"] = df.apply(
    lambda row: get_first_existing(
        row,
        ["view_count", "views", "video_view_count"],
        0,
    ),
    axis=1,
)

df["comment_count_enriched"] = df.apply(
    lambda row: get_first_existing(
        row,
        ["comment_count", "comments_count", "video_comment_count"],
        0,
    ),
    axis=1,
)

df["channel_id_enriched"] = df.apply(
    lambda row: get_first_existing(
        row,
        ["channel_id", "author_channel_id", "creator_id"],
        "",
    ),
    axis=1,
)

df["channel_title_enriched"] = df.apply(
    lambda row: get_first_existing(
        row,
        ["channel_title", "channel_name", "creator_name", "author"],
        "",
    ),
    axis=1,
)

for column in [
    "like_count_enriched",
    "view_count_enriched",
    "comment_count_enriched",
]:
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

df["log_views"] = np.log1p(df["view_count_enriched"])
df["log_likes"] = np.log1p(df["like_count_enriched"])
df["log_comments"] = np.log1p(df["comment_count_enriched"])

df["like_rate"] = np.where(
    df["view_count_enriched"] > 0,
    df["like_count_enriched"] / df["view_count_enriched"],
    np.nan,
)

df["comment_rate"] = np.where(
    df["view_count_enriched"] > 0,
    df["comment_count_enriched"] / df["view_count_enriched"],
    np.nan,
)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Input rows: {len(df)}")
print(f"Saved: {OUTPUT_FILE}")

print("\nSentiment distribution:")
print(df["sentiment_label"].value_counts())

print("\nSentiment targets:")
print(df["sentiment_target"].value_counts())

print("\nHealth domains:")
print(df["health_domain_enriched"].value_counts())

print("\nPrimary slang:")
print(df["primary_slang"].value_counts().head(20))