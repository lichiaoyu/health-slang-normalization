import pandas as pd
import re

INPUT = "data_raw/youtube_candidate_slang.csv"
OUTPUT = "data_raw/youtube_candidate_slang_cleaned.csv"

df = pd.read_csv(INPUT)

SLANG_TERMS = [
    "dead", "dying", "slay", "bussin",
    "hits different", "lowkey", "highkey", "fr", "fr fr",
    "no cap", "locked in", "delulu", "girl dinner", "snatched",
    "glow up", "brain fog", "fighting for my life",
    "knocked me out", "had me dying", "had me in shambles",
    "sent me", "took me out", "messed me up", "wrecked me"
]

HEALTH_TERMS = [
    "ozempic", "glp1", "semaglutide", "mounjaro", "zepbound", "wegovy",
    "side effect", "nausea", "bloating", "gut", "probiotic", "ibs",
    "melatonin", "sleep", "protein", "pre workout", "creatine",
    "calorie", "deficit", "diet", "weight loss", "fat loss",
    "gym", "workout", "supplement", "magnesium", "girl dinner",
    "gummies", "sleep", "bulk", "cutting", "fitness", "bodybuilding"
]

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def row_text_for_slang(row):
    """
    Important:
    - For comment rows, slang must appear in the comment text itself.
      Otherwise the row may be a false hit from video_title.
    - For video rows, use title + description.
    """
    source = safe_str(row.get("source_file", ""))

    if "comments" in source:
        return safe_str(row.get("text", "")).lower()

    return " ".join([
        safe_str(row.get("title", "")),
        safe_str(row.get("description", ""))
    ]).lower()

def row_text_for_health(row):
    """
    Health relevance can come from video title/query/comment combined.
    """
    return " ".join([
        safe_str(row.get("query", "")),
        safe_str(row.get("title", "")),
        safe_str(row.get("description", "")),
        safe_str(row.get("video_title", "")),
        safe_str(row.get("text", ""))
    ]).lower()

def find_slang(text):
    hits = []
    for term in SLANG_TERMS:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, text):
            hits.append(term)
    return hits

def is_health_related(text):
    return any(term in text for term in HEALTH_TERMS)

def is_false_ate(text):
    false_patterns = [
        r"\bi ate\b",
        r"\bwhat i ate\b",
        r"\bway i ate\b",
        r"\bate a lot\b",
        r"\bate protein\b",
        r"\beat right\b",
        r"\bpreviously ate\b",
        r"\bchanged the way i ate\b",
    ]
    return any(re.search(p, text) for p in false_patterns)

def is_false_dead(text):
    # Keep humorous/slang dead only.
    if "💀" in text or "dead 😂" in text or "dead lol" in text or "i'm dead" in text or "im dead" in text:
        return False
    return True

def is_likely_slang(row):
    slang_text = row_text_for_slang(row)
    health_text = row_text_for_health(row)

    slang_hits = find_slang(slang_text)

    if not slang_hits:
        return False

    if not is_health_related(health_text):
        return False

    if "ate" in slang_hits and is_false_ate(slang_text):
        return False

    if "dead" in slang_hits and is_false_dead(slang_text):
        return False

    return True

df["slang_text_checked"] = df.apply(row_text_for_slang, axis=1)
df["health_text_checked"] = df.apply(row_text_for_health, axis=1)
df["clean_slang_hits"] = df["slang_text_checked"].apply(find_slang)
df["is_health_related"] = df["health_text_checked"].apply(is_health_related)
df["keep_for_annotation"] = df.apply(is_likely_slang, axis=1)

cleaned = df[df["keep_for_annotation"]].copy()
cleaned.to_csv(OUTPUT, index=False)

print("Original candidates:", len(df))
print("Cleaned candidates:", len(cleaned))

print("\nSource distribution after cleaning:")
print(cleaned["source_file"].value_counts())

print("\nClean slang distribution after cleaning:")
print(cleaned["clean_slang_hits"].value_counts())

cols = [c for c in [
    "source_file", "query", "clean_slang_hits",
    "title", "video_title", "text"
] if c in cleaned.columns]

print("\nPreview:")
print(cleaned[cols].head(40).to_string())