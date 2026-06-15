import json
import pandas as pd
import re

SLANG_TERMS = [
    # Gen Z / internet slang
    "cooked", "dead", "dying", "slay", "bussin", "hits different",
    "lowkey", "highkey", "fr", "fr fr", "no cap", "locked in",
    "delulu", "girl dinner", "snatched", "glow up", "brainrot",
    "fighting for my life", "knocked me out", "had me dying",
    "had me in shambles", "sent me", "took me out",

    # health informal / symptom-like expressions
    "brain fog", "food noise", "crashed", "energy crash", "wired",
    "jittery", "shaky", "heart racing", "bloated", "bloated af",
    "couldn't function", "could barely function", "felt like death",
    "felt like i was dying", "felt awful", "felt terrible",
    "stomach was wrecked", "wrecked my stomach", "messed up my stomach",
    "destroyed my stomach", "couldn't eat", "could not eat",
    "couldn't sleep", "slept all day", "out cold",
    "zombie", "nuked my appetite", "killed my appetite"
]

def find_slang(text):
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    hits = []
    for term in SLANG_TERMS:
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", text_lower):
            hits.append(term)
    return hits

rows = []

for path in ["data_raw/batch_health_broad_01_youtube_videos.jsonl", "data_raw/batch_health_broad_01_youtube_comments.jsonl"]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = " ".join([
                str(item.get("title", "")),
                str(item.get("description", "")),
                str(item.get("text", "")),
                str(item.get("video_title", ""))
            ])
            hits = find_slang(text)
            if hits:
                item["slang_hits"] = hits
                item["source_file"] = path
                rows.append(item)

df = pd.DataFrame(rows)
df.to_csv("data_raw/youtube_candidate_slang.csv", index=False)

print(f"Candidate slang rows: {len(df)}")
print(df[["source_file", "query", "slang_hits"]].head(20))