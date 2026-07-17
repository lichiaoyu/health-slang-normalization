import json
import pandas as pd
import re
from pathlib import Path

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

# 自動抓 data_raw 底下「所有 batch」的 videos/comments jsonl，
# 不用每次收集新一批資料就手動改檔名列表。
paths = sorted(
    list(Path("data_raw").glob("*_youtube_videos.jsonl"))
    + list(Path("data_raw").glob("*_youtube_comments.jsonl"))
)

print(f"Found {len(paths)} source files:")
for p in paths:
    print(f"  {p}")

for path in paths:
    # 檔名格式為 {batch_name}_youtube_videos.jsonl / {batch_name}_youtube_comments.jsonl
    batch_name = path.name.replace("_youtube_videos.jsonl", "").replace("_youtube_comments.jsonl", "")

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
                item["source_file"] = str(path)
                item["batch_name"] = batch_name
                rows.append(item)

df = pd.DataFrame(rows)
df.to_csv("data_raw/youtube_candidate_slang.csv", index=False)

print(f"Candidate slang rows: {len(df)}")
print(df[["source_file", "query", "slang_hits"]].head(20))