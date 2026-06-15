import json
import pandas as pd
from pathlib import Path

files = list(Path("data_raw").glob("*youtube_videos.jsonl")) + list(Path("data_raw").glob("*youtube_comments.jsonl"))

rows = []

for path in files:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("published_at"):
                rows.append({
                    "file": path.name,
                    "published_at": item.get("published_at")
                })

df = pd.DataFrame(rows)

if df.empty:
    print("No timestamp found.")
else:
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    print("Total timestamped rows:", len(df))
    print("Earliest:", df["published_at"].min())
    print("Latest:", df["published_at"].max())

    print("\nBy year:")
    print(df["published_at"].dt.year.value_counts().sort_index())

    print("\nBy file:")
    print(df.groupby("file")["published_at"].agg(["count", "min", "max"]))