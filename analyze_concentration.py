from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("data_raw/youtube_candidate_slang_enriched.csv")
OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

counts = df["primary_slang"].value_counts()
shares = counts / counts.sum()

summary = pd.DataFrame({
    "count": counts,
    "share": shares,
    "cumulative_share": shares.cumsum(),
})

summary.to_csv(OUTPUT_DIR / "slang_concentration.csv")

print("Total cleaned candidates:", len(df))
print("\nTop slang concentration:")
print(summary.head(20))

for n in [1, 3, 5, 10]:
    top_share = shares.head(n).sum()
    print(f"Top {n} slang terms account for {top_share:.1%} of the dataset")

# Herfindahl–Hirschman Index
hhi = np.square(shares).sum()

# Normalized entropy: higher means more diverse
entropy = -(shares * np.log(shares)).sum()
max_entropy = np.log(len(shares))
normalized_entropy = entropy / max_entropy if max_entropy > 0 else np.nan

print(f"\nHHI concentration score: {hhi:.4f}")
print(f"Normalized slang diversity: {normalized_entropy:.4f}")