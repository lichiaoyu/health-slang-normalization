import pandas as pd

FILES = {
    "raw_candidates": "data_raw/youtube_candidate_slang.csv",
    "cleaned_candidates": "data_raw/youtube_candidate_slang_cleaned.csv",
    "balanced_candidates": "data_raw/youtube_candidate_slang_balanced.csv",
}

def load_if_exists(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

for name, path in FILES.items():
    df = load_if_exists(path)
    if df is None:
        print(f"\n{name}: file not found")
        continue

    print("\n" + "=" * 80)
    print(name.upper())
    print("=" * 80)
    print("Rows:", len(df))

    if "source_file" in df.columns:
        print("\nSource distribution:")
        print(df["source_file"].value_counts())

    if "primary_slang" in df.columns:
        print("\nPrimary slang distribution:")
        print(df["primary_slang"].value_counts().head(20))
    elif "clean_slang_hits" in df.columns:
        print("\nClean slang distribution:")
        print(df["clean_slang_hits"].value_counts().head(20))
    elif "slang_hits" in df.columns:
        print("\nRaw slang distribution:")
        print(df["slang_hits"].value_counts().head(20))

    if "health_domain" in df.columns:
        print("\nHealth domain distribution:")
        print(df["health_domain"].value_counts())

    if "text" in df.columns:
        comment_df = df[df["text"].notna()]
        if len(comment_df) > 0:
            comment_df = comment_df.copy()
            comment_df["text_len"] = comment_df["text"].astype(str).str.len()
            print("\nComment text length:")
            print(comment_df["text_len"].describe())

print("\nDone.")