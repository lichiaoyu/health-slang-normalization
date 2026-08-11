"""
Slang 詞共現分析：同一則留言裡，哪些 slang 詞最常一起出現。

用 clean_candidates.py 產生的 clean_slang_hits 欄位——這個欄位本來就是
find_slang() 對該則留言「所有命中的 slang 詞」的完整清單（不像
primary_slang 只取第一個），所以本來就是共現分析要的資料格式,
不需要重新掃描原始文字。

一樣先做跨 batch 去重（同一個上游問題），避免同一組共現組合因為
留言被重複收集而被算兩次。

用法：
    python analyze_slang_cooccurrence.py
"""

import ast
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = "data_raw/youtube_candidate_slang_cleaned.csv"
OUTPUT_PATH = "data_raw/slang_cooccurrence.csv"


def make_merge_key(df: pd.DataFrame) -> pd.Series:
    """跟 merge_sentiment_with_taxonomy.py / recompute_overall_sentiment_stats.py
    用同一套邏輯，確保「哪些列算重複」的判斷標準一致。"""
    comment_id = df["comment_id"].astype(str) if "comment_id" in df.columns else pd.Series([""] * len(df))
    video_id = df["video_id"].astype(str) if "video_id" in df.columns else pd.Series([""] * len(df))
    is_comment_row = (
        comment_id.notna()
        & (comment_id.str.strip() != "")
        & (comment_id.str.lower() != "nan")
    )
    key = np.where(is_comment_row, "c_" + comment_id, "v_" + video_id)
    return pd.Series(key, index=df.index)


def parse_hits(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []


def remove_nested_hits(hits):
    """
    詞庫裡有些詞彼此是子字串關係（例如 "dying" 是 "had me dying" 的
    一部分,"fr" 是 "fr fr" 的一部分）。find_slang() 逐詞掃描時,只要
    留言裡出現「had me dying」,規則式比對一定會同時命中「dying」
    ——這不是使用者真的用了兩個獨立的詞,是詞庫本身的巢狀關係造成的
    重複計算假象。這裡把「是其他命中詞子字串」的詞去掉,只保留
    比較完整、具體的那個詞,共現分析才不會被這種假象污染。
    """
    unique_hits = list(set(hits))
    result = []
    for term in unique_hits:
        is_nested_in_another = any(
            other != term and term in other for other in unique_hits
        )
        if not is_nested_in_another:
            result.append(term)
    return sorted(result)


def main():
    if not Path(INPUT_PATH).exists():
        raise FileNotFoundError(f"找不到 {INPUT_PATH}，請先跑過 clean_candidates.py")

    df = pd.read_csv(INPUT_PATH)
    n_before = len(df)

    df["_merge_key"] = make_merge_key(df)
    df = df.drop_duplicates(subset="_merge_key", keep="first")
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"⚠️ 發現 {n_dropped} 筆跨 batch 重複留言，已於共現統計前移除。")

    df["parsed_hits"] = df["clean_slang_hits"].apply(parse_hits)
    df["n_hits"] = df["parsed_hits"].apply(len)

    multi_hit = df[df["n_hits"] >= 2]
    print(f"\n總留言數（去重後）: {len(df)}")
    print(f"命中 2 個以上 slang 詞的留言數: {len(multi_hit)} "
          f"({len(multi_hit) / len(df) * 100:.1f}%)")

    if len(multi_hit) == 0:
        print("\n沒有任何留言同時命中兩個以上的 slang 詞，無法做共現分析。")
        return

    pair_counter = Counter()
    n_comments_with_nested_artifact = 0
    for hits in multi_hit["parsed_hits"]:
        cleaned_hits = remove_nested_hits(hits)
        if len(cleaned_hits) < len(set(hits)):
            n_comments_with_nested_artifact += 1
        for pair in combinations(cleaned_hits, 2):
            pair_counter[pair] += 1

    if n_comments_with_nested_artifact > 0:
        print(f"\n⚠️ {n_comments_with_nested_artifact} 筆留言的命中詞裡有巢狀子字串關係")
        print("（例如「dying」是「had me dying」的一部分），已在共現統計前排除，")
        print("否則會把「同一個詞被算兩次」誤報成「兩個不同的詞共現」。")

    rows = [{"term_a": a, "term_b": b, "co_occurrence_count": n} for (a, b), n in pair_counter.items()]
    cooc_df = pd.DataFrame(rows).sort_values("co_occurrence_count", ascending=False)
    cooc_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 60}")
    print(f"Top 20 最常共現的 slang 詞組合")
    print(f"{'=' * 60}")
    print(cooc_df.head(20).to_string(index=False))

    # 每個詞總共跟多少「不同」的詞共現過，可以看出誰是共現網絡裡的樞紐
    partner_count = Counter()
    for (a, b), n in pair_counter.items():
        partner_count[a] += 1
        partner_count[b] += 1

    print(f"\n{'=' * 60}")
    print("共現對象數最多的詞（樞紐程度，不看次數只看跟幾個不同的詞共現過）")
    print(f"{'=' * 60}")
    for term, n_partners in Counter(partner_count).most_common(10):
        print(f"  {term}: 跟 {n_partners} 個不同的詞共現過")

    print(f"\n已存檔: {OUTPUT_PATH}")
    print("\n⚠️ 方法學註記：共現只代表兩個詞出現在同一則留言裡，不代表語意上")
    print("有關聯（例如純粹講很多話、剛好兩個詞都提到）。如果要寫進論文，")
    print("建議挑幾筆高頻共現的留言實際讀一下內容，確認共現背後有沒有")
    print("值得討論的語意關係，而不是只報數字。")


if __name__ == "__main__":
    main()