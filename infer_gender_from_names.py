"""
離線性別推論。不呼叫任何 YouTube API。

讀取 enrich_commenter_gender_fetch.py 產生的 comment_author_lookup.jsonl，
用 gender-guesser 套件依「看起來像名字」的部分猜測性別，輸出成 CSV，
之後可以跟主要留言資料（health_domain / primary_slang）合併做交叉分析。

用法：
    pip install gender-guesser --break-system-packages
    python infer_gender_from_names.py
"""

import json
import re
from pathlib import Path

import pandas as pd
import gender_guesser.detector as gender

AUTHOR_LOOKUP_PATH = "data_raw/comment_author_lookup.jsonl"
OUTPUT_PATH = "data_raw/comment_gender_inferred.csv"

d = gender.Detector(case_sensitive=False)

# gender-guesser 原生回傳值：male / mostly_male / female / mostly_female /
# andy（androgynous，中性名）/ unknown。這裡把它收斂成四類，
# "andy" 保留成 ambiguous 而不是強行分進 male/female，避免製造假精確度。
GENDER_MAP = {
    "male": "male",
    "mostly_male": "male",
    "female": "female",
    "mostly_female": "female",
    "andy": "ambiguous",
    "unknown": "unknown",
}


def extract_first_name(display_name):
    """
    YouTube 顯示名稱大多是暱稱/帳號名，不是真實姓名
    （例如 "xXGamerBoy99Xx"、"user8827332"），這裡只做最基本的處理：
    取第一個空白分隔的詞、去掉數字跟符號，嘗試抓出「看起來像名字」的部分。
    抓不到就回傳 None，之後會被歸類成 unknown。

    這一步本身就是這個方法的核心限制：YouTube 顯示名稱的「真實姓名率」
    遠低於 Reddit/一般社群網站，這個推論結果只能當作粗略估計。
    """
    if not display_name or not isinstance(display_name, str):
        return None
    stripped = display_name.strip()
    if not stripped:
        return None
    first_token = stripped.split()[0]
    cleaned = re.sub(r"[^A-Za-z]", "", first_token)
    if len(cleaned) < 2:
        return None
    return cleaned.capitalize()


def main():
    if not Path(AUTHOR_LOOKUP_PATH).exists():
        raise FileNotFoundError(
            f"找不到 {AUTHOR_LOOKUP_PATH}，請先執行 enrich_commenter_gender_fetch.py"
        )

    rows = []
    with open(AUTHOR_LOOKUP_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            display_name = item.get("author_display_name")
            name = extract_first_name(display_name)
            raw_gender = d.get_gender(name) if name else "unknown"
            rows.append({
                "comment_id": item.get("comment_id"),
                "author_channel_id": item.get("author_channel_id"),
                "author_display_name": display_name,
                "extracted_first_name": name,
                "inferred_gender": GENDER_MAP.get(raw_gender, "unknown"),
            })

    gender_df = pd.DataFrame(rows)
    gender_df.to_csv(OUTPUT_PATH, index=False)

    print(f"共處理 {len(gender_df)} 則留言")
    print("\n推論性別分布：")
    print(gender_df["inferred_gender"].value_counts())

    identifiable = (gender_df["inferred_gender"] != "unknown").mean() * 100
    print(f"\n可辨識比例（非 unknown）：{identifiable:.1f}%")
    print(f"已存檔：{OUTPUT_PATH}")

    print("\n⚠️ 重要限制（務必寫進方法學章節）：")
    print("YouTube 顯示名稱大多是暱稱/帳號名而非真實姓名，這個推論結果的")
    print("可信度遠低於使用真實姓名資料庫的情境，只能當作粗略估計，")
    print("不是可靠的人口統計數據。這跟你朋友文件裡對性別推論的限制說明是同一類問題。")


if __name__ == "__main__":
    main()
