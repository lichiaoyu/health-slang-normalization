"""
Normalizer prototype（路線 B 的核心產出）。

給定一則留言（或影片標題/描述，如果是影片列）+ 偵測到的 slang 詞 +
影片標題當脈絡，讓 Claude 做兩件事：
    1. 判斷 literal / nonliteral / ambiguous
    2. 產生一句去掉 slang、白話version的 normalized_meaning

這是論文 Method 章節「Normalization Prototype」小節對應的程式碼。
評估（跟人工標註比對）在 evaluate_normalizer.py 裡做。

重要用法變更：這支腳本現在的輸出不是拿來取代人工標註，而是拿來給
annotator 工具當「草稿建議」——標註者看到 Claude 的判斷後，可以按
「Accept」直接採用，也可以自己重新判斷、覆蓋掉。這樣做的原因寫在
論文 Related Work 裡：Wang et al. (2024, CHI) 的做法是 LLM 先產生
標籤，人類再驗證/覆蓋，而不是直接把 LLM 輸出當成標準答案——如果
標準答案本身就是 LLM 生成的，就沒辦法回答「LLM 判斷得準不準」這個
問題了。

用法：
    pip install anthropic pandas --break-system-packages
    export ANTHROPIC_API_KEY=sk-ant-...
    python normalize_slang.py
"""

import os
import json
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic, APIStatusError, APIConnectionError

MODEL = "claude-sonnet-5"

# 修正：原本這裡指向 data_raw/youtube_annotation_pool.csv，那是還沒
# 跨 batch 去重的舊版（574 筆，其中 27 對是重複收集的留言）。
# 應該指向去重後的版本，否則會浪費 API 額度去預測本來就該被移除的
# 重複列，而且預測結果的 annotation_id 會跟你目前手上乾淨版本的
# annotation_id 對不起來。
INPUT_PATH = "data_raw/youtube_annotation_pool_cleaned.csv"
OUTPUT_PATH = "data_raw/normalizer_predictions.jsonl"
PROGRESS_PATH = "data_raw/normalizer_progress.json"

# 優先跑最需要消歧義的兩類：ambiguous, informal_symptom_language。
# internet_slang 樣本量大、語意相對穩定，先不跑，
# 之後如果想擴大評估範圍再考慮加進來（會產生額外 API 費用）。
PRIORITY_EXPRESSION_TYPES = ["ambiguous", "informal_symptom_language"]

SYSTEM_PROMPT = """You are annotating health-related slang for a research dataset. \
Given a social media comment (or video title/description if no comment is present), \
a detected slang term, and the source video's title for context, do two things:

1. Classify the slang term's use as one of: "literal", "nonliteral", or "ambiguous".
   - "literal": the phrase describes a real physical/medical state or event.
   - "nonliteral": the phrase is hyperbole, humor, or a social/reactive expression \
with no real medical referent.
   - "ambiguous": genuinely unclear even with the given context.
2. Write a short, plain-language paraphrase of what the commenter is actually \
describing, with the slang removed (the "normalized_meaning").

Respond ONLY with a JSON object of this exact shape, no other text:
{"literal_or_nonliteral": "...", "normalized_meaning": "..."}"""


def build_user_prompt(row) -> str:
    text = row.get("text")
    if pd.isna(text) or not str(text).strip():
        text = f"(no comment text; video title/description): {row.get('title', '')} {row.get('description', '')}"
    return (
        f"Slang term: {row.get('primary_slang', '')}\n"
        f"Video title: {row.get('video_title', row.get('title', ''))}\n"
        f"Comment: {text}"
    )


def call_model(client, row, max_retries=4):
    prompt = build_user_prompt(row)
    raw = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"⚠️ 模型回傳非 JSON，略過這筆（annotation_id={row.get('annotation_id')}）: {raw[:150] if raw else ''}")
            return None
        except APIConnectionError as e:
            wait = 3 * (2 ** attempt)
            print(f"⚠️ 網路錯誤（第 {attempt+1}/{max_retries} 次），{wait} 秒後重試: {e}")
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code == 429:
                wait = 5 * (2 ** attempt)
                print(f"⚠️ Rate limit（第 {attempt+1}/{max_retries} 次），{wait} 秒後重試")
                time.sleep(wait)
            else:
                print(f"⚠️ API 錯誤，略過這筆: {e}")
                return None
    print(f"⚠️ 重試多次後仍然失敗，略過這筆（annotation_id={row.get('annotation_id')}）")
    return None


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_ids": []}


def save_progress(p):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    if not Path(INPUT_PATH).exists():
        raise FileNotFoundError(
            f"找不到 {INPUT_PATH}，請確認你已經有去重後的 annotation pool 檔案，"
            f"檔名要是 youtube_annotation_pool_cleaned.csv（547 筆版本），"
            f"不是原本 make_annotation_pool.py 產生的舊版 574 筆檔案。"
        )

    df = pd.read_csv(INPUT_PATH)
    subset = df[df["expression_type"].isin(PRIORITY_EXPRESSION_TYPES)].copy()
    print(f"對 {len(subset)} 筆優先類別資料跑 normalizer（{', '.join(PRIORITY_EXPRESSION_TYPES)}）")

    client = Anthropic()  # 讀取 ANTHROPIC_API_KEY 環境變數

    progress = load_progress()
    completed = set(progress["completed_ids"])

    if not os.path.exists(OUTPUT_PATH):
        open(OUTPUT_PATH, "w", encoding="utf-8").close()

    remaining = subset[~subset["annotation_id"].isin(completed)]
    print(f"已完成: {len(completed)}，剩餘: {len(remaining)}")

    for _, row in remaining.iterrows():
        result = call_model(client, row)
        if result is None:
            continue
        record = {
            "annotation_id": row["annotation_id"],
            "primary_slang": row["primary_slang"],
            "model_literal_or_nonliteral": result.get("literal_or_nonliteral"),
            "model_normalized_meaning": result.get("normalized_meaning"),
        }
        append_jsonl(OUTPUT_PATH, record)
        completed.add(row["annotation_id"])
        progress["completed_ids"] = list(completed)
        save_progress(progress)
        time.sleep(0.3)

    print(f"\n✅ 完成，結果存到 {OUTPUT_PATH}")
    print("下一步：跑 merge_ai_drafts_into_pool.py，把這些預測接進")
    print("annotation pool CSV，變成 annotator 工具裡的草稿建議。")


if __name__ == "__main__":
    main()