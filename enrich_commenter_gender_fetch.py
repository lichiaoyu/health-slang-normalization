"""
補齊留言者的顯示名稱（authorDisplayName），為性別推論做準備。

背景：get_comments() 從一開始收集留言時，只存了 author_channel_id
（一串不可讀的 ID），沒有存 authorDisplayName（使用者顯示的暱稱/名字）。
性別推論需要的是後者。

這裡不用重新跑一次完整的收集流程（重新搜尋影片 + 重新分頁抓留言），
而是直接拿現有留言的 comment_id，用 comments().list 依 ID 批次查詢
（一次最多 50 個 ID），只把缺的欄位補齊。這比重新收集省很多 quota。

用法：
    python enrich_commenter_gender_fetch.py
    （quota 用盡會自動存檔並停下，重新執行會從中斷處繼續）
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise ValueError("Missing YOUTUBE_API_KEY. Please check your .env file.")

AUTHOR_LOOKUP_PATH = "data_raw/comment_author_lookup.jsonl"
PROGRESS_PATH = "data_raw/enrich_commenter_gender_progress.json"


class QuotaExceededError(Exception):
    """quota 用盡時拋出，讓 main() 可以優雅停下並保留已收集的資料。"""
    pass


def get_youtube_client():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def load_progress() -> Dict[str, Any]:
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_batches": 0}


def save_progress(p: Dict[str, Any]) -> None:
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False)


def append_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_all_comment_ids() -> List[str]:
    comment_ids = set()
    for path in sorted(Path("data_raw").glob("*_youtube_comments.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                cid = item.get("comment_id")
                if cid:
                    comment_ids.add(cid)
    return sorted(comment_ids)


def fetch_comment_authors_batch(youtube, comment_ids_batch: List[str]) -> List[Dict[str, Any]]:
    try:
        resp = youtube.comments().list(part="snippet", id=",".join(comment_ids_batch)).execute()
    except HttpError as e:
        if "quota" in str(e).lower():
            raise QuotaExceededError(str(e))
        print(f"comments.list 失敗: {e}")
        return []

    out = []
    for item in resp.get("items", []):
        snippet = item.get("snippet", {})
        out.append({
            "comment_id": item.get("id"),
            "author_display_name": snippet.get("authorDisplayName"),
            "author_channel_id": snippet.get("authorChannelId", {}).get("value"),
        })
    return out


def main():
    os.makedirs("data_raw", exist_ok=True)
    youtube = get_youtube_client()
    progress = load_progress()

    all_ids = load_all_comment_ids()
    batches = list(chunked(all_ids, 50))

    print(f"共 {len(all_ids)} 則不重複留言，{len(batches)} 個 batch")
    print(f"粗估 quota 成本：約 {len(batches)} units（comments.list 每次呼叫只要 1 unit，"
          f"比原本收集用的 search.list 每次 100 units 便宜很多）")

    start_idx = progress["completed_batches"]
    if start_idx == 0 and not os.path.exists(AUTHOR_LOOKUP_PATH):
        open(AUTHOR_LOOKUP_PATH, "w", encoding="utf-8").close()
    elif start_idx > 0:
        print(f"偵測到先前進度：已完成 {start_idx}/{len(batches)} 個 batch，將從中斷處繼續。")

    for i in tqdm(range(start_idx, len(batches)), desc="補齊留言者顯示名稱"):
        try:
            records = fetch_comment_authors_batch(youtube, batches[i])
        except QuotaExceededError as e:
            print(f"\n⚠️ Quota 用盡，batch {i}：{e}")
            print(f"已完成 {i}/{len(batches)} 個 batch，資料已存檔。"
                  f"直接重新執行這支腳本即可從中斷處繼續。")
            return
        append_jsonl(AUTHOR_LOOKUP_PATH, records)
        progress["completed_batches"] = i + 1
        save_progress(progress)
        time.sleep(0.05)

    print("✅ 全部完成")
    print(f"下一步：執行 infer_gender_from_names.py（離線，不用 API）進行性別推論。")


if __name__ == "__main__":
    main()
