"""
補齊每支影片的 engagement 數據（觀看數 / 按讚數 / 留言數）。

收集階段用的是 search().list，只回傳 part="snippet"，不含統計數據。
這裡用 videos().list(part="statistics") 批次補齊，跟 enrich_geo.py
的 videos().list 呼叫成本相近，比 search.list 便宜很多。

用法：
    python enrich_video_stats.py
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

VIDEO_STATS_PATH = "data_raw/video_stats_lookup.jsonl"
PROGRESS_PATH = "data_raw/enrich_video_stats_progress.json"


class QuotaExceededError(Exception):
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


def load_all_video_ids() -> List[str]:
    video_ids = set()
    for path in sorted(Path("data_raw").glob("*_youtube_videos.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                vid = item.get("video_id")
                if vid:
                    video_ids.add(vid)
    return sorted(video_ids)


def fetch_stats_batch(youtube, video_ids_batch: List[str]) -> List[Dict[str, Any]]:
    try:
        resp = youtube.videos().list(part="statistics", id=",".join(video_ids_batch)).execute()
    except HttpError as e:
        if "quota" in str(e).lower():
            raise QuotaExceededError(str(e))
        print(f"videos.list(statistics) 失敗: {e}")
        return []

    out = []
    for item in resp.get("items", []):
        stats = item.get("statistics", {})
        out.append({
            "video_id": item.get("id"),
            # 用 int() 轉型前先確認欄位存在，有些影片會關閉讚數/留言數顯示
            "view_count": int(stats["viewCount"]) if "viewCount" in stats else None,
            "like_count": int(stats["likeCount"]) if "likeCount" in stats else None,
            "comment_count": int(stats["commentCount"]) if "commentCount" in stats else None,
        })
    return out


def main():
    os.makedirs("data_raw", exist_ok=True)
    youtube = get_youtube_client()
    progress = load_progress()

    all_video_ids = load_all_video_ids()
    batches = list(chunked(all_video_ids, 50))
    print(f"共 {len(all_video_ids)} 支不重複影片，{len(batches)} 個 batch")

    start_idx = progress["completed_batches"]
    if start_idx == 0 and not os.path.exists(VIDEO_STATS_PATH):
        open(VIDEO_STATS_PATH, "w", encoding="utf-8").close()
    elif start_idx > 0:
        print(f"偵測到先前進度：已完成 {start_idx}/{len(batches)} 個 batch，將從中斷處繼續。")

    for i in tqdm(range(start_idx, len(batches)), desc="補齊影片 engagement 數據"):
        try:
            records = fetch_stats_batch(youtube, batches[i])
        except QuotaExceededError as e:
            print(f"\n⚠️ Quota 用盡，batch {i}：{e}")
            print("已收集的資料都已存檔。直接重新執行這支腳本即可從中斷處繼續。")
            return
        append_jsonl(VIDEO_STATS_PATH, records)
        progress["completed_batches"] = i + 1
        save_progress(progress)
        time.sleep(0.05)

    print("✅ 全部完成")

    import pandas as pd
    df = pd.read_json(VIDEO_STATS_PATH, lines=True)
    print(f"\n平均觀看數: {df['view_count'].mean():.0f}")
    print(f"平均按讚數: {df['like_count'].mean():.0f}")
    print(f"平均留言數: {df['comment_count'].mean():.0f}")
    print(f"\n下一步：可以把這份資料跟 collection_domain 合併，比較各 domain 的 engagement 差異。")


if __name__ == "__main__":
    main()
