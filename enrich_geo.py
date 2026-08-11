"""
補齊地理位置資料（創作者 + 留言者，一起處理，共用查詢）。

背景：
- 創作者地理位置：需要頻道的 country 欄位，但收集階段只存了
  channel_title（頻道名稱），沒存 channel_id，所以要先用 videos().list
  backfill 出每支影片對應的 channel_id。
- 留言者地理位置：其實不需要額外收集！author_channel_id 這個欄位
  從一開始收集留言時就有存了，可以直接拿來查。

兩邊最後都是同一件事：「已知一批 channel_id，查它們的 country」，
所以這支腳本把兩邊的 channel_id 合併，一次查完，比分開處理更省 API quota。

用法：
    python enrich_geo.py
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

VIDEO_CHANNEL_LOOKUP_PATH = "data_raw/video_channel_lookup.jsonl"
CHANNEL_GEO_LOOKUP_PATH = "data_raw/channel_geo_lookup.jsonl"
PROGRESS_PATH = "data_raw/enrich_geo_progress.json"


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
    return {"completed_video_batches": 0, "completed_channel_batches": 0}


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


def load_all_commenter_channel_ids() -> List[str]:
    """author_channel_id 從一開始收集留言時就有存了，直接讀出來即可，
    不需要任何額外的 API 呼叫。"""
    channel_ids = set()
    for path in sorted(Path("data_raw").glob("*_youtube_comments.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                cid = item.get("author_channel_id")
                if cid:
                    channel_ids.add(cid)
    return sorted(channel_ids)


def fetch_video_channel_batch(youtube, video_ids_batch: List[str]) -> List[Dict[str, Any]]:
    try:
        resp = youtube.videos().list(part="snippet", id=",".join(video_ids_batch)).execute()
    except HttpError as e:
        if "quota" in str(e).lower():
            raise QuotaExceededError(str(e))
        print(f"videos.list 失敗: {e}")
        return []

    out = []
    for item in resp.get("items", []):
        snippet = item.get("snippet", {})
        out.append({
            "video_id": item.get("id"),
            "channel_id": snippet.get("channelId"),
            "channel_title": snippet.get("channelTitle"),
        })
    return out


def fetch_channel_country_batch(youtube, channel_ids_batch: List[str]) -> List[Dict[str, Any]]:
    try:
        resp = youtube.channels().list(part="snippet", id=",".join(channel_ids_batch)).execute()
    except HttpError as e:
        if "quota" in str(e).lower():
            raise QuotaExceededError(str(e))
        print(f"channels.list 失敗: {e}")
        return []

    out = []
    for item in resp.get("items", []):
        snippet = item.get("snippet", {})
        out.append({
            "channel_id": item.get("id"),
            "channel_title": snippet.get("title"),
            # country 常常是 None，因為這是創作者自己選擇性填寫的欄位，
            # 不是每個頻道都有設定。這個「填寫率低」本身也是一個要在
            # 方法學裡誠實交代的限制。
            "country": snippet.get("country"),
        })
    return out


def main():
    os.makedirs("data_raw", exist_ok=True)
    youtube = get_youtube_client()
    progress = load_progress()

    # ========== 階段一：video_id -> channel_id（只有創作者這側需要）==========
    all_video_ids = load_all_video_ids()
    video_batches = list(chunked(all_video_ids, 50))
    print(f"階段一：{len(all_video_ids)} 支不重複影片，{len(video_batches)} 個 batch")

    start_v = progress["completed_video_batches"]
    if start_v == 0 and not os.path.exists(VIDEO_CHANNEL_LOOKUP_PATH):
        open(VIDEO_CHANNEL_LOOKUP_PATH, "w", encoding="utf-8").close()
    elif start_v > 0:
        print(f"偵測到先前進度：階段一已完成 {start_v}/{len(video_batches)} 個 batch")

    stage1_complete = start_v >= len(video_batches)

    for i in tqdm(range(start_v, len(video_batches)), desc="階段一：video → channel"):
        try:
            records = fetch_video_channel_batch(youtube, video_batches[i])
        except QuotaExceededError as e:
            print(f"\n⚠️ Quota 用盡於階段一，batch {i}：{e}")
            print("已收集的資料都已存檔。直接重新執行這支腳本即可從中斷處繼續。")
            return
        append_jsonl(VIDEO_CHANNEL_LOOKUP_PATH, records)
        progress["completed_video_batches"] = i + 1
        save_progress(progress)
        time.sleep(0.05)
    else:
        stage1_complete = True

    if not stage1_complete:
        return

    print("✅ 階段一完成")

    # ========== 階段二：合併創作者 + 留言者的 channel_id，一起查國家 ==========
    creator_channel_ids = set()
    with open(VIDEO_CHANNEL_LOOKUP_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cid = item.get("channel_id")
            if cid:
                creator_channel_ids.add(cid)

    commenter_channel_ids = set(load_all_commenter_channel_ids())

    all_channel_ids = sorted(creator_channel_ids | commenter_channel_ids)
    channel_batches = list(chunked(all_channel_ids, 50))

    print(f"\n階段二：創作者頻道 {len(creator_channel_ids)} 個、"
          f"留言者頻道 {len(commenter_channel_ids)} 個，"
          f"合併去重後共 {len(all_channel_ids)} 個，{len(channel_batches)} 個 batch")

    start_c = progress["completed_channel_batches"]
    if start_c == 0 and not os.path.exists(CHANNEL_GEO_LOOKUP_PATH):
        open(CHANNEL_GEO_LOOKUP_PATH, "w", encoding="utf-8").close()
    elif start_c > 0:
        print(f"偵測到先前進度：階段二已完成 {start_c}/{len(channel_batches)} 個 batch")

    for i in tqdm(range(start_c, len(channel_batches)), desc="階段二：channel → country"):
        try:
            records = fetch_channel_country_batch(youtube, channel_batches[i])
        except QuotaExceededError as e:
            print(f"\n⚠️ Quota 用盡於階段二，batch {i}：{e}")
            print("階段一的結果已經保留。直接重新執行這支腳本即可從中斷處繼續。")
            return
        append_jsonl(CHANNEL_GEO_LOOKUP_PATH, records)
        progress["completed_channel_batches"] = i + 1
        save_progress(progress)
        time.sleep(0.05)

    print("✅ 全部完成")

    # ========== 總結：分別看創作者 vs 留言者的國家分布 ==========
    from collections import Counter

    country_by_channel = {}
    with open(CHANNEL_GEO_LOOKUP_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            country_by_channel[item["channel_id"]] = item.get("country") or "UNKNOWN"

    creator_countries = Counter(country_by_channel.get(cid, "UNKNOWN") for cid in creator_channel_ids)
    commenter_countries = Counter(country_by_channel.get(cid, "UNKNOWN") for cid in commenter_channel_ids)

    def print_dist(name, counter, total):
        known = total - counter.get("UNKNOWN", 0)
        print(f"\n{name}（總數 {total}，有填國家 {known} 筆，{known/total*100:.1f}%）：")
        for country, n in counter.most_common(10):
            print(f"  {country}: {n} ({n/total*100:.1f}%)")

    print_dist("創作者國家分布", creator_countries, len(creator_channel_ids))
    print_dist("留言者國家分布", commenter_countries, len(commenter_channel_ids))


if __name__ == "__main__":
    main()
