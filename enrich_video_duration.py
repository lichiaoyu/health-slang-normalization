"""
補齊每支影片的長度（duration）。

跟 enrich_video_stats.py 是同一個 videos().list endpoint，只是這裡要的是
part="contentDetails"（YouTube API 對 quota 的計價方式是「每次呼叫固定
成本」，不是依 part 數量計費，所以另外開一支腳本不會比把兩個 part 塞進
同一支貴，只是要多跑一輪 API 呼叫）。這裡選擇獨立成一支腳本，是為了
不影響你已經跑完、有 checkpoint 記錄的 enrich_video_stats.py 進度
——如果改寫那支腳本去加 part，會讓它誤判「還沒抓過 contentDetails」
而想要整批重新呼叫。

YouTube 回傳的 duration 是 ISO 8601 格式（例如 PT1M30S 代表 1分30秒），
這裡用簡單的 regex 轉成秒數，不需要額外裝 isodate 套件。

用法：
    python enrich_video_duration.py
    （quota 用盡會自動存檔並停下，重新執行會從中斷處繼續）
"""

import os
import json
import re
import time
import socket
import ssl
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

VIDEO_DURATION_PATH = "data_raw/video_duration_lookup.jsonl"
PROGRESS_PATH = "data_raw/enrich_video_duration_progress.json"

ISO_DURATION_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


class QuotaExceededError(Exception):
    pass


TRANSIENT_EXCEPTIONS = (socket.timeout, ConnectionError, ssl.SSLError, TimeoutError)


def execute_with_retry(request_callable, max_retries=4, base_delay=3):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return request_callable()
        except HttpError as e:
            if "quota" in str(e).lower():
                raise QuotaExceededError(str(e))
            last_exc = e
        except TRANSIENT_EXCEPTIONS as e:
            last_exc = e

        wait = base_delay * (2 ** attempt)
        print(f"\n⚠️ 暫時性網路錯誤（第 {attempt + 1}/{max_retries} 次嘗試）：{last_exc}")
        print(f"{wait} 秒後重試...")
        time.sleep(wait)

    raise last_exc


def get_youtube_client():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def parse_iso8601_duration_to_seconds(duration_str):
    """PT1M30S -> 90。抓不到格式就回傳 None，不要讓整支腳本崩潰。"""
    if not duration_str:
        return None
    m = ISO_DURATION_RE.match(duration_str)
    if not m:
        return None
    hours, minutes, seconds = (int(x) if x else 0 for x in m.groups())
    return hours * 3600 + minutes * 60 + seconds


def duration_bucket(seconds):
    """
    分桶邏輯給後續分析用。YouTube Shorts 原本上限是 60 秒，2024 年之後
    放寬到 3 分鐘，所以這裡用兩層判斷：<=60s 是「傳統定義的 Short」，
    60-180s 是「新版 Shorts 上限內，但不是傳統 Short」，超過 180s
    基本上就是一般影片混進了搜尋結果（你的 query 沒有強制篩選成
    Shorts-only，這點也值得在方法學裡註明）。
    """
    if seconds is None:
        return "unknown"
    if seconds <= 60:
        return "short_under_1min"
    elif seconds <= 180:
        return "short_1_to_3min"
    elif seconds <= 600:
        return "medium_3_to_10min"
    else:
        return "long_over_10min"


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


def fetch_duration_batch(youtube, video_ids_batch: List[str]) -> List[Dict[str, Any]]:
    def _do_request():
        return youtube.videos().list(part="contentDetails", id=",".join(video_ids_batch)).execute()

    resp = execute_with_retry(_do_request)

    out = []
    for item in resp.get("items", []):
        content_details = item.get("contentDetails", {})
        duration_iso = content_details.get("duration")
        seconds = parse_iso8601_duration_to_seconds(duration_iso)
        out.append({
            "video_id": item.get("id"),
            "duration_iso8601": duration_iso,
            "duration_seconds": seconds,
            "duration_bucket": duration_bucket(seconds),
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
    if start_idx == 0 and not os.path.exists(VIDEO_DURATION_PATH):
        open(VIDEO_DURATION_PATH, "w", encoding="utf-8").close()
    elif start_idx > 0:
        print(f"偵測到先前進度：已完成 {start_idx}/{len(batches)} 個 batch，將從中斷處繼續。")

    for i in tqdm(range(start_idx, len(batches)), desc="補齊影片長度"):
        try:
            records = fetch_duration_batch(youtube, batches[i])
        except QuotaExceededError as e:
            print(f"\n⚠️ Quota 用盡，batch {i}：{e}")
            print("已收集的資料都已存檔。直接重新執行這支腳本即可從中斷處繼續。")
            return
        except Exception as e:
            print(f"\n⚠️ batch {i} 重試多次後仍然失敗：{e}")
            print("已收集的資料都已存檔。直接重新執行這支腳本即可從中斷處繼續。")
            return
        append_jsonl(VIDEO_DURATION_PATH, records)
        progress["completed_batches"] = i + 1
        save_progress(progress)
        time.sleep(0.05)

    print("✅ 全部完成")

    import pandas as pd
    df = pd.read_json(VIDEO_DURATION_PATH, lines=True)
    print(f"\n無法解析長度的影片數: {df['duration_seconds'].isna().sum()}")
    print("\n長度分桶分布:")
    print(df["duration_bucket"].value_counts())
    print(f"\n下一步：跑 analyze_video_length.py，把長度資料跟 engagement/slang 資料合併分析。")


if __name__ == "__main__":
    main()