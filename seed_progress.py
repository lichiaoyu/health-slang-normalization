"""
一次性補登腳本。

背景：你先前跑 collect_youtube.py 用的是「沒有 checkpoint 機制」的舊版，
所以 data_raw/ 底下沒有 batch_health_broad_02_progress.json 這個檔案。
如果直接把腳本換成新版（含 Track C + checkpoint）就執行，新版會誤判
「找不到 progress.json = 全新收集」，把你已經收集好的 jsonl 清空、
重新呼叫 API 跑完整 80 + 16 個 query，白白浪費 quota。

這支腳本只做一件事：讀你現有的 jsonl 檔案，反推出「哪些 query 已經
跑過了」，手動寫一份 progress.json，讓新版腳本知道要跳過這 80 個
舊 query，只跑新加的 16 個 Track C query。

完全不呼叫 YouTube API，零 quota 成本。

用法：
    1. 先把 collect_youtube.py 換成最新版（含 Track C）
    2. 執行這支：python seed_progress.py
    3. 確認印出的訊息正確後，再執行：python collect_youtube.py
"""

import json
import importlib.util
from pathlib import Path

SCRIPT_PATH = "collect_youtube.py"  # 你目前的腳本檔名

# 動態載入你的 collect_youtube.py，直接拿它裡面定義的 QUERY_LIST，
# 確保這裡用的 query 清單跟腳本裡的完全一致，不會手動謄寫打錯字。
spec = importlib.util.spec_from_file_location("collect_youtube_module", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

QUERY_LIST = mod.QUERY_LIST
VIDEOS_PATH = mod.VIDEOS_PATH
PROGRESS_PATH = mod.PROGRESS_PATH

# 只把「非 Track C」的 query 標記為已完成 —— 這些是你之前那次已經
# 真正呼叫過 API、跑完的 80 個 query。Track C 是新加的，要留給
# collect_youtube.py 正常執行。
already_run_queries = [q for (q, domain, track) in QUERY_LIST if track != "track_c_rare_slang_boost"]

print(f"從 QUERY_LIST 判斷，之前已經跑過的 query 數量：{len(already_run_queries)}")

# 從現有的 videos jsonl 讀出所有已經收集到的 video_id，
# 讓新版腳本的跨 query 去重機制知道這些影片都看過了。
seen_video_ids = []
videos_path = Path(VIDEOS_PATH)

if not videos_path.exists():
    raise FileNotFoundError(
        f"找不到 {VIDEOS_PATH}，請確認你人在 health_slang_dataset 資料夾底下執行這支腳本，"
        f"且 batch_health_broad_02 的 jsonl 檔案確實存在。"
    )

with open(videos_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        vid = item.get("video_id")
        if vid:
            seen_video_ids.append(vid)

print(f"從 {VIDEOS_PATH} 讀到已收集的影片數量：{len(seen_video_ids)}")

progress = {
    "completed_queries": already_run_queries,
    "seen_video_ids": seen_video_ids,
}

with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
    json.dump(progress, f, ensure_ascii=False)

print(f"\n✅ 已寫入 {PROGRESS_PATH}")
print(f"下次執行 collect_youtube.py 時，會自動跳過這 {len(already_run_queries)} 個已完成的 query，")
print(f"只會處理 Track C 新增的 {len(QUERY_LIST) - len(already_run_queries)} 個 query，不會重打 API。")